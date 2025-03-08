import inspect
import logging
from functools import partial
from typing import Any, Callable, Protocol, cast

import torch
from sbi.inference import NPE, SNPE, DirectPosterior
from sbi.utils import RestrictedPrior, get_density_thresholder
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from simglucose.simulation.sim_engine import SimObj

from glucose_sbi.glucose_simulator import (
    run_glucose_simulator,
)
from glucose_sbi.prepare_priors import (
    InferredParams,
    Prior,
)

script_logger = logging.getLogger("sbi_logger.sbi_api")


class Distribution(Protocol):
    """Abstract class for distributions. A distribution is an object that can be sampled from."""

    def sample(
        self, sample_shape: tuple[int, ...], x: torch.Tensor | None
    ) -> torch.Tensor: ...
    def event_shape(self) -> tuple[int, ...]: ...


def get_simulation_params(
    simulation_object: SimObj, inferred_params: InferredParams
) -> dict:
    """Returns the patient parameters that were used in the simulation.

    Parameters
    ----------
    simulation_object : SimObj
        simglucose simulation object
    inferred_params : InferredParams
        dataclass containing the names of the inferred parameters

    Returns
    -------
    dict
        dictionary containing the patient parameters used in the simulation

    """
    params = [
        getattr(simulation_object.env.patient._params, param)  # noqa: SLF001
        for param in inferred_params.params_names
        if "meal" not in param
    ]
    params.extend([simulation_object.env.scenario.scenario[i][1] for i in range(5)])
    param_names = inferred_params.params_names + ["meal_" + str(i) for i in range(5)]
    return dict(zip(param_names, params))


def _check_prior_for_meals(prior: Prior) -> bool:
    """Check if the prior contains meal parameters."""
    return any("meal" in param for param in prior.params_names)


def set_up_sbi_simulator(
    prior: Prior,
    default_simulation_object: SimObj,
    glucose_simulator: Callable[..., torch.Tensor],
    *,
    device: torch.device,
    infer_meal_params: bool = False,
) -> Callable:
    """Sets up and checks the simulator for the Simulation Bayesian Inference (SBI) framework.

    Parameters
    ----------
    prior : Prior
        Dataclass object containing the prior distribution of the parameters, its type and
        the names of the inferred parameters.
    default_simulation_object : SimObj
        The simulation object with all default presets like meal scenario and patient parameters, ready to `.simulate`
    glucose_simulator : callable
        Function that runs the 'simglucose' simulator
    device : torch.device
        Device used to run the inference process on: "cpu" or "cuda"
    infer_meal_params : bool, optional
        Whether to infer meal parameters, by default False

    Returns
    -------
    callable
        The SBI simulator function used to infer the parameters

    """
    if infer_meal_params and not _check_prior_for_meals(prior):
        script_logger.warning(
            "The prior distribution is missing meal parameters despite infer_meal_params being True"
        )

    processed_priors, _, _ = process_prior(prior.params_prior_distribution)
    script_logger.info(
        "Using prior distribution of shape: %s", processed_priors.event_shape
    )
    wrapper = partial(
        glucose_simulator,
        default_simulation_object=default_simulation_object,
        inferred_params=InferredParams(prior.params_names),
        device=device,
        infer_meal_params=infer_meal_params,
    )

    sbi_simulator = process_simulator(
        wrapper, processed_priors, is_numpy_simulator=True
    )

    check_sbi_inputs(sbi_simulator, processed_priors)

    return sbi_simulator


def sample_positive(
    distribution: Distribution,
    num_samples: int,
    x_true: torch.Tensor | None = None,
    *,
    batch_size: int | None = None,  # Adjustable batch size for efficiency
) -> torch.Tensor:
    """Samples only positive values from a distribution using the `.sample` method
    of the distribution object.

    Parameters
    ----------
    distribution : Distribution
        The distribution object to sample from. Can belong to different classes of sbi or torch.distributions.
    num_samples : int
        The number of positive samples to generate.
    x_true : torch.Tensor, optional
        The conditioning observation, if applicable.
    batch_size : int, optional
        The number of samples to draw in each batch to improve efficiency.

    Returns
    -------
    torch.Tensor
        The tensor of positive samples of shape (num_samples, num_params).

    """
    if not batch_size:
        batch_size = num_samples // 10

    collected = []
    # Determine if x should be passed
    sample_params = inspect.signature(distribution.sample).parameters

    kwargs: dict[str, Any] = {}
    if "x" in sample_params:
        kwargs["x"] = x_true
    if "show_progress_bars" in sample_params:
        kwargs["show_progress_bars"] = False

    total_collected = 0.0
    last_logged_pct = 0.0

    while total_collected < num_samples:
        batch_samples = distribution.sample((batch_size,), **kwargs)

        positive_samples = batch_samples[torch.all(batch_samples > 0, dim=1)]

        collected.append(positive_samples)
        total_collected = sum(t.shape[0] for t in collected)

        # Report progress every 10%
        pct_complete = min(total_collected / num_samples * 100, 100)

        milestone = 10.0
        if pct_complete - last_logged_pct >= milestone:
            script_logger.info("Collected %.2f%% of positive samples", pct_complete)
            last_logged_pct = pct_complete  # Update last logged percentage

    # Concatenate and return exactly num_samples
    return torch.cat(collected, dim=0)[:num_samples]


def bayes_flow(
    prior_distribution: Distribution,
    simulator: Callable,
    num_simulations: int,
    *,
    sampling_method: str = "direct",
    device: torch.device,
) -> DirectPosterior:
    """Runs the BayesFlow algorithm (single round of NPE).

    Parameters
    ----------
    prior_distribution : Distribution
        The prior distribution for the parameters to infer.
    simulator : Callable
        The sbi simulator function that generates the data and runs the simulation.
    num_simulations : int
        The number of simulations to run.
    sampling_method : str, optional
        The sampling method used to build the posterior: ["direct" | "mcmc" | "rejection" | "vi"], by default "mcmc"
    device : torch.device
        The device to use for the inference

    Returns
    -------
    DirectPosterior
        The posterior distribution of inferred parameters.

    """
    script_logger.info(
        "Running BayesFlow inference on prior of shape: %s",
        prior_distribution.event_shape,
    )
    inference = NPE(prior=prior_distribution, device=device)
    theta = sample_positive(prior_distribution, num_simulations)
    x = simulator(theta)
    theta = theta.to(device)
    x = x.to(device)
    inference.append_simulations(theta, x).train()
    return inference.build_posterior(sample_with=sampling_method)


def tsnpe(
    prior_distribution: Distribution,
    simulator: Callable,
    true_observation: torch.Tensor,
    *,
    device: torch.device,
    sample_proposal_with: str = "rejection",
    sampling_method: str = "direct",
    num_rounds: int = 10,
    num_simulations: int = 1000,
) -> Distribution:
    """Runs the Truncated Sequential Neural Posterior Estimation (TSNPE) algorithm.

    Parameters
    ----------
    prior_distribution : Distribution
        The prior distribution of the parameters to infer.
    simulator : callable
        The sbi simulator function that generates the data and runs the simulation.
    true_observation : torch.Tensor
        The true observation to compare the inference results to.
    device : torch.device
        The device to use for the inference process.
    sample_proposal_with : str, optional
        The sampling method to sample from the proposal distribution: ["rejection" | "sir"], by default "rejection"
    sampling_method : str, optional
        The sampling method used to build the posterior: ["direct" | "mcmc" | "rejection" | "vi"], by default "mcmc"
    num_rounds : int, optional
        number  of inference rounds, by default 10
    num_simulations : int, optional
        number of simulations per inference round, by default 1000

    Returns
    -------
    Distribution
        The posterior distribution of the inferred parameters.

    """
    script_logger.info(
        "Running TSNPE inference with %s proposal sampling", sample_proposal_with
    )
    inference = SNPE(prior=prior_distribution, device=device)
    proposal = prior_distribution
    for r in range(num_rounds):
        script_logger.info("Running round %s of %s ..., ", r + 1, num_rounds)
        theta = sample_positive(proposal, num_samples=num_simulations)
        script_logger.info("Simulating theta of shape: %s", theta.shape)
        x = simulator(theta)
        theta = theta.to(device)
        x = x.to(device)

        _ = inference.append_simulations(theta, x).train(force_first_round_loss=True)
        posterior = inference.build_posterior(
            sample_with=sampling_method
        ).set_default_x(true_observation)

        accept_reject_fn = get_density_thresholder(posterior, quantile=1e-4)
        proposal = RestrictedPrior(
            prior_distribution,
            accept_reject_fn,
            sample_with=sample_proposal_with,
            device=device,
            posterior=posterior if sample_proposal_with == "sir" else None,
        )

    return posterior


def apt(
    prior_distribution: Distribution,
    simulator: Callable,
    true_observation: torch.Tensor,
    *,
    device: torch.device,
    sampling_method: str = "mcmc",
    num_rounds: int = 10,
    num_simulations: int = 1000,
) -> DirectPosterior:
    """Runs the Automatic Posterior Transformation (APT) NPE algorithm.

    Parameters
    ----------
    prior_distribution : Distribution
        The prior distribution of the parameters to infer.
    simulator : callable
        The sbi simulator function that generates the data and runs the simulation.
    true_observation : torch.Tensor
        The true observation to compare the inference results to.
    device : torch.device
        The device to use for the inference process.
    sampling_method : str, optional
        The sampling method used to build the posterior: ["direct" | "mcmc" | "rejection" | "vi"], by default "mcmc"
    num_rounds : int, optional
        number  of inference rounds, by default 10
    num_simulations : int, optional
        number of simulations per inference round, by default 1000

    Returns
    -------
    Distribution
        The posterior distribution of the inferred parameters.

    """
    script_logger.info(
        "Running APT inference on prior of shape: %s", prior_distribution.event_shape
    )

    inference = SNPE(prior=prior_distribution, device=device)

    proposal = prior_distribution  # start with prior

    for r in range(num_rounds):
        script_logger.info("Running round %s of %s", r + 1, num_rounds)

        theta = sample_positive(proposal, num_simulations)
        x = simulator(theta)
        theta = theta.to(device)
        x = x.to(device)

        _ = inference.append_simulations(theta, x, proposal=proposal).train()

        posterior_dist = inference.build_posterior(
            sample_with=sampling_method
        ).set_default_x(true_observation)

        proposal = posterior_dist

    return posterior_dist


def run_npe(
    algorithm: str,
    true_observation: torch.Tensor,
    prior_distribution: Distribution,
    sampling_method: str,
    sampling_proposal_with: str,
    simulator: Callable,
    device: torch.device,
    num_rounds: int = 10,
    num_simulations: int = 1000,
) -> Distribution:
    """Run the specified NPE algorithm.

    Parameters
    ----------
    algorithm : str
        The name of the NPE algorithm to run.
    true_observation : torch.Tensor
        The true observation to compare the inference results to.
    prior_distribution,
        The prior distribution of the parameters to infer.
    sampling_method : str
        The sampling method to use in `build_posterior` method.
    sampling_proposal_with : str
        The sampling method to sample from the proposal distribution (TSNPE only).
    simulator : Callable
        The simulator function that generates the data.
    device : torch.device
        The device to use for the simulation.
    num_rounds : int, optional
        The number of inference rounds, by default 10
    num_simulations : int, optional
        The number of simulations per inference round, by default 1000

    Returns
    -------
    Distribution
        The posterior distribution of the parameters.

    """
    if algorithm == "TSNPE":
        return tsnpe(
            prior_distribution=prior_distribution,
            simulator=simulator,
            true_observation=true_observation,
            device=device,
            sample_proposal_with=sampling_proposal_with,
            sampling_method=sampling_method,
            num_rounds=num_rounds,
            num_simulations=num_simulations,
        )
    if algorithm == "APT":
        return apt(
            prior_distribution=prior_distribution,
            simulator=simulator,
            device=device,
            sampling_method=sampling_method,
            true_observation=true_observation,
            num_rounds=num_rounds,
            num_simulations=num_simulations,
        )
    if algorithm == "BayesFlow":
        return bayes_flow(
            prior_distribution=prior_distribution,
            simulator=simulator,
            num_simulations=num_simulations,
            sampling_method=sampling_method,
            device=device,
        )
    msg = f"Invalid NPE algorithm: {algorithm}"
    raise ValueError(msg)


def run_inference(
    prior: Prior,
    config: dict,
    device: torch.device,
    default_simulation_object: SimObj,
    true_observation: torch.Tensor,
) -> Distribution:
    """Run the parameter inference process."""
    sbi_settings = config["sbi_settings"]
    sbi_simulator = set_up_sbi_simulator(
        prior=prior,
        default_simulation_object=default_simulation_object,
        glucose_simulator=run_glucose_simulator,
        device=device,
        infer_meal_params=config["infer_meal_params"],
    )
    return run_npe(
        algorithm=sbi_settings["algorithm"],
        simulator=sbi_simulator,
        true_observation=true_observation,
        prior_distribution=cast(Distribution, prior.params_prior_distribution),
        sampling_method=sbi_settings.get("sampling_method", "direct"),
        sampling_proposal_with=sbi_settings.get("sample_proposal_with", "rejection"),
        device=device,
        num_rounds=sbi_settings["num_rounds"],
        num_simulations=sbi_settings["num_simulations"],
    )
