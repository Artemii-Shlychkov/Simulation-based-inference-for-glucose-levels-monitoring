import argparse
import json
import logging
from collections.abc import Generator
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Callable, Protocol

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sbi.inference import (
    NPE,
    DirectPosterior,
)
from sbi.utils import RestrictedPrior, get_density_thresholder
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from simglucose.simulation.env import T1DSimEnv
from sklearn.metrics import mean_squared_error
from torch.distributions import Distribution

from glucose_sbi.glucose_simulator import (
    DeafultSimulationEnv,
    load_default_simulation_env,
    run_glucose_simulator,
)
from glucose_sbi.prepare_priors import Prior, prepare_prior
from glucose_sbi.process_results import plot_simulation
from glucose_sbi.sample_non_negative import sample_non_negative


class Posterior(Protocol):
    def sample(
        self, sample_shape: tuple[int, ...], x: torch.Tensor
    ) -> torch.Tensor: ...


def set_up_logging(saving_path: Path) -> logging.Logger:
    """Set up the logging configuration for the script."""
    logger = logging.getLogger("sbi_logger")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(Path(saving_path, "inference_execution.log"))
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def load_config(script_dir: Path, config_name: "str") -> dict:
    """Loads the configuration file.

    Parameters
    ----------
    script_dir : Path
        The path to the script directory.
    config_name : str
        The name of the configuration file to load.

    Returns
    -------
    dict
        The configuration file as a dictionary.

    """
    with Path(script_dir / "simulation_configs" / config_name).open("r") as file:
        script_logger.info("Loaded configuration file: %s", config_name)
        return yaml.safe_load(file)


def set_up_device() -> torch.device:
    """Set up the device for the simulation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_logger.info("Device used for execution: %s", device)
    return device


def get_patient_params(env: T1DSimEnv, prior: Prior) -> dict:
    """Returns the patient parameters that were used in the simulation.

    Parameters
    ----------
    env : T1DSimEnv
        simglucose simulation environment
    prior : Prior
        dataclass containing the priors for the patient parameters
        (we only want to look at the parameters that are being inferred)

    Returns
    -------
    dict
        dictionary containing the patient parameters used in the simulation

    """
    param_names = prior.params_names
    params = [
        getattr(env.env.patient._params, param)  # noqa: SLF001
        for param in param_names
    ]
    return dict(zip(param_names, params))


def set_up_sbi_simulator(
    prior: Prior,
    default_settings: DeafultSimulationEnv,
    device: torch.device,
    glucose_simulator: Callable[..., torch.Tensor],
) -> Callable:
    """Sets up and checks the simulator for the Sequential Bayesian Inference (SBI) framework.

    Parameters
    ----------
    default_settings : DeafultSimulationEnv
        DataClass object containing the default simulation environment settings.
    prior : Prior
        DataClass object containing the priors for the parameters
    device : torch.device
        Device used to run the simulation
    glucose_simulator : callable, optional
        Function that runs the glucose simulator, by default run_glucose_simulator
    processed_priors : Distribution, optional
        Processed priors for the parameters, by default None

    Returns
    -------
    callable
        The SBI simulator function used to infer the parameters

    """
    processed_priors, _, _ = process_prior(prior.params_prior_distribution)
    script_logger.info(
        "Using prior distribution of shape: %s", processed_priors.event_shape
    )
    wrapper = partial(
        glucose_simulator,
        default_settings=default_settings,
        inferred_params=prior,
        device=device,
        logger=script_logger,
    )

    sbi_simulator = process_simulator(
        wrapper, processed_priors, is_numpy_simulator=True
    )

    check_sbi_inputs(sbi_simulator, processed_priors)

    return sbi_simulator


def get_true_observation(
    prior: Prior, env_settings: DeafultSimulationEnv, hours: int = 24
) -> tuple[torch.Tensor, dict]:
    """Returns the single glucose dynamcis simulation from the default simulation environment parameters and these parameters.

    Parameters
    ----------
    prior : Prior
        DataClass object containing the priors for the parameters
    env_settings : DeafultSimulationEnv
        DataClass object containing the default simulation environment settings.
    hours : int, optional
        Duration of the simulation, by default 24

    Returns
    -------
    np.ndarray
        Time series of glucose dynamics.

    """
    default_simulation_env = load_default_simulation_env(
        env_settings=env_settings, hours=hours
    )
    default_simulation_env.simulate()
    true_params = get_patient_params(default_simulation_env, prior)
    true_observation = default_simulation_env.results()["CGM"].to_numpy()
    true_observation = torch.from_numpy(true_observation).float().to(device)
    return true_observation, true_params


def positive_sample_generator(
    distribution: Distribution,
) -> Generator[torch.Tensor, None, None]:
    """Generates all-positive samples from a distribution that has a `sample` method.

    Parameters
    ----------
    distribution : Distribution | Posterior
        The distribution to sample from.

    Yields
    ------
    Generator[torch.Tensor, None, None]
        The generated sample.

    """
    while True:
        sample = distribution.sample()
        if torch.all(sample > 0):
            yield sample


def sample_positive(distribution: Distribution, num_samples: int) -> torch.Tensor:
    """Samples positive values from a distribution.

    Parameters
    ----------
    distribution : Distribution | Posterior
        The distribution to sample from.
    num_samples : int
        The number of samples to generate.

    Returns
    -------
    torch.Tensor
        The tensor of positive samples of shape (num_samples, num_params)

    """
    gen = positive_sample_generator(distribution)
    collected: list[torch.Tensor] = []

    while len(collected) < num_samples:
        collected.append(next(gen))
        # report every 10% of the samples
        step = num_samples // 10
        if len(collected) % step == 0:
            pct_complete = len(collected) / num_samples * 100
            script_logger.info("Collected %s %% of positive samples", pct_complete)

    return torch.stack(collected)


def sample_from_posterior(
    posterior: Posterior,
    x_true: np.ndarray,
    num_samples: int = 1000,
    *,
    only_non_negative: bool = True,
) -> torch.Tensor:
    """Sample from the posterior distribution.

    Parameters
    ----------
    posterior : Distribution
        The posterior distribution of the parameters.
    x_true : np.ndarray
        The true observation to compare the inference results to.
    num_samples : int, optional
        number of required samples, by default 1000
    only_non_negative : bool, optional
        Whether to sample only non-negative parameters, by default True
        If true, will call the sample_non_negative function from glucose_sbi.sample_non_negative script

    Returns
    -------
    torch.Tensor
        The posterior samples.

    """
    if only_non_negative:
        return sample_non_negative(
            posterior, num_samples=num_samples, true_observation=torch.tensor(x_true)
        )
    return posterior.sample(
        sample_shape=(num_samples,),
        x=torch.tensor(x_true, dtype=torch.float32, device=device),
    )


def bayes_flow(
    prior: Distribution, simulator: Callable, num_sims: int
) -> DirectPosterior:
    """Run the BayesFlow algorithm (single round of NPE).

    Parameters
    ----------
    prior : Distribution
        The prior distribution for the parameters to infer.
    simulator : Callable
        The simulator function that generates the data.
    num_sims : int
        The number of simulations to run.

    Returns
    -------
    DirectPosterior
        The posterior distribution of the parameters.

    """
    script_logger.info(
        "Running BayesFlow inference on prior of shape: %s", prior.event_shape
    )
    inference = NPE(prior=prior, device=device)
    theta = sample_positive(prior, num_sims)
    x = simulator(theta)
    theta = theta.to(device)
    x = x.to(device)
    inference.append_simulations(theta, x).train()
    return inference.build_posterior()


def tsnpe(
    prior: Distribution,
    simulator: Callable,
    true_observation: torch.Tensor,
    device: torch.device,
    sampling_method: str,
    num_rounds: int = 10,
    num_simulations: int = 1000,
) -> DirectPosterior:
    """Runs the Truncated Sequential Neural Posterior Estimation (TSNPE) algorithm.

    Parameters
    ----------
    prior : Distribution
        The prior distribution for the parameters to infer.
    simulator : callable
        The simulator function that generates the data.
    true_observation : torch.Tensor
        The true observation to compare the inference results to.
    device : torch.device
        The device to use for the simulation.
    sampling_method : str
        The sampling method to use.
    num_rounds : int, optional
        number  of inference rounds, by default 10
    num_simulations : int, optional
        number of simulations per inferenceround, by default 1000

    Returns
    -------
    Posterior
        The posterior distribution of the parameters.

    """
    script_logger.info(
        "Running TSNPE inference on prior of shape: %s", prior.event_shape
    )

    inference = NPE(prior=prior, device=device)
    proposal = prior
    for r in range(num_rounds):
        script_logger.info("Running round %s of %s", r + 1, num_rounds)

        theta = sample_positive(proposal, num_simulations)
        script_logger.info("Simulating theta of shape: %s", theta.shape)
        x = simulator(theta)
        # Optional sanity check: ensure on same device
        theta = theta.to(device)
        x = x.to(device)

        _ = inference.append_simulations(theta, x).train(force_first_round_loss=True)
        posterior = inference.build_posterior().set_default_x(true_observation)

        accept_reject_fn = get_density_thresholder(posterior, quantile=1e-4)
        proposal = RestrictedPrior(
            prior,
            accept_reject_fn,
            sample_with=sampling_method,
            device=device,
            posterior=posterior if sampling_method == "sir" else None,
        )

    return posterior


def apt(
    prior: Distribution | DirectPosterior,
    simulator: Callable,
    true_observation: torch.Tensor,
    device: torch.device,
    num_rounds: int = 10,
    num_simulations: int = 1000,
) -> DirectPosterior:
    """Runs the Automatic Posterior Transformation (APT) NPE algorithm.

    Parameters
    ----------
    prior : Distribution | DirectPosterior
        The prior distribution for the parameters to infer.
    simulator : callable
        The simulator function that generates the data.
    true_observation : torch.Tensor
        The true observation to compare the inference results to.
    device : torch.device
        The device to use for the simulation.
    num_rounds : int, optional
        number of inference rounds, by default 10
    num_simulations : int, optional
        number of simulations per inference round, by default 1000

    Returns
    -------
    Distribution
        The posterior distribution of the parameters.

    """
    script_logger.info("Running APT inference on prior of shape: %s", prior.event_shape)

    inference = NPE(prior=prior, device=device)

    proposal = prior  # start with prior

    for r in range(num_rounds):
        script_logger.info("Running round %s of %s", r + 1, num_rounds)

        theta = proposal.sample((num_simulations,))
        x = simulator(theta)
        theta = theta.to(device)
        x = x.to(device)

        _ = inference.append_simulations(theta, x, proposal=proposal).train()

        posterior_dist = inference.build_posterior().set_default_x(true_observation)

        proposal = posterior_dist

    return posterior_dist


def run_npe(
    algorithm: str,
    true_observation: torch.Tensor,
    prior: Prior,
    sampling_method: str,
    simulator: Callable,
    device: torch.device,
    num_rounds: int = 10,
    num_simulations: int = 1000,
) -> Posterior:
    """Run the specified NPE algorithm.

    Parameters
    ----------
    algorithm : str
        The name of the NPE algorithm to run.
    true_observation : torch.Tensor
        The true observation to compare the inference results to.
    prior : Prior
        DataClass object containing the priors for the parameters.
    sampling_method : str
        The sampling method to use.
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
    prior_distribution = prior.params_prior_distribution

    if algorithm == "TSNPE":
        return tsnpe(
            prior=prior_distribution,
            simulator=simulator,
            device=device,
            sampling_method=sampling_method,
            num_rounds=num_rounds,
            num_simulations=num_simulations,
            true_observation=true_observation,
        )
    if algorithm == "APT":
        return apt(
            prior=prior_distribution,
            simulator=simulator,
            device=device,
            true_observation=true_observation,
            num_rounds=num_rounds,
            num_simulations=num_simulations,
        )
    if algorithm == "BayesFlow":
        return bayes_flow(
            prior=prior_distribution, simulator=simulator, num_sims=num_simulations
        )
    msg = f"Invalid NPE algorithm: {algorithm}"
    raise ValueError(msg)


def set_up_saving_path(script_dir: Path) -> Path:
    """Set up the saving path for the simulation results."""
    date_time = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    saving_path = Path(script_dir / "results" / date_time)
    saving_path.mkdir(parents=True, exist_ok=True)
    return saving_path


def save_meta(
    config: dict,
    device: str,
    selected_params: list[str],
    save_path: Path,
    mse_simulation: float | str,
    mse_parametric: float | str,
) -> None:
    """Save the metadata of the simulation into a yaml file.

    Parameters
    ----------
    config : dict
        The configuration file as a dictionary.
    device : str
        The device used for the simulation.
    selected_params : list[str]
        The names of the parameters that were selected for inference.
    save_path : Path
        The path to save the metadata file.
    mse_simulation : float
        The mean squared error achieved in the simulation.
    mse_parametric : float
        The mean squared error achieved in the parametric space.

    """
    config_copy = config.copy()
    config_copy["device"] = device
    config_copy["selected_params"] = selected_params
    config_copy["--results"] = {}
    config_copy["--results"]["mse_simulation"] = (
        mse_simulation if mse_simulation else None
    )
    config_copy["--results"]["mse_parametric"] = (
        mse_parametric if mse_parametric else None
    )
    with Path(save_path, "simulation_config.yaml").open("w") as f:
        yaml.dump(config_copy, f)


def save_experimental_setup(
    save_path: Path,
    prior: Prior,
    default_settings: DeafultSimulationEnv,
    true_observation: torch.Tensor,
    true_params: dict,
) -> None:
    """Save the experimental setup."""
    folder = save_path / "Experimental Setup"
    folder.mkdir(parents=True, exist_ok=True)

    torch.save(prior, Path(folder, "prior.pt"))

    with Path(folder, "inferred_params.json").open("w") as f:
        json.dump(prior.params_names, f)

    torch.save(true_observation, Path(folder, "true_observation.pt"))

    with Path(folder, "true_params.json").open("w") as f:
        json.dump(true_params, f)

    with Path(folder, "default_settings.json").open("w") as f:
        json.dump(asdict(default_settings), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        help="The name of the configuration file",
        default="test_config.yaml",
    )

    parser.add_argument(
        "--simulate_with_posterior",
        action="store_true",
        help="Simulate with the posterior",
        default=False,
    )

    parser.add_argument(
        "--plot", action="store_true", help="Plot the results", default=False
    )

    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    save_path = set_up_saving_path(script_dir=script_dir)
    script_logger = set_up_logging(saving_path=save_path)
    script_logger.info("Starting the parameter inference session")

    config = load_config(script_dir, args.config)

    device = set_up_device()
    def_hours = 24
    pathos = True
    sbi_settings: dict = config["sbi_settings"]
    default_settings = DeafultSimulationEnv(
        patient_name=config["patient_name"],
        sensor_name=config["sensor_name"],
        pump_name=config["pump_name"],
        scenario=config["scenario"],
        hours=config["hours"],
    )

    prior_settings: dict = config["prior_settings"]
    prior: Prior = prepare_prior(
        script_dir=script_dir,
        data_file=prior_settings["priors_data_file"],
        prior_type=prior_settings["prior_type"],
        number_of_params=prior_settings["number_of_params"],
        inflation_factor=prior_settings["inflation_factor"],
        mean_shift=prior_settings["mean_shift"],
        device=device,
    )
    script_logger.info(
        "Constructed prior distribution of type: %s, shape: %s",
        prior.type,
        prior.params_prior_distribution.event_shape,
    )

    sbi_simulator = set_up_sbi_simulator(
        prior=prior,
        default_settings=default_settings,
        device=device,
        glucose_simulator=run_glucose_simulator,
    )
    true_observation, true_params = get_true_observation(
        prior=prior, env_settings=default_settings, hours=config["hours"]
    )
    if args.plot:
        plt.plot(true_observation.to("cpu").numpy())
        plt.savefig(Path(save_path, "true_observation.png"))

    sbi_settings = config["sbi_settings"]
    posterior_distribution = run_npe(
        algorithm=sbi_settings["algorithm"],
        simulator=sbi_simulator,
        true_observation=true_observation,
        prior=prior,
        sampling_method=sbi_settings["sampling_method"],
        device=device,
        num_rounds=sbi_settings["num_rounds"],
        num_simulations=sbi_settings["num_simulations"],
    )
    torch.save(posterior_distribution, Path(save_path, "posterior_distribution.pt"))

    posterior_samples = sample_non_negative(
        posterior_distribution,
        num_samples=sbi_settings["n_samples_from_posterior"],
        true_observation=true_observation,
    )

    torch.save(posterior_samples, Path(save_path, "posterior_samples.pt"))

    mse_simulation = "N/A"
    mse_parametric = "N/A"

    if args.simulate_with_posterior:
        hours = config["simulate_posterior_hours"]
        glucose_dynamics_inferred = run_glucose_simulator(
            theta=posterior_samples,
            default_settings=default_settings,
            inferred_params=prior,
            device=device,
            hours=hours,
            logger=script_logger,
        )
        if hours != def_hours:
            true_observation, _ = get_true_observation(
                prior=prior, env_settings=default_settings, hours=hours
            )
        # save the glucose dynamics
        torch.save(
            glucose_dynamics_inferred, Path(save_path, "inferred_glucose_dynamics.pt")
        )

        inferred_dynamics_array = glucose_dynamics_inferred.cpu().numpy()
        true_observation_array = true_observation.cpu().numpy()
        mean_glucose_dynamics = np.mean(inferred_dynamics_array, axis=0)
        std_glucose_dynamics = np.std(inferred_dynamics_array, axis=0)

        mse_simulation = mean_squared_error(
            true_observation_array, mean_glucose_dynamics
        )
        posterior_samples_array = posterior_samples.cpu().numpy()
        true_params_values = np.array(list(true_params.values()))
        mse_parametric = mean_squared_error(
            true_params_values, np.mean(posterior_samples_array, axis=0)
        )

        if args.plot:
            fig, ax = plot_simulation(
                x_true=true_observation,
                x_inferred=inferred_dynamics_array,
                config=config,
                mse=mse_simulation,
            )

            fig.savefig(Path(save_path, "simulation_results.png"))

    save_meta(
        config=config,
        device="cuda" if torch.cuda.is_available() else "cpu",
        selected_params=prior.params_names,
        save_path=save_path,
        mse_simulation=mse_simulation,
        mse_parametric=mse_parametric,
    )

    save_experimental_setup(
        save_path=save_path,
        prior=prior,
        default_settings=default_settings,
        true_observation=true_observation,
        true_params=true_params,
    )

    script_logger.info("Parameter inference session completed")
    script_logger.info("_" * 80)
