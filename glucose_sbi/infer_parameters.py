import argparse
import json
import logging
import pickle
import shutil
import time
from collections.abc import Generator
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from typing import Callable, Protocol

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from pathos.multiprocessing import ProcessingPool as Pool
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
from simglucose.actuator.pump import InsulinPump
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj
from sklearn.metrics import mean_squared_error
from torch.distributions import Distribution
from tqdm import tqdm

from prepare_priors import Prior, prepare_prior
from sample_non_negative import sample_non_negative


class Posterior(Protocol):
    def sample(
        self, sample_shape: tuple[int, ...], x: torch.Tensor
    ) -> torch.Tensor: ...


@dataclass
class DeafultSimulationEnv:
    """Dataclass for the default simulation environment."""

    patient_name: str
    sensor_name: str
    pump_name: str
    scenario: list[tuple[int, int]] = field(default_factory=list)
    hours: int = 24  # hours to simulate


def set_up_logging(saving_path: Path) -> logging.Logger:
    """Set up the logging configuration for the script."""
    logger = logging.getLogger("script_logger")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(Path(saving_path, "inference_execution.log"))
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def load_config(config_name: "str") -> dict:
    """Loads the configuration file.

    Parameters
    ----------
    config_name : str
        The name of the configuration file to load.

    Returns
    -------
    dict
        The configuration file as a dictionary.

    """
    with Path(f"simulation_configs/{config_name}").open() as file:
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


def load_default_simulation_env(
    env_settings: DeafultSimulationEnv, hours: int = 24
) -> T1DSimEnv:
    """Load the default simulation environment.

    Parameters
    ----------
    env_settings : DeafultSimulationEnv
        DataClass object containing the default simulation environment settings.
    hours : int, optional
        The number of hours to simulate, by default 24

    Returns
    -------
    T1DSimEnv
        The simulation environment object.

    """
    now = datetime.now(tz=timezone.utc)
    start_time = datetime.combine(now.date(), datetime.min.time())

    patient = T1DPatient.withName(env_settings.patient_name)
    sensor = CGMSensor.withName(env_settings.sensor_name, seed=1)
    pump = InsulinPump.withName(env_settings.pump_name)
    scenario = CustomScenario(start_time=start_time, scenario=env_settings.scenario)
    controller = BBController()
    env = T1DSimEnv(patient=patient, sensor=sensor, pump=pump, scenario=scenario)

    return SimObj(
        env=env, controller=controller, sim_time=timedelta(hours=hours), animate=False
    )


def set_custom_params(patient: T1DPatient, theta: torch.Tensor, prior: Prior) -> None:
    """Apply the custom parameters (used for a particular simulation) for the patient.

    Parameters
    ----------
    patient : T1DPatient
        The patient object
    theta : torch.Tensor
        One set of custom paraeters to apply to the patient
    prior : Prior
        The prior for the parameters
        (we need only the names of the parameters that will actually be used in the simulation)

    """
    theta_copy = deepcopy(theta)
    custom_params_values = theta_copy.tolist()
    param_names = prior.params_names

    for i, param in enumerate(param_names):
        setattr(patient._params, param, custom_params_values[i])  # noqa: SLF001


def create_simulation_envs_with_custom_params(
    theta: torch.Tensor,
    default_settings: DeafultSimulationEnv,
    prior: Prior,
    hours: int = 24,
) -> list[T1DSimEnv]:
    """Creates a list of simulation environments with custom parameters.

    Parameters
    ----------
    theta : torch.Tensor
        Sets of custom parameters to use for the simulation of shape (N_sets, N_params)
    default_settings : DeafultSimulationEnv
        DataClass object containing the default simulation environment settings.
    prior : Prior
        DataClass object containing the prior for the parameters
    hours : int, optional
        Duration of simulation, by default 24

    Returns
    -------
    list[T1DSimEnv]
        List of simulation environments with custom parameters

    """
    default_simulation_env = load_default_simulation_env(
        hours=hours, env_settings=default_settings
    )
    simulation_envs = []
    for _, theta_i in enumerate(theta):
        custom_sim_env = deepcopy(default_simulation_env)

        set_custom_params(custom_sim_env.env.patient, theta_i, prior)
        simulation_envs.append(custom_sim_env)

    return simulation_envs


def simulate_glucose_dynamics(simulation_env: T1DSimEnv) -> np.ndarray:
    """Simulates the glucose dynamics for a given simulation environment.

    Parameters
    ----------
    simulation_env : T1DSimEnv
        The simulation environment object

    Returns
    -------
    np.ndarray
        The glucose dynamics

    """
    simulation_env.simulate()
    return simulation_env.results()["CGM"].to_numpy()


def simulate_batch(simulations: list[T1DSimEnv], device: torch.device) -> torch.Tensor:
    """Simulate a batch of simulation environments in parallel.

    Parameters
    ----------
    simulations : list[T1DSimEnv]
        List of simulation environments
    device : torch.device
        The device to use for the simulation

    Returns
    -------
    torch.Tensor
        The glucose dynamics for each simulation

    """
    tic = time.time()
    pathos = True
    if pathos:
        with Pool() as p:
            script_logger.info("Using pathos for multiprocessing")
            results = p.map(simulate_glucose_dynamics, simulations)
    else:
        script_logger.info("Pathos not available, using standard multiprocessing")
        results = [simulate_glucose_dynamics(s) for s in tqdm(simulations)]
    toc = time.time()
    script_logger.info("Simulation took %s sec.", toc - tic)
    results = np.stack(results)
    return torch.from_numpy(results).float().to(device)


def run_glucose_simulator(
    theta: torch.Tensor,
    default_settings: DeafultSimulationEnv,
    prior: Prior,
    device: torch.device,
    hours: int = 24,
) -> torch.Tensor:
    """Run the glucose simulator for a batch of custom parameters.

    Parameters
    ----------
    theta : torch.Tensor
        Sets of custom parameters to use for the simulation of shape (N_sets, N_params)
    default_settings : DeafultSimulationEnv
        DataClass object containing the default simulation environment settings.
    prior : Prior
        DataClass object containing the priors for the parameters
    hours : int, optional
        Duration of the simulation, by default 24
    device : torch.device, optional
        Device used to run the simulation, by default torch.device('cpu')

    Returns
    -------
    torch.Tensor
        The glucose dynamics time series for each simulation

    """
    script_logger.info(
        "Starting glucose simulator  with theta of shape: %s", theta.shape
    )
    simulation_envs = create_simulation_envs_with_custom_params(
        theta=theta,
        default_settings=default_settings,
        prior=prior,
        hours=hours,
    )
    return simulate_batch(simulation_envs, device)


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
        prior=prior,
        device=device,
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
    inference = NPE(prior=prior, device=device)
    theta = prior.sample((num_sims,))
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
) -> Posterior:
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
) -> Posterior:
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
    # Initialize NPE on device
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


def set_up_saving_path() -> Path:
    """Set up the saving path for the simulation results."""
    date_time = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d_%H-%M")
    saving_path = Path(f"results/{date_time}")
    saving_path.mkdir(parents=True, exist_ok=True)
    return saving_path


def save_meta(
    device: str,
    selected_params: list[str],
    patient: str,
    scenario: list,
    save_path: Path,
    mse: float,
    algorithm: str,
    n_rounds: int,
    n_simulations: int,
    pathos: str,
) -> None:
    """Save the metadata of the simulation."""
    meta = {
        "date_time": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "device": device,
        "pathos": pathos,
        "patient": patient,
        "scenario": scenario,
        "used_params": selected_params,
        "SNE method": algorithm,
        "num_rounds": n_rounds,
        "num_simulations": n_simulations,
        "best_simulation_score": mse,
    }
    with Path(save_path, "meta.json").open("w") as f:
        json.dump(meta, f)


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
    with Path(folder, "priors.pkl").open("wb") as f:
        pickle.dump(prior, f)
    with Path(folder, "default_settings.pkl").open("wb") as f:
        pickle.dump(default_settings, f)
    with Path(folder, "true_observation.pkl").open("wb") as f:
        pickle.dump(true_observation, f)
    with Path(folder, "true_params.pkl").open("wb") as f:
        pickle.dump(true_params, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        help="The name of the configuration file",
        default="default_config.yaml",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Plot the results", default=False
    )
    args = parser.parse_args()

    save_path = set_up_saving_path()
    script_logger = set_up_logging(saving_path=save_path)
    script_logger.info("Starting the parameter inference session")

    config = load_config(args.config)

    device = set_up_device()

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

    with Path(save_path, "posterior_distribution.pkl").open("wb") as f:
        pickle.dump(posterior_distribution, f)

    posterior_samples = sample_non_negative(
        posterior_distribution,
        num_samples=sbi_settings["n_samples_from_posterior"],
        true_observation=true_observation,
    )

    with Path(save_path, "posterior_samples.pkl").open("wb") as f:
        pickle.dump(posterior_samples, f)

    if args.plot:
        glucose_dynamics = run_glucose_simulator(
            theta=posterior_samples,
            default_settings=default_settings,
            prior=prior,
            device=device,
        )
        glucose_dynamics_array = glucose_dynamics.cpu().numpy()
        mean_glucose_dynamics = np.mean(glucose_dynamics_array, axis=0)
        std_glucose_dynamics = np.std(glucose_dynamics_array, axis=0)
        mse = mean_squared_error(true_observation.cpu().numpy(), mean_glucose_dynamics)
        timeline = np.arange(glucose_dynamics.shape[1])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            timeline,
            mean_glucose_dynamics,
            label="Mean of simulations a-posteriori",
            color="black",
        )
        ax.fill_between(
            timeline,
            mean_glucose_dynamics - std_glucose_dynamics,
            mean_glucose_dynamics + std_glucose_dynamics,
            alpha=0.2,
            color="black",
        )
        ax.plot(true_observation.cpu().numpy(), label="True", color="red")

        sns.despine()
        ax.legend(loc="upper right")
        ax.set_xlabel("Time, min")
        ax.set_ylabel("Glucose Dynamics")
        ax.text(
            0.1,
            0.1,
            f"MSE True vs. Posterior Mean: {mse:.2f}",
            transform=ax.transAxes,
        )
        plt.tight_layout()
        plt.savefig(Path(save_path, "posterior_samples.png"))

    save_meta(
        device="cuda" if torch.cuda.is_available() else "cpu",
        pathos="True" if pathos else "False",
        selected_params=prior.params_names,
        patient=default_settings.patient_name,
        scenario=default_settings.scenario,
        save_path=save_path,
        mse=mse,
        algorithm=sbi_settings["algorithm"],
        n_rounds=sbi_settings["num_rounds"],
        n_simulations=sbi_settings["num_simulations"],
    )
    save_experimental_setup(
        save_path=save_path,
        prior=prior,
        default_settings=default_settings,
        true_observation=true_observation,
        true_params=true_params,
    )

    shutil.copyfile(
        Path("simulation_configs") / args.config,
        Path(save_path) / Path(args.config).name,
    )
    script_logger.info("Parameter inference session completed")
    script_logger.info("_" * 80)
