import argparse
import json
import logging
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from simglucose.simulation.sim_engine import SimObj
from sklearn.metrics import mean_squared_error

from glucose_sbi.check_config import check_config
from glucose_sbi.glucose_simulator import (
    EnvironmentSettings,
    create_simulation_object,
    generate_true_observation,
    run_glucose_simulator,
)
from glucose_sbi.prepare_priors import (
    InferredParams,
    Prior,
    prepare_prior,
)
from glucose_sbi.process_results import plot_meals, plot_simulation
from glucose_sbi.sbi_framework import (
    get_simulation_params,
    run_inference,
    sample_positive,
)


def _random_scenario() -> list[tuple[int, int]]:
    """Generate a random scenario."""
    return [
        (7, random.randint(1, 100)),
        (12, random.randint(1, 100)),
        (16, random.randint(1, 100)),
        (18, random.randint(1, 100)),
        (23, random.randint(1, 100)),
    ]


def set_up_logging(saving_path: Path) -> logging.Logger:
    """Set up the logging for the simulation results."""
    logger = logging.getLogger("sbi_logger")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(Path(saving_path, "inference_execution.log"))
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logger.addHandler(handler)

    return logger


def set_up_saving_path(script_dir: Path) -> Path:
    """Set up the saving path for the simulation results."""
    date_time = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    saving_path = Path(script_dir / "results" / date_time)
    saving_path.mkdir(parents=True, exist_ok=True)
    return saving_path


def save_experimental_setup(
    save_path: Path,
    prior: Prior,
    default_settings: EnvironmentSettings,
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


def set_up_default_simulation_object(config: dict) -> SimObj:
    """Set up the default simulation object for the simulation based on specifications
    in the configuration file. The simulation object has default presets like meal scenario and
    patient parameters, ready to `.simulate`.

    Parameters
    ----------
    config : dict
        The configuration file as a dictionary.

    Returns
    -------
    SimObj
        The resulting simulation object with all the necessary presets and ready to `.simulate`

    """
    scenario = config.get("scenario", _random_scenario())
    default_settings = EnvironmentSettings(
        patient_name=config["patient_name"],
        sensor_name=config["sensor_name"],
        pump_name=config["pump_name"],
        scenario=scenario,
        hours=config["hours"],
    )
    return create_simulation_object(default_settings)


def set_up_prior(config: dict) -> Prior:
    """Set up the Prior dataclass object, containing the prior distribution and its type,
    as well as the names of the parameters to be inferred.

    Parameters
    ----------
    config : dict
        The configuration file as a dictionary.

    Returns
    -------
    Prior
        A dataclass holding:
            - type of the distribution
            - params_names: the list of parameter names which will be inferred
            - params_prior_distribution: the resulting prior distribution

    """
    prior_settings: dict = config["prior_settings"]
    return prepare_prior(
        script_dir=script_dir,
        data_file=prior_settings["priors_data_file"],
        prior_type=prior_settings["prior_type"],
        number_of_params=prior_settings["number_of_params"],
        inflation_factor=prior_settings["inflation_factor"],
        mean_shift=prior_settings["mean_shift"],
        device=device,
        infer_meal_params=config["infer_meal_params"],
    )


def simulate_with_posterior(
    posterior_samples: torch.Tensor,
    default_simulation_object: SimObj,
    *,
    hours: int = 24,
    device: torch.device,
    inferred_params: InferredParams,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulates the glucose dynamics using the inferred (posterior samples) and the true parameters.
    Returns two corresponding torch.Tensor arrays of glucose dynamics.

    Parameters
    ----------
    posterior_samples : torch.Tensor
        The posterior samples obtained from the inference.
    default_simulation_object : SimObj
        The default simulation object.
    hours : int, optional
        hours to simulate, by default 24
    device : torch.device, optional
        The device used for computations
    inferred_params : InferredParams, optional
        The nemase of the inferred parameters.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Two torch.Tensor arrays of glucose dynamics: the inferred and the true.

    """
    glucose_dynamics_inferred = run_glucose_simulator(
        theta=posterior_samples,
        default_simulation_object=default_simulation_object,
        inferred_params=inferred_params,
        hours=hours,
        device=device,
        infer_meal_params=True,
    )

    true_observation = generate_true_observation(
        default_simulation_object=default_simulation_object,
        device=device,
        hours=hours,
    )

    return glucose_dynamics_inferred, true_observation


def calculate_mse(
    inferred_dynamics: torch.Tensor,
    true_observation: torch.Tensor,
    posterior_samples: torch.Tensor,
    true_params: dict,
) -> tuple[float | None, float | None]:
    """Calculate the mean squared error in the signal space (simulation results)
    and in the parametric space.

    Parameters
    ----------
    inferred_dynamics : torch.Tensor
        Glucose dynamics with inferred parameters.
    true_observation : torch.Tensor
        Glucose dynamics with true parameters.
    posterior_samples : torch.Tensor
        The posterior samples obtained from the inference (Inferred parameters).
    true_params : dict
        The true parameters.

    Returns
    -------
    tuple[float | None, float | None]
        The mean squared error in the signal space and in the parametric space.

    """
    mse_parametric = None
    mse_simulation = None
    inferred_dynamics_array = inferred_dynamics.cpu().numpy()
    true_observation_array = true_observation.cpu().numpy()
    mean_glucose_dynamics = np.mean(inferred_dynamics_array, axis=0)

    mse_simulation = mean_squared_error(true_observation_array, mean_glucose_dynamics)

    posterior_samples_array = posterior_samples.cpu().numpy()

    try:
        true_params_values = np.array(list(true_params.values()))
        mse_parametric = mean_squared_error(
            true_params_values, np.mean(posterior_samples_array, axis=0)
        )
    except ValueError:
        script_logger.warning("Could not calculate MSE in the parametric space")
    return mse_simulation, mse_parametric


def save_meta(
    config: dict,
    device: str,
    selected_params: list[str],
    save_path: Path,
    mse_simulation: float | None,
    mse_parametric: float | None,
) -> None:
    """Save the metadata of the simulation into a yaml file.

    Parameters
    ----------
    config : dict
        The configuration file as a dictionary.
    device : str
        device used for computations
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
    config_copy["--results"]["mse_simulation"] = mse_simulation
    config_copy["--results"]["mse_parametric"] = mse_parametric
    with Path(save_path, "simulation_config.yaml").open("w") as f:
        yaml.dump(config_copy, f)


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

    config = load_config(script_dir, args.config)
    check_config(config_file=Path(script_dir / "simulation_configs" / args.config))

    device = set_up_device()
    default_simulation_object = set_up_default_simulation_object(config)
    prior = set_up_prior(config)

    true_observation = generate_true_observation(
        default_simulation_object, device=device, hours=config["hours"]
    )
    true_params = get_simulation_params(
        simulation_object=default_simulation_object,
        inferred_params=InferredParams(prior.params_names),
    )

    save_experimental_setup(
        save_path=save_path,
        prior=prior,
        default_settings=default_simulation_object.env.settings,
        true_observation=true_observation,
        true_params=true_params,
    )

    posterior_distribution = run_inference(
        prior=prior,
        true_observation=true_observation,
        device=device,
        config=config,
        default_simulation_object=default_simulation_object,
    )
    torch.save(posterior_distribution, Path(save_path, "posterior_distribution.pt"))

    script_logger.info("Sampling from the posterior distribution...")
    n_posterior_samples = config["n_posterior_samples"]
    posterior_samples = sample_positive(
        posterior_distribution,
        num_samples=n_posterior_samples,
        x_true=true_observation,
    )
    torch.save(posterior_samples, Path(save_path, "posterior_samples.pt"))

    if args.simulate_with_posterior:
        if config["simulate_posterior_hours"] != config["hours"]:
            inferred_dynamics, true_observation = simulate_with_posterior(
                hours=config["hours"],
                posterior_samples=posterior_samples,
                device=device,
                default_simulation_object=default_simulation_object,
                inferred_params=InferredParams(prior.params_names),
            )
        else:
            inferred_dynamics, _ = simulate_with_posterior(
                posterior_samples=posterior_samples,
                device=device,
                default_simulation_object=default_simulation_object,
                inferred_params=InferredParams(prior.params_names),
            )

        mse_simulation, mse_parametric = calculate_mse(
            inferred_dynamics, true_observation, posterior_samples, true_params
        )
    if args.plot:
        fig, ax = plot_simulation(
            x_true=true_observation,
            x_inferred=inferred_dynamics,
            config=config,
            mse=mse_simulation,
        )

        fig.savefig(Path(save_path, "simulation_results.png"))
        true_scenario = default_simulation_object.env.scenario.scenario
        inferred_scenario = posterior_samples.cpu().numpy()[:, -5:]

        fig, ax = plot_meals(
            true_scenario=true_scenario,
            inferred_scenario=inferred_scenario,
        )
        fig.savefig(Path(save_path, "meal_results.png"))

    save_meta(
        config=config,
        device="cuda" if torch.cuda.is_available() else "cpu",
        selected_params=prior.params_names,
        save_path=save_path,
        mse_simulation=mse_simulation,
        mse_parametric=mse_parametric,
    )

    script_logger.info("Parameter inference session completed")
    script_logger.info("_" * 80)
