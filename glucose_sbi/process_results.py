import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from matplotlib import ticker
from sbi.inference import DirectPosterior

from glucose_sbi.glucose_simulator import DeafultSimulationEnv, run_glucose_simulator
from glucose_sbi.prepare_priors import Prior


@dataclass
class Results:
    default_settings: DeafultSimulationEnv
    prior: Prior
    true_observation: torch.Tensor
    true_params: dict
    posterior_dist: DirectPosterior
    posterior_samples: torch.Tensor
    config: dict


def load_results(results_folder: Path) -> Results:
    """Load results of a particular parameter inference experiment.

    Parameters
    ----------
    results_folder : Path
        The folder containing the results of the parameter inference experiment.

    Returns
    -------
    Results
        A dataclass containing the results of the parameter inference experiment.

    """
    setup_folder = results_folder / "Experimental Setup"
    default_settings = pickle.load((setup_folder / "default_settings.pkl").open("rb"))
    prior = pickle.load((setup_folder / "priors.pkl").open("rb"))
    true_observation = pickle.load((setup_folder / "true_observation.pkl").open("rb"))
    true_params = pickle.load((setup_folder / "true_params.pkl").open("rb"))
    posterior_dist = pickle.load(
        (results_folder / "posterior_distribution.pkl").open("rb")
    )
    posterior_samples = pickle.load(
        (results_folder / "posterior_samples.pkl").open("rb")
    )

    # find and load the yaml file
    config_file = Path.glob(results_folder, "*.yaml")
    config = yaml.safe_load(next(config_file).open("r"))
    return Results(
        default_settings,
        prior,
        true_observation,
        true_params,
        posterior_dist,
        posterior_samples,
        config,
    )


def simulate_true_and_inferred(
    results: Results, device: torch.device, hours: int
) -> tuple[np.ndarray, np.ndarray]:
    """Runs the simulation with true and inferred parameters.

    Parameters
    ----------
    results : Results
        A dataclass containing the results of the inference.
    device : torch.device
        The device used in the simulator runner, by default torch.device("cpu")
    hours : int
        The number of hours to run the simulation for.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the true and inferred simulations

    """
    posterior_samples = results.posterior_samples
    true_parameters = results.true_params
    default_settings = results.default_settings
    prior = results.prior
    theta_true = torch.tensor(
        [value for _, value in true_parameters.items()]
    ).unsqueeze(0)
    sim_true = run_glucose_simulator(
        theta=theta_true,
        default_settings=default_settings,
        prior=prior,
        device=device,
        hours=hours,
    )
    sim_inferred = run_glucose_simulator(
        theta=posterior_samples,
        default_settings=default_settings,
        prior=prior,
        device=device,
        hours=hours,
    )
    sim_true_array = sim_true.detach().cpu().numpy()
    sim_inferred_array = sim_inferred.detach().cpu().numpy()

    expected_shape = 1
    if len(sim_true_array.shape) != expected_shape:
        sim_true_array = np.squeeze(sim_true_array)
    return sim_true_array, sim_inferred_array


def plot_simulation(
    x_true: torch.Tensor | np.ndarray,
    x_inferred: torch.Tensor | np.ndarray,
    config: dict | None,
    mse: float | str = "N/A",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the results of a simulation.

    Parameters
    ----------
    x_true : torch.Tensor
        Glucose dynamics with true parameters.
    x_inferred : torch.Tensor
        Glucose dynamics with inferred parameters.
    save_path : str | None, optional
        Directory to save the figure, by default None
    config : dict | None, optional
        Configuration settings for the simulation, by default None
    mse : float | None, optional
        Mean squared error between the true and inferred glucose dynamics, by default
        None

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The figure and axes of the plot.

    """
    x_true_array = x_true.cpu().numpy() if isinstance(x_true, torch.Tensor) else x_true

    # check shape and squeeze if necessary
    single_dim = 1
    if len(x_true_array.shape) != single_dim:
        x_true_array = np.squeeze(x_true_array)

    x_inferred_array = (
        x_inferred.cpu().numpy() if isinstance(x_inferred, torch.Tensor) else x_inferred
    )

    mean_x_inf = np.mean(x_inferred_array, axis=0)
    std_x_inf = np.std(x_inferred_array, axis=0)
    timeline = np.arange(
        0, 3 * x_true_array.shape[0], 3
    )  # results are reported every 3 minutes

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(
        timeline, x_true_array, label="Simulation with true parameters", color="lime"
    )

    ax.plot(timeline, mean_x_inf, label="Mean a-posteriori simulation", color="black")

    ax.fill_between(
        timeline,
        mean_x_inf - std_x_inf,
        mean_x_inf + std_x_inf,
        color="black",
        alpha=0.3,
        label="Standard deviation",
    )
    ax.scatter([0], [0], alpha=0, label=f"MSE = {mse:.2f}") if mse else None
    # set major ticks as hours and minor ticks as half-hours
    ax.set_xticks(
        timeline[::60]
    )  # Major ticks every hour (60 minutes / 3 minutes per step)
    ax.set_xticks(
        timeline[::20], minor=True
    )  # Minor ticks every half-hour (30 minutes / 3 minutes per step)
    ax.set_xticklabels(timeline[::60] // 60, fontdict={"fontsize": 12})

    ax.set_ylim(min(0, np.min(x_true_array) - 100), np.max(x_true_array) + 100)
    y_ticks = ax.get_yticks()
    ax.yaxis.set_major_locator(ticker.FixedLocator(y_ticks.tolist()))
    ax.set_yticklabels(y_ticks, fontdict={"fontsize": 12})

    ax.set_xlabel("Time, hours", fontsize=12, fontweight="bold")

    ax.set_ylabel("CGM", fontsize=12, fontweight="bold")
    ax.legend(loc="lower left")

    if config:
        sbi_settings = config["sbi_settings"]
        ax.set_title(
            f"{config['patient_name']}\n{sbi_settings['algorithm']} - {sbi_settings['num_rounds']} round(s) - {sbi_settings['num_simulations']} simulations"
        )
    sns.despine()
    plt.tight_layout()
    return fig, ax
