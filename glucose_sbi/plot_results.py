from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def plot_simulation(
    x_true: torch.Tensor, x_inferred: torch.Tensor, save_path: str | None = None
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

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The figure and axes of the plot.

    """
    x_true_array = x_true.cpu().numpy()
    x_inferred_array = x_inferred.cpu().numpy()

    mean_x_inf = np.mean(x_inferred_array, axis=0)
    std_x_inf = np.std(x_inferred_array, axis=0)
    timeline = np.arange(x_true_array.shape[1])

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(
        timeline, x_true_array, label="Simulation with true parameters", color="lime"
    )

    ax.plot(timeline, mean_x_inf, label="Mean a-posteriori simulation", color="red")
    ax.fill_between(
        timeline,
        mean_x_inf - std_x_inf,
        mean_x_inf + std_x_inf,
        color="black",
        alpha=0.2,
    )

    ax.set_xlabel("Time, min")
    ax.set_ylabel("CGM")
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(Path(save_path) / "results.png")
    return fig, ax
