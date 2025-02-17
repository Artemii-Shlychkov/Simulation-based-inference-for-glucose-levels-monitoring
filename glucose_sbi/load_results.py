import pickle
from dataclasses import dataclass
from pathlib import Path

import torch
from sbi.inference import DirectPosterior

from glucose_sbi.infer_parameters import DeafultSimulationEnv
from glucose_sbi.prepare_priors import Prior


@dataclass
class Results:
    default_settings: DeafultSimulationEnv
    prior: Prior
    true_observation: torch.Tensor
    true_params: dict
    posterior_dist: DirectPosterior
    posterior_samples: torch.Tensor


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
    posterior_dist = pickle.load((results_folder / "posterior_dist.pkl").open("rb"))
    posterior_samples = pickle.load(
        (results_folder / "posterior_samples.pkl").open("rb")
    )

    return Results(
        default_settings,
        prior,
        true_observation,
        true_params,
        posterior_dist,
        posterior_samples,
    )
