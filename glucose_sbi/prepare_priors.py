import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sbi.utils import BoxUniform
from torch.distributions import (
    ExpTransform,
    MultivariateNormal,
    TransformedDistribution,
    Uniform,
)

logger = logging.getLogger("glucose_sbi.prepare_priors")


@dataclass
class InferredParams:
    """Dataclass containing the names of the inferred parameters."""

    params_names: list[str]


@dataclass
class Prior(InferredParams):
    """Dataclass containing the prior distribution of the inferred parameters and its type."""

    type: str
    params_prior_distribution: BoxUniform | MultivariateNormal | TransformedDistribution


def _select_random_keys(n_keys: int, d: dict) -> list[str]:
    """Selects a random subset of keys from the dictionary."""
    return random.sample(sorted(d), n_keys)


def _select_random_dict_subset(n_params: int, d: dict) -> dict:
    """Selects a random subset of key-value pairs from a dictionary."""
    return {k: d[k] for k in _select_random_keys(n_keys=n_params, d=d)}


def _generate_from_uniform(n_samples: int, low: int, high: int) -> np.ndarray:
    """Generates random numbers from a uniform distribution."""
    rng = np.random.default_rng()
    return rng.integers(low=low, high=high, size=n_samples)


def _include_meal_params(data: dict, n_observations: int) -> dict:
    """Adds meal parameters to the data dictionary."""
    meal_dict = {
        f"meal_{time}": _generate_from_uniform(n_observations, 1, 100)
        for time in [7, 12, 16, 18, 23]
    }
    data.update(meal_dict)
    return data


def construct_mvn_prior(
    data: dict,
    *,
    device: torch.device,
    cov_inflation_factor: float = 1.0,
    mean_shift_scale: float = 0.0,
    numerical_stability_factor: float = 1e-4,
) -> MultivariateNormal:
    """Given some observed values for patient parameters, constructs a multivariate normal distribution,
    then optionally distorts it to avoid overfitting/bias:
    - inflates the covariance by `cov_inflation_factor`,
    - shifts the mean stochastically by `mean_shift_scale * abs(mean)`.

    Parameters
    ----------
    data : dict
        patient parameters. The data is expected to have structure: {param_name: [val_patient1, val_patient2, ...], ...}
    device : torch.device
        Device on which to place the resulting distribution's tensors: ["cuda" | "cpu"]
    cov_inflation_factor : float, default=1.0
        Factor by which to multiply the covariance matrix. (1.0 => no inflation)
    mean_shift_scale : float, default=0.0
        If > 0, we sample a random shift for each dimension:
            shift_i ~ Normal(0, mean_shift_scale * abs(mean_i))
        Then add it to the empirical mean.
        If 0, no random shift is applied.
    numerical_stability_factor : float, default=1e-4
        Minimum inflation added to the diagonal to prevent singular covariance.

    Returns
    -------
    MultivariateNormal
        A multivariate normal distribution with the inflated covariance and possibly shifted mean.

    """
    logger.info("Constructing MVN prior...")

    data_array = np.array([data[key] for key in data])

    mean_emp = np.mean(data_array, axis=1)
    cov_emp = np.cov(data_array)

    if cov_inflation_factor < 1.0:
        logger.warning("Warning: cov_inflation_factor < 1.0 => shrinking cov?")

    cov_inflated = cov_emp * cov_inflation_factor
    cov_inflated += np.eye(cov_inflated.shape[0]) * numerical_stability_factor

    mean_emp_t = torch.tensor(mean_emp, dtype=torch.float32, device=device)
    cov_inflated_t = torch.tensor(cov_inflated, dtype=torch.float32, device=device)

    if mean_shift_scale > 0:
        crit_mean_shift = 0.5
        if mean_shift_scale > crit_mean_shift:
            logger.warning("Warning: mean_shift_scale > 0.5 => too large shifts?")
        gen = torch.Generator(device=device)

        stddev_vec = mean_emp_t.abs() * mean_shift_scale

        shift_vec = torch.normal(mean=0.0, std=stddev_vec, generator=gen)
        mean_final_t = mean_emp_t + shift_vec
    else:
        mean_final_t = mean_emp_t

    return MultivariateNormal(loc=mean_final_t, covariance_matrix=cov_inflated_t)


def construct_lognormal_prior(
    data: dict,
    *,
    device: torch.device,
    cov_inflation_factor: float = 1.0,
    mean_shift_scale: float = 0.0,
    numerical_stability_factor: float = 1e-4,
) -> TransformedDistribution:
    """Creates a log-multivariate normal distribution from positive data, by:
      1) taking log of the data,
      2) computing mean, cov,
      3) optionally inflating covariance and shifting mean,
      4) constructing a base MVN in log-space,
      5) exponentiating via ExpTransform -> final distribution is lognormal in real space.
    The final returned distribution is a `TransformedDistribution(base_mvn, ExpTransform())`.
    Sampling from it yields strictly positive vectors.

    Parameters
    ----------
    data : dict
        patient parameters. The data is expected to have structure:
          {param_name: [val_patient1, val_patient2, ...], ...}
        *All values must be > 0* (strictly positive).
    device : torch.device
        Device on which to place the resulting distribution's tensors: ["cuda" | "cpu"]
    cov_inflation_factor : float, default=1.0
        Factor by which to multiply the covariance matrix in log-space. (1.0 => no inflation)
    mean_shift_scale : float, default=0.0
        If > 0, we sample a random shift in log-space for each dimension:
            shift_i ~ Normal(0, mean_shift_scale * abs(log_mean_i))
        Then add it to the empirical log-mean.
    numerical_stability_factor : float, default=1e-4
        Minimum inflation added to the diagonal of the covariance (in log-space)
        to ensure it's positive-definite.

    Returns
    -------
    TransformedDistribution
        A log-multivariate normal distribution. i.e. samples are always > 0.
        This object has .sample() and .log_prob() methods in real (positive) space.

    """
    logger.info("Constructing lognormal prior...")

    data_array = np.array([data[key] for key in data])

    if not np.all(data_array > 0):
        m = "All data values must be strictly > 0 for log transform."
        raise ValueError(m)

    log_data = np.log(data_array)

    mean_emp_log = np.mean(log_data, axis=1)
    cov_emp_log = np.cov(log_data)

    if cov_inflation_factor < 1.0:
        logger.warning("Warning: cov_inflation_factor < 1.0 => shrinking cov?")

    cov_inflated = cov_emp_log * cov_inflation_factor

    cov_inflated += np.eye(cov_inflated.shape[0]) * numerical_stability_factor

    mean_log_t = torch.tensor(mean_emp_log, dtype=torch.float32, device=device)
    cov_log_t = torch.tensor(cov_inflated, dtype=torch.float32, device=device)

    if mean_shift_scale > 0:
        crit_mean_shift = 0.5
        if mean_shift_scale > crit_mean_shift:
            logger.warning("Warning: mean_shift_scale > 0.5 => too large shifts?")
        gen = torch.Generator(device=device)
        stddev_vec = mean_log_t.abs() * mean_shift_scale
        shift_vec = torch.normal(mean=0.0, std=stddev_vec, generator=gen)
        mean_log_t = mean_log_t + shift_vec

    base_mvn = MultivariateNormal(loc=mean_log_t, covariance_matrix=cov_log_t)

    return TransformedDistribution(base_mvn, ExpTransform())


def construct_box_uniform_prior(
    data: dict, *, device: torch.device, inflation_factor: float = 1.0
) -> Uniform:
    """Given some observed values for patient parameters, constructs a multivariate uniform distribution, then optionally expands the range by `inflation_factor`.

    Parameters
    ----------
    data : dict
        Patient parameters. The data is expected to have structure: {param_name: [val_patient1, val_patient2, ...], ...} or {param_name: [max_val, min_val], ...}
    device : torch.device
        Device on which to place the resulting distribution's tensors: ["cuda" | "cpu"]
    inflation_factor : float
        Amount to inflate the parameter range by. By default, no inflation is applied.


    Returns
    -------
    Prior
        A dataclass holding:
            - type='Uniform'
            - params_names: the list of parameter names in `data_file`
            - params_prior_distribution: the resulting BoxUniform distribution

    """
    logger.info("Constructing BoxUniform prior...")
    if inflation_factor < 1.0:
        logger.warning("Warning: inflation_factor < 1.0 => shrinking range?")

    uniform_params = {}
    for key, values in data.items():
        max_val = max(values) * inflation_factor
        min_val = min(values) / inflation_factor
        min_val = max(0, min_val)  # Ensure non-negative values
        uniform_params[key] = (min_val, max_val)

    return BoxUniform(
        low=torch.tensor([uniform_params[key][0] for key in data], device=device),
        high=torch.tensor([uniform_params[key][1] for key in data], device=device),
    )


def mvn_from_mean_std(
    data: dict,
) -> MultivariateNormal:
    """Creates a prior distribution from a data file containing the mean and
    standard deviation of the parameters.

    Parameters
    ----------
    data: dict
        patient parameters. The data is expected to have structure: {param_name: {Mean: val, Std: val}, ...}

    Returns
    -------
    MultivariateNormal
        A multivariate normal distribution with the mean and standard deviation of the parameters.

    """
    param_means = [data[key]["Mean"] for key in data]
    param_stds = [data[key]["Std"] for key in data]
    mean_tensor = torch.tensor(param_means, dtype=torch.float32)
    std_tensor = torch.tensor(param_stds, dtype=torch.float32)
    cov = torch.diag(std_tensor**2)
    stability_factor = 1e-4
    cov += torch.eye(cov.shape[0]) * stability_factor

    return MultivariateNormal(loc=mean_tensor, covariance_matrix=cov)


def prepare_prior(
    script_dir: Path,
    data_file: str,
    prior_type: str,
    number_of_params: int,
    inflation_factor: float,
    mean_shift: float,
    *,
    infer_meal_params: bool = False,
    device: torch.device,
) -> Prior:
    """Creates a prior distribution from the known parameters of the simglucose patients.

    Parameters
    ----------
    script_dir : Path
        path to the directory containing the data_file.
    data_file : str
        path to a json file containing the patient parameters.
        data_file is expected to have structure: {param_name: [val_patient1, val_patient2, ...], ...} or {param_name: {Mean: val, Std: val}, ...}
    prior_type : str
        type of prior to create. Either "mvn" or "BoxUniform".
    number_of_params : int
        number of parameters to use. IF n_params < len(list_of_params), a random subset of parameters is selected.
    inflation_factor : float
        amount to inflate the parameter range by.
        For mvn, this is the jitter factor, inflating the covariance matrix.
        For the BoxUniform, this is the factor by which to inflate the parameter range.
    mean_shift : float
        amount to shift the mean vector by. Only used for the mvn prior.
    device : torch.device
        device to store the tensors on.
    infer_meal_params : bool, optional
        If True, the meal parameters are inferred as well, by default False

    Returns
    -------
    Prior
        A dataclass holding:
            - type of the distribution
            - params_names: the list of parameter names which will be inferred
            - params_prior_distribution: the resulting prior distribution

    Raises
    ------
    ValueError
        If an invalid prior type is provided.

    """
    with Path(script_dir / data_file).open("r") as f:
        all_patients_params = json.load(f)

    list_of_params = list(all_patients_params.keys())

    if number_of_params < len(list_of_params):
        selected_data = _select_random_dict_subset(
            number_of_params, all_patients_params
        )

    else:
        selected_data = all_patients_params
    n_observations = len(next(iter(selected_data.values()))) if selected_data else 10

    if infer_meal_params:
        selected_data = _include_meal_params(selected_data, n_observations)

    selected_params = list(selected_data.keys())

    if len(selected_params) == 0:
        msg = "No parameters were selected."
        logger.error(msg)
        raise ValueError(msg)

    match prior_type:
        case "mvn":
            return Prior(
                type="mvn",
                params_names=selected_params,
                params_prior_distribution=construct_mvn_prior(
                    data=selected_data,
                    device=device,
                    cov_inflation_factor=inflation_factor,
                    mean_shift_scale=mean_shift,
                ),
            )
        case "uniform":
            return Prior(
                type="uniform",
                params_names=selected_params,
                params_prior_distribution=construct_box_uniform_prior(
                    data=selected_data,
                    inflation_factor=inflation_factor,
                    device=device,
                ),
            )
        case "lognormal":
            return Prior(
                type="lognormal",
                params_names=selected_params,
                params_prior_distribution=construct_lognormal_prior(
                    data=selected_data,
                    device=device,
                    cov_inflation_factor=inflation_factor,
                    mean_shift_scale=mean_shift,
                ),
            )
        case "mvn_from_mean_std":
            return Prior(
                type="mvn_from_mean_std",
                params_names=selected_params,
                params_prior_distribution=mvn_from_mean_std(
                    data=selected_data,
                ),
            )

    msg = "Invalid prior type or not implemented yet."
    raise ValueError(msg)
