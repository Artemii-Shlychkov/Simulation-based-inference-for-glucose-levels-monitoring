import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sbi.utils.torchutils import BoxUniform
from torch.distributions import Distribution, MultivariateNormal


@dataclass
class Prior:
    type: str
    params_names: list[str]
    params_prior_distribution: Distribution | BoxUniform | MultivariateNormal


def select_random_params(n_params: int, list_of_params: list[str]) -> list[str]:
    return random.sample(list_of_params, n_params)


def mvn_from_simglucose_patients(
    data_file: str,
    device: torch.device,
    number_of_params: int,
    cov_inflation_factor: float = 1.2,
    mean_shift_scale: Optional[float] = 0.1,
    numerical_stability_factor: float = 1e-4,
) -> Prior:
    """Builds a MultivariateNormal distribution from patient parameter data, then distorts it to avoid overfitting/bias,
    inflates the covariance by `cov_inflation_factor`,
    optionally shiftd the mean stochastically by `mean_shift_scale * abs(mean)`.

    Parameters
    ----------
    data_file : str
        Path to JSON (or similarly structured) file with patient parameters.
        The JSON is expected to have structure: {param_name: [val_patient1, val_patient2, ...], ...}
    device : torch.device
        Device on which to place the resulting distribution's tensors.
    number_of_params : int
        Number of parameters to use. If n_params < len(list_of_params), a random subset of parameters is selected.
    cov_inflation_factor : float, default=1.2
        Factor by which to multiply the covariance matrix. (1.0 => no inflation)
    mean_shift_scale : float or None, default=0.1
        If not None, we sample a random shift for each dimension:
            shift_i ~ Normal(0, mean_shift_scale * abs(mean_i))
        Then add it to the empirical mean.
        If None, no random shift is applied.
    numerical_stability_factor : float, default=1e-4
        Minimum inflation added to the diagonal to prevent singular covariance.

    Returns
    -------
    Prior
        A dataclass holding:
          - type='mvn'
          - params_names: the list of parameter names in `data_file`
          - params_prior_distribution: the resulting MultivariateNormal

    """
    with Path(data_file).open("r") as f:
        all_patients_params = json.load(f)

    # Suppose all_patients_params is like:
    # We'll create an array of shape (n_params, n_patients).
    param_names = list(all_patients_params.keys())

    if number_of_params < len(param_names):
        selected_params = select_random_params(number_of_params, param_names)
    else:
        selected_params = param_names

    data_array = np.array([all_patients_params[p] for p in selected_params])

    mean_emp = np.mean(data_array, axis=1)  # shape (n_params,)
    cov_emp = np.cov(data_array)  # shape (n_params, n_params)

    # --- 2) Inflate the covariance. ---
    if cov_inflation_factor < 1.0:
        print(
            f"Warning: cov_inflation_factor={cov_inflation_factor} < 1.0 => shrinking cov?"
        )
    # Add a small diagonal for numerical stability & scale overall
    cov_inflated = cov_emp * cov_inflation_factor
    cov_inflated += np.eye(cov_inflated.shape[0]) * numerical_stability_factor

    # --- 3) Optionally shift the mean stochastically. ---
    mean_emp_t = torch.tensor(mean_emp, dtype=torch.float32, device=device)
    cov_inflated_t = torch.tensor(cov_inflated, dtype=torch.float32, device=device)

    if mean_shift_scale is not None and mean_shift_scale > 0:
        gen = torch.Generator(device=device)

        stddev_vec = mean_emp_t.abs() * mean_shift_scale

        shift_vec = torch.normal(mean=0.0, std=stddev_vec, generator=gen)
        mean_final_t = mean_emp_t + shift_vec
    else:
        mean_final_t = mean_emp_t

    # --- 4) Build the MVN distribution. ---
    return MultivariateNormal(loc=mean_final_t, covariance_matrix=cov_inflated_t)


def box_uniform_from_simglucose_patients(
    data_file: str, n_params: int, inflation_factor: float, device: torch.device
) -> Prior:
    """Creates a box uniform prior distribution from the known parameters of the simglucose patients.

    Parameters
    ----------
    data_file : str
        path to a JSON file containing the patient parameters.
        The JSON is expected to have structure: {param_name: [val_patient1, val_patient2, ...], ...}
    n_params : int
        number of parameters to use. IF n_params < len(list_of_params), a random subset of parameters is selected.
    inflation_factor : float
        amount to inflate the parameter range by.
    device : torch.device
        device to store the tensors on.

    Returns
    -------
    Prior
        A dataclass holding:
            - type='BoxUniform'
            - params_names: the list of parameter names in `data_file`
            - params_prior_distribution: the resulting BoxUniform distribution

    """
    if inflation_factor <= 1:
        print("Inflation factor should be greater than 1. Setting it to 1.")
        inflation_factor = 1

    with Path(data_file).open() as f:
        all_patients_params = json.load(f)
    list_of_params = list(all_patients_params.keys())

    if n_params < len(list_of_params):
        selected_params = select_random_params(n_params, list_of_params)
    else:
        selected_params = list_of_params

    uniform_params = {}
    for key in selected_params:
        max_val = max(all_patients_params[key]) * inflation_factor
        min_val = min(all_patients_params[key]) / inflation_factor
        uniform_params[key] = (min_val, max_val)

    return BoxUniform(
        low=torch.tensor(
            [uniform_params[key][0] for key in selected_params], device=device
        ),
        high=torch.tensor(
            [uniform_params[key][1] for key in selected_params], device=device
        ),
    )


def mvn_from_domain_knowledge(
    data_file: str,
    n_params: int,
) -> MultivariateNormal:
    """Creates a prior distribution from a .csv file containing the mean and \
        standard deviation of the parameters.

    Parameters
    ----------
    data_file : str
        The path to the .csv file containing the mean and standard deviation of the parameters.
    n_params : int
        The number of parameters to use. If n_params < len(list_of_params), a random subset of parameters is selected.

    Returns
    -------
    MultivariateNormal
        A multivariate normal distribution with the mean and standard deviation of the parameters.

    """
    params_df = pd.read_csv(data_file, header=0)
    param_names = params_df["Parameter"].tolist()

    if n_params < len(param_names):
        selected_params = select_random_params(n_params, param_names)
    else:
        selected_params = param_names

    param_means = params_df["Mean"].to_numpy()
    param_stds = params_df["Std"].to_numpy()
    mean_tensor = torch.tensor(param_means, dtype=torch.float32)
    std_tensor = torch.tensor(param_stds, dtype=torch.float32)
    cov = torch.diag(std_tensor**2)
    stability_factor = 1e-4
    cov += torch.eye(cov.shape[0]) * stability_factor

    return MultivariateNormal(loc=mean_tensor, covariance_matrix=cov)


def prepare_prior(
    data_file: str,
    prior_type: str,
    number_of_params: int,
    inflation_factor: float,
    mean_shift: float,
    device: torch.device,
) -> Prior:
    """Creates a prior distribution from the known parameters of the simglucose patients.

    Parameters
    ----------
    data_file : str
        path to a file containing the patient parameters.
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

    Returns
    -------
    Prior
        A dataclass holding:
            - type='mvn' or 'BoxUniform'
            - params_names: the list of parameter names in `data_file`
            - params_prior_distribution: the resulting prior distribution

    Raises
    ------
    ValueError
        If an invalid prior type is provided.

    """
    if prior_type == "mvn":
        return mvn_from_simglucose_patients(
            data_file=data_file,
            device=device,
            number_of_params=number_of_params,
            cov_inflation_factor=inflation_factor,
            mean_shift_scale=mean_shift,
        )
    if prior_type == "uniform":
        return box_uniform_from_simglucose_patients(
            data_file=data_file,
            n_params=number_of_params,
            inflation_factor=inflation_factor,
            device=device,
        )
    if prior_type == "mvn_domain_knowledge":
        return mvn_from_domain_knowledge(data_file, device)
    msg = "Invalid prior type or not implemented yet."
    raise ValueError(msg)
