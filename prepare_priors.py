import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sbi.utils.torchutils import BoxUniform
from torch.distributions import Distribution


@dataclass
class ParamPriors:
    """Dataclass for the prior distributions of the model parameters."""

    x0_10: tuple[float, float] | None = None
    x0_11: tuple[float, float] | None = None
    x0_12: tuple[float, float] | None = None
    x0_13: tuple[float, float] | None = None
    BW: tuple[float, float] | None = None
    EGPb: tuple[float, float] | None = None
    Gb: tuple[float, float] | None = None
    Ib: tuple[float, float] | None = None
    kabs: tuple[float, float] | None = None
    kmax: tuple[float, float] | None = None
    kmin: tuple[float, float] | None = None
    b: tuple[float, float] | None = None
    d: tuple[float, float] | None = None
    Vg: tuple[float, float] | None = None
    Vi: tuple[float, float] | None = None
    Ipb: tuple[float, float] | None = None
    Vmx: tuple[float, float] | None = None
    Km0: tuple[float, float] | None = None
    k2: tuple[float, float] | None = None
    k1: tuple[float, float] | None = None
    p2u: tuple[float, float] | None = None
    m1: tuple[float, float] | None = None
    m5: tuple[float, float] | None = None
    CL: tuple[float, float] | None = None
    HEb: tuple[float, float] | None = None
    m2: tuple[float, float] | None = None
    m4: tuple[float, float] | None = None
    m30: tuple[float, float] | None = None
    Ilb: tuple[float, float] | None = None
    ki: tuple[float, float] | None = None
    kp2: tuple[float, float] | None = None
    kp3: tuple[float, float] | None = None
    Gpb: tuple[float, float] | None = None
    Gtb: tuple[float, float] | None = None
    Vm0: tuple[float, float] | None = None
    Rdb: tuple[float, float] | None = None
    PCRb: tuple[float, float] | None = None
    kd: tuple[float, float] | None = None
    ksc: tuple[float, float] | None = None
    ka1: tuple[float, float] | None = None
    ka2: tuple[float, float] | None = None
    u2ss: tuple[float, float] | None = None
    isc1ss: tuple[float, float] | None = None
    isc2ss: tuple[float, float] | None = None
    kp1: tuple[float, float] | None = None


@dataclass
class Prior:
    type: str
    params_names: list[str]
    params_prior_distribution: Distribution | BoxUniform


def select_random_params(n_params: int, list_of_params: list[str]) -> list[str]:
    return random.sample(list_of_params, n_params)


import json
import numpy as np
import torch
from torch.distributions import MultivariateNormal
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class Prior:
    """Simple dataclass container for your prior distribution."""

    type: str
    params_names: list
    params_prior_distribution: MultivariateNormal


def mvn_from_simglucose_patients(
    data_file: str,
    device: torch.device,
    number_of_params: int,
    cov_inflation_factor: float = 1.2,
    mean_shift_scale: Optional[float] = 0.1,
    numerical_stability_factor: float = 1e-4,
    # random_state: int = 42,
) -> Prior:
    """Builds a MultivariateNormal distribution from patient parameter data, then distorts it to avoid overfitting/bias,
    inflates the covariance by `cov_inflation_factor`,
    optionally shiftd the mean stochastically by `mean_shift_scale * abs(mean)`.

    Parameters
    ----------
    data_file : str
        Path to JSON (or similarly structured) file with patient parameters.
        The JSON is expected to have structure: {param_name: [val_patient1, val_patient2, ...], ...}
    cov_inflation_factor : float, default=1.2
        Factor by which to multiply the covariance matrix. (1.0 => no inflation)
    mean_shift_scale : float or None, default=0.1
        If not None, we sample a random shift for each dimension:
            shift_i ~ Normal(0, mean_shift_scale * abs(mean_i))
        Then add it to the empirical mean.
        If None, no random shift is applied.
    device : torch.device
        Device on which to place the resulting distribution's tensors.
    min_inflation : float, default=1e-4
        Minimum inflation added to the diagonal to prevent singular covariance.
    random_state : int, default=42
        Random seed for reproducible shift.

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

    # data_array shape = (n_params, n_patients)
    # We want mean, cov along columns => axis=1 is the "patient" dimension.
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
        # gen.manual_seed(random_state)  # set random seed for reproducibility
        # sample shift_i ~ Normal(0, mean_shift_scale * abs(mean_i))
        stddev_vec = mean_emp_t.abs() * mean_shift_scale

        shift_vec = torch.normal(mean=0.0, std=stddev_vec, generator=gen)
        mean_final_t = mean_emp_t + shift_vec
    else:
        mean_final_t = mean_emp_t

    # --- 4) Build the MVN distribution. ---
    final_mvn = MultivariateNormal(loc=mean_final_t, covariance_matrix=cov_inflated_t)

    # --- 5) Return as a Prior dataclass. ---
    return Prior(
        type="mvn",
        params_names=param_names,
        params_prior_distribution=final_mvn,
    )


def box_uniform_from_simglucose_patients(
    n_params: int, inflation_factor: float, device: torch.device
) -> Prior:
    """Creates a box uniform prior distribution from the known parameters of the simglucose patients.

    Parameters
    ----------
    n_params : int
        number of parameters to use. IF n_params < len(list_of_params), a random subset of parameters is selected.
    inflation_factor : float
        amount to inflate the parameter range by.
    device : torch.device
        device to store the tensors on.

    Returns
    -------
    Prior
        The dataclass containing the prior distribution.

    """
    if inflation_factor <= 1:
        print("Inflation factor should be greater than 1. Setting it to 1.")
        inflation_factor = 1
    with Path("all_patients_params.json").open() as f:
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

    return Prior(
        type="BoxUniform",
        params_names=selected_params,
        params_prior_distribution=BoxUniform(
            low=torch.tensor(
                [uniform_params[key][0] for key in selected_params], device=device
            ),
            high=torch.tensor(
                [uniform_params[key][1] for key in selected_params], device=device
            ),
        ),
    )


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
        The dataclass containing the prior distribution.

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
    if prior_type == "BoxUniform":
        return box_uniform_from_simglucose_patients(
            number_of_params, inflation_factor, device
        )
    msg = "Invalid prior type or not implemented yet."
    raise ValueError(msg)
