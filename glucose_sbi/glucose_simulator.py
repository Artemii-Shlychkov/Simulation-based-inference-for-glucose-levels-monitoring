import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
from pathos.multiprocessing import ProcessingPool as Pool
from simglucose.actuator.pump import InsulinPump
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj
from tqdm import tqdm

from glucose_sbi.prepare_priors import InferredParams

pathos = True

logger = logging.getLogger("sbi_logger.glucose_simulator")


@dataclass
class EnvironmentSettings:
    """Dataclass to store the initial presets of the simulation environment."""

    patient_name: str
    sensor_name: str
    pump_name: str
    scenario: list[tuple[int, int]] = field(default_factory=list)
    hours: int = 24  # hours to simulate


def run_glucose_simulator(
    theta: torch.Tensor,
    default_simulation_object: SimObj,
    inferred_params: InferredParams,
    *,
    hours: int = 24,
    device: torch.device,
    infer_meal_params: bool = False,
) -> torch.Tensor:
    """For every set of custom parameter values in theta, creates a modified corresponding simulation object with these parameters
    and runs the simulation for all of them in batch.

    Parameters
    ----------
    theta : torch.Tensor
        Sets of custom parameters to use for the simulation of shape (N_sets, N_params)
    default_simulation_object : SimObj
        The simulation object with all default presets like meal scenario and patient parameters, ready to `.simulate`
    inferred_params : InferredParams
        Dataclass object containing the names of inferred parameters
    hours : int, optional
        Duration of the simulation, by default 24
    device : torch.device
        Device used to store the results
    logger : logging.Logger, optional
        The logger object, by default None
    infer_meal_params : bool, optional
        Whether to infer meal parameters, by default False

    Returns
    -------
    torch.Tensor
        The glucose dynamics time series for each simulation

    """
    logger.info("Running the glucose simulator on theta of shape, %s", theta.shape)
    simulation_envs = create_simulation_objects_with_custom_params(
        theta=theta,
        default_simulation_object=default_simulation_object,
        inferred_params=inferred_params,
        hours=hours,
        infer_meal_params=infer_meal_params,
    )
    return simulate_batch(simulation_envs, device=device)


def simulate_batch(
    simulations: list[SimObj],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Simulates a batch of simulation objects in parallel.

    Parameters
    ----------
    simulations : list[SimObj]
        List of simulation objects with all the necessary presets and ready to `.simulate`
    device : torch.device
        The device to store the results on
    logger : logging.Logger, optional
        The logger object, by default None

    Returns
    -------
    torch.Tensor
        The tensor storing the resulting glucose dynamics for each simulation

    """
    pathos = True
    tic = time.time()
    if pathos:
        logger.info("Using pathos for parallel processing")
        with Pool() as p:
            results = p.map(simulate_glucose_dynamics, simulations)
    else:
        results = [simulate_glucose_dynamics(s) for s in tqdm(simulations)]
    results = np.stack(results)
    toc = time.time()

    logger.info("Simulation took %s seconds", toc - tic)
    return torch.from_numpy(results).float().to(device)


def simulate_glucose_dynamics(simulation_env: SimObj) -> np.ndarray:
    """Simulates the glucose dynamics for one given simulation object.

    Parameters
    ----------
    simulation_env : SimObj
        The simulation object with all the necessary presets and ready to `.simulate`

    Returns
    -------
    np.ndarray
        Resulting glucose dynamics

    """
    simulation_env.simulate()
    return simulation_env.results()["CGM"].to_numpy()


def create_simulation_objects_with_custom_params(
    theta: torch.Tensor,
    default_simulation_object: SimObj,
    inferred_params: InferredParams,
    *,
    hours: int = 24,
    infer_meal_params: bool = False,
) -> list[SimObj]:
    """Creates a list of simulation objecs with custom parameter values.
    The parameters that are inferred (listed in InferredParams dataclass)
    are updated with the values from theta.

    Parameters
    ----------
    theta : torch.Tensor
        A tensor of custom parameter values of shape (N_simulations, N_params) to be used for simulations.
    default_simulation_object : SimObj
        The simulation object with all default presets like meal scenario and patient parameters, ready to `.simulate`
    inferred_params : InferredParams
        Dataclass object containing the names of inferred parameters
    hours : int, optional
        Duration of each simulation, by default 24
    infer_meal_params : bool, optional
        Whether to infer meal parameters, by default False

    Returns
    -------
    list[SimObj]
        List of simulation objects with adjusted parameter values

    """
    simulation_objects = []

    for _, theta_i in enumerate(theta):
        custom_sim_obj = deepcopy(default_simulation_object)
        custom_sim_obj.sim_time = timedelta(hours=hours)
        set_custom_params(
            custom_sim_obj,
            theta_i,
            inferred_params,
            infer_meal_params=infer_meal_params,
        )
        simulation_objects.append(custom_sim_obj)

    return simulation_objects


def set_custom_params(
    default_simulation_obj: SimObj,
    theta: torch.Tensor,
    inferred_params: InferredParams,
    *,
    infer_meal_params: bool = False,
) -> None:
    """Change the default parameters of the patient and scenario in the simulation object
    for a given set of corresponding custom parameters (these are inferred).

    Parameters
    ----------
    default_simulation_obj : SimObj
        The simulation object containing the patient parameters and the meal scenario.
    theta : torch.Tensor
        One set of custom parameters to apply to the patient.
    inferred_params : InferredParams
        DataClass object containing the names of inferred parameters.
    infer_meal_params : bool, optional
        Whether to infer meal parameters, by default False

    """
    theta_list = theta.clone().tolist()
    param_names = inferred_params.params_names
    patient = default_simulation_obj.env.patient

    # Separate meal and non-meal parameters
    meal_indices, meal_values, other_params, other_values = _separate_parameters(
        param_names, theta_list
    )

    if infer_meal_params and meal_indices:
        # Update meal parameters in the scenario
        _update_meal_parameters(
            default_simulation_obj.env.scenario.scenario, meal_values
        )

    # Update other parameters in the patient
    _update_patient_parameters(patient, other_params, other_values)


def _separate_parameters(
    param_names: list[str], theta_list: list[float]
) -> tuple[list[int], list[float], list[str], list[float]]:
    """Separate meal and non-meal parameters."""
    meal_indices = [i for i, param in enumerate(param_names) if "meal" in param]
    meal_values = [theta_list[i] for i in meal_indices]

    non_meal_indices_and_params = [
        (i, param) for i, param in enumerate(param_names) if "meal" not in param
    ]
    other_params = [param for _, param in non_meal_indices_and_params]
    other_values = [theta_list[i] for i, _ in non_meal_indices_and_params]

    return meal_indices, meal_values, other_params, other_values


def _update_meal_parameters(
    scenario: list[tuple[str, float]], meal_values: list[float]
) -> None:
    """Update meal parameters in the scenario."""
    for i, (meal_name, _) in enumerate(scenario):
        scenario[i] = (meal_name, meal_values[i])


def _update_patient_parameters(
    patient: T1DPatient, params: list[str], values: list[float]
) -> None:
    """Update non-meal parameters in the patient."""
    for param, value in zip(params, values):
        setattr(patient._params, param, value)  # noqa: SLF001


def create_simulation_object(
    env_settings: EnvironmentSettings, hours: int = 24
) -> SimObj:
    """Creates the simulation object based on a patient, sensor, pump and scenario, specified in
    environment settings.

    Parameters
    ----------
    env_settings : EnvironmentSettings
        Dataclass object containing the basic simulation environment settings.
    hours : int, optional
        The number of hours to simulate, by default 24

    Returns
    -------
    SimObj
        The resulting simulation object with all the necessary presets and ready to `.simulate`

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


def generate_true_observation(
    default_simulation_object: SimObj, *, device: torch.device, hours: int = 24
) -> torch.Tensor:
    """Simulates glucose dynamics for default simulation environment settings
    and returns the single simulated glucose dynamics.

    Parameters
    ----------
    default_simulation_object : SimObj
        The simulation object with all default presets like meal scenario and patient parameters, ready to `.simulate`
    device : torch.device
        Device to save the simulation results on.
    hours : int, optional
        Duration of the simulation, by default 24

    Returns
    -------
    torch.Tensor:
        Resulting glucose levels dynamincs

    """
    default_simulation_object.sim_time = timedelta(hours=hours)
    default_simulation_object.simulate()
    true_observation = default_simulation_object.results()["CGM"].to_numpy()
    return torch.from_numpy(true_observation).float().to(device)
