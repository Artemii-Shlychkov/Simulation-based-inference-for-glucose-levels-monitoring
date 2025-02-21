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

from glucose_sbi.prepare_priors import Prior

pathos = True


@dataclass
class DeafultSimulationEnv:
    """Dataclass for the default simulation environment."""

    patient_name: str
    sensor_name: str
    pump_name: str
    scenario: list[tuple[int, int]] = field(default_factory=list)
    hours: int = 24  # hours to simulate


def run_glucose_simulator(
    theta: torch.Tensor,
    default_settings: DeafultSimulationEnv,
    prior: Prior,
    device: torch.device,
    hours: int = 24,
    logger: logging.Logger | None = None,
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
        Device used to store the results, by default torch.device("cpu")
    logger : logging.Logger, optional
        The logger object, by default None

    Returns
    -------
    torch.Tensor
        The glucose dynamics time series for each simulation

    """
    if logger:
        logger.info("Running the glucose simulator on theta of shape, %s", theta.shape)
    simulation_envs = create_simulation_envs_with_custom_params(
        theta=theta,
        default_settings=default_settings,
        prior=prior,
        hours=hours,
    )
    return simulate_batch(simulation_envs, device, logger)


def simulate_batch(
    simulations: list[T1DSimEnv],
    device: torch.device,
    logger: logging.Logger | None = None,
) -> torch.Tensor:
    """Simulate a batch of simulation environments in parallel.

    Parameters
    ----------
    simulations : list[T1DSimEnv]
        List of simulation environments
    device : torch.device
        The device to store the results on, by default torch.device("cpu")
    logger : logging.Logger, optional
        The logger object, by default None

    Returns
    -------
    torch.Tensor
        The glucose dynamics for each simulation

    """
    pathos = True
    tic = time.time()
    if pathos:
        if logger:
            logger.info("Using pathos for parallel processing")
        with Pool() as p:
            results = p.map(simulate_glucose_dynamics, simulations)
    else:
        results = [simulate_glucose_dynamics(s) for s in tqdm(simulations)]
    results = np.stack(results)
    toc = time.time()
    if logger:
        # log in seconds
        logger.info("Simulation took %s seconds", toc - tic)
    return torch.from_numpy(results).float().to(device)


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
