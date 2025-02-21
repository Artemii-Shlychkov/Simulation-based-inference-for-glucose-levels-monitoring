# Parameter Inference for Glucose Simulation
## Requirements

### Dependencies
Install necessary packages by running
```bash
pip install -r requirements.txt
```

For GPU computations make sure `torch` supports cuda by checking if
```bash
torch.cuda.is_available()
```
returns `True` in Python

## Main module `infer_parameters`
This module performs parameter inference using different Neural Posterior Estimation (NPE) techniques for glucose dynamics simulation. It utilizes the `sbi` package for Bayesian inference, `simglucose` for simulating glucose dynamics, and `matplotlib`, `numpy`, and `torch` for analysis and computation.

## Features
- Supports multiple inference algorithms: TSNPE, APT, and BayesFlow.
- Simulates glucose dynamics using patient parameters.
- Saves posterior distributions and inferred parameter samples.
- Evaluates performance using Mean Squared Error (MSE).
- Saves experimental setup and inference results.

## Usage

### Running the Script
To run the parameter inference:
```bash
python script.py --config <config_file> [--simulate_with_posterior] [--plot]
```

### Arguments
- `--config <config_file>`: Specifies the configuration file for simulation settings. Default: `test_config.yaml`.
- `--simulate_with_posterior`: Enables simulation with the inferred posterior distribution.
- `--plot`: Generates and saves plots of simulation results.

### Example
```bash
python script.py --config test_config.yaml --simulate_with_posterior --plot
```

## Configuration
The script uses a configuration YAML file specifying the prior distributions, simulation settings, and inference parameters. The configuration file must be located in a folder `simulation_configs` in the package root. 
Refer to a `test_config.yaml` for an example


## Output
The script generates the following outputs:
- `results/<timestamp>/posterior_distribution.pkl`: Pickle file containing the inferred posterior distribution.
- `results/<timestamp>/posterior_samples.pkl`: Pickle file containing sampled posterior parameter values.
- `results/<timestamp>/glucose_dynamics.pkl`: Simulated glucose dynamics from the inferred posterior.
- `results/<timestamp>/simulation_results.png`: Plot comparing true and inferred glucose dynamics (if `--plot` is enabled).
- `results/<timestamp>/simulation_config.yaml`: YAML file storing metadata of the inference session.

## Logging
The script logs all operations in `results/<timestamp>/inference_execution.log`.

# Module `prepare_priors`

The prepare_priors.py script is responsible for constructing and managing prior distributions for parameter inference, based on some domain knowledge like the range of parameter values or their mean / std. 
Available data to date is extracted from the simglucose "patients" and stored in the `all_sg_patients_params_values.json`. Based on this data, the module provides functions to:

- Select a random subset of patient parameters for inference.

- Construct different types of prior distributions, including multivariate normal (MVN), log-normal, and box-uniform priors.

- Apply transformations such as covariance inflation and mean shifting to enhance parameter variability.

# Module `process_results`

The process_results.py script is responsible for loading, processing, and visualizing the inference results. It provides functions to:

- Load experiment results, including priors, true observations, posterior distributions, and inferred samples.

- Simulate glucose dynamics using both true and inferred parameters.

- Generate visual comparisons of true and inferred glucose levels.

- Compute and display mean squared error (MSE) between true and inferred simulations.

# Module `glucose_simulator`
This module contains logic to run simglucose simulations with given sets of patient parameters (theta), different from default ones.



