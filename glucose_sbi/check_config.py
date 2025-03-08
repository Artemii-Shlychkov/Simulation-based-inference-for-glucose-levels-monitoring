import logging
from pathlib import Path

import yaml

logger = logging.getLogger("sbi_logger")


def check_config(config_file: Path) -> None:
    if not config_file.exists():
        msg = f"Config file {config_file} does not exist."
        logger.error(msg)
        raise FileNotFoundError(msg)

    # Load the configuration (use an empty dict if the file is empty)
    with config_file.open("r") as f:
        config = yaml.safe_load(f) or {}

    # Define default values for the configuration
    defaults = {
        "hours": 24,
        "infer_meal_params": False,
        "n_posterior_samples": 100,
        "patient_name": "adolescent#001",
        "prior_settings": {
            "priors_data_file": "all_sg_patients_params_values.json",
            "prior_type": "uniform",
            "number_of_params": 5,
        },
        "pump_name": "Insulet",
        "sbi_settings": {
            "algorithm": "BayesFlow",
            "num_simulations": 1000,
            "num_rounds": 1,
            "sample_proposal_with": "sir",
            "sampling_method": "direct",
        },
        "sensor_name": "Dexcom",
        "simulate_posterior_hours": 24,
    }

    def merge_defaults(user_conf: dict, default_conf: dict, key_path: str = "") -> None:
        """Recursively updates user_conf with keys and values from default_conf.
        Logs a warning for each missing key.
        """
        for key, default_val in default_conf.items():
            full_key = f"{key_path}.{key}" if key_path else key
            if key not in user_conf:
                logger.warning(
                    "%s not found in config file. Using default %r.",
                    full_key,
                    default_val,
                )
                user_conf[key] = default_val
            elif isinstance(default_val, dict) and isinstance(user_conf.get(key), dict):
                merge_defaults(user_conf[key], default_val, full_key)

    merge_defaults(config, defaults)

    # Write the updated configuration back to the file
    with config_file.open("w") as f:
        yaml.dump(config, f)
