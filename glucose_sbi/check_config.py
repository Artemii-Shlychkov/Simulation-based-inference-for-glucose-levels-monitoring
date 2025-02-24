from pathlib import Path

import yaml


def check_config(config_file: Path) -> None:
    assert config_file.exists(), f"Config file {config_file} does not exist."
    with Path(config_file).open("r") as f:
        config = yaml.safe_load(f)

    assert "patient_name" in config, "Config file must contain 'patient_name'."
    assert "sensor_name" in config, "Config file must contain 'sensor_name'."
    assert "pump_name" in config, "Config file must contain 'pump_name'."
    assert "scenario" in config, "Config file must contain 'scenario'."
    assert "hours" in config, "Config file must contain 'hours'."
    assert "prior_settings" in config, "Config file must contain 'prior_settings'."
    prior_setting = config["prior_settings"]
    assert "priors_data_file" in prior_setting, (
        "Config file prior_settings must contain 'priors_data_file'."
    )
    assert "prior_type" in prior_setting, (
        "Config file prior_settings must contain 'prior_type'."
    )
    assert "number_of_params" in prior_setting, (
        "Config file prior_settings must contain 'number_of_params'."
    )

    assert "sbi_settings" in config, "Config file must contain 'sbi_settings'."
    sbi_settings = config["sbi_settings"]
    assert "algorithm" in sbi_settings, (
        "Config file sbi_settings must contain 'algorithm'."
    )
    if sbi_settings["algorithm"] == "TSPE":
        assert "sampling_method" in sbi_settings, (
            "Config file sbi_settings must contain 'sampling_method' for TSNPE."
        )
    assert "num_simulations" in sbi_settings, (
        "Config file sbi_settings must contain 'num_simulations'."
    )
    assert "num_rounds" in sbi_settings, (
        "Config file sbi_settings must contain 'num_rounds'."
    )
    assert "n_samples_from_posterior" in sbi_settings, (
        "Config file sbi_settings must contain 'n_samples_from_posterior'."
    )

    assert "simulate_posterior_hours" in config, (
        "Config file must contain 'simulate_posterior_hours'."
    )
