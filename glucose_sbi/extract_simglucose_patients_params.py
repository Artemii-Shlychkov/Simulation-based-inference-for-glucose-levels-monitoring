import json
from pathlib import Path

from simglucose.patient.t1dpatient import T1DPatient

patient = T1DPatient.withName("adolescent#001")
param_dict = patient._params.to_dict()  # noqa: SLF001
for key, value in param_dict.items():
    param_dict[key] = [value]


ixd = ["02", "03", "04", "05", "06", "07", "08", "09", "10"]
for i in ixd:
    patient = T1DPatient.withName(f"adolescent#0{i}")
    for key, value in patient._params.to_dict().items():  # noqa: SLF001
        param_dict[key].append(value)
    patient = T1DPatient.withName(f"adult#0{i}")
    for key, value in patient._params.to_dict().items():  # noqa: SLF001
        param_dict[key].append(value)
    patient = T1DPatient.withName(f"child#0{i}")
    for key, value in patient._params.to_dict().items():  # noqa: SLF001
        param_dict[key].append(value)

# some params are not used or have bad names
for key in [
    "Name",
    "x0_ 1",
    "x0_ 2",
    "x0_ 3",
    "x0_ 4",
    "x0_ 5",
    "x0_ 6",
    "x0_ 7",
    "x0_ 8",
    "x0_ 9",
    "f",
    "ke1",
    "ke2",
    "Fsnc",
    "dosekempt",
    "patient_history",
    "i",
]:
    param_dict.pop(key)

with Path("all_sg_patients_params_values.json").open("w") as f:
    json.dump(param_dict, f)
