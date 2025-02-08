import json
import pickle
from pathlib import Path

import numpy as np
import torch
from simglucose.patient.t1dpatient import T1DPatient
from torch.distributions import MultivariateNormal, TransformedDistribution
from torch.distributions.transforms import ExpTransform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
patient = T1DPatient.withName("adolescent#001")
param_dict = patient._params.to_dict()
for key, value in param_dict.items():
    param_dict[key] = [value]


ixd = ["02", "03", "04", "05", "06", "07", "08", "09", "10"]
for i in ixd:
    patient = T1DPatient.withName(f"adolescent#0{i}")
    for key, value in patient._params.to_dict().items():
        param_dict[key].append(value)
    patient = T1DPatient.withName(f"adult#0{i}")
    for key, value in patient._params.to_dict().items():
        param_dict[key].append(value)
    patient = T1DPatient.withName(f"child#0{i}")
    for key, value in patient._params.to_dict().items():
        param_dict[key].append(value)


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

with Path("param_dict.json").open("w") as f:
    json.dump(param_dict, f)

data = np.array([param_dict[key] for key in param_dict])

# mean and covariance
mean = np.mean(data, axis=1)
cov = np.cov(data)
cov = cov + np.eye(cov.shape[0]) * 1e-4

mean = torch.tensor(mean, dtype=torch.float32, device=device)
cov = torch.tensor(cov, dtype=torch.float32, device=device)


# create multivariate normal
mvn = MultivariateNormal(loc=mean, covariance_matrix=cov)

pos_mvn = TransformedDistribution(mvn, ExpTransform())
with Path("pos_mvn_prior.pkl").open("wb") as f:
    pickle.dump(pos_mvn, f)

print("Constructed multivariate normal prior with Cov matrix of shape:", cov.shape)
with Path("mvn_prior.pkl").open("wb") as f:
    pickle.dump(mvn, f)


def create_uniform_prior(param_dict):
    for key, value in param_dict.items():
        max_val = max(value)
        min_val = min(value)
        param_dict[key] = (min_val, max_val)
    return param_dict


param_dict = create_uniform_prior(param_dict)
