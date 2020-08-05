import fire

import os
import torch

from torch.utils.data import DataLoader

import pandas as pd

from nasmc.models import NonlinearSSM, NonlinearSSMProposal
from nasmc.datasets import NonlinearSSMDataset
from nasmc.filters import nonlinear_ssm_smc


def generate_trajectories(*,
                          run_dir: str,
                          csv_path: str,
                          sequence_length: int = 1000,
                          num_particles: int = 1000,
                          device_name: str = 'cpu',
                          num_trajectories: int = 1):
    device = torch.device(device_name)
    checkpoint_path = os.path.join(run_dir, 'checkpoint.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)

    proposal = NonlinearSSMProposal()
    model = NonlinearSSM()

    proposal.to(device)
    model.to(device)

    proposal.load_state_dict(checkpoint['proposal'])
    model.load_state_dict(checkpoint['model'])

    data = []
    data_loader = DataLoader(NonlinearSSMDataset(
        model, sequence_length=sequence_length),
                             batch_size=1,
                             num_workers=1)
    for i, example in zip(range(num_trajectories), data_loader):
        observations = example.observations
        observations = observations.to(device)

        smc_result = nonlinear_ssm_smc(proposal, model, observations,
                                       num_particles)

        mean_intermediate_states = torch.sum(
            smc_result.intermediate_weights *
            smc_result.intermediate_trajectories,
            dim=0)[0, :, 0]
        raw_states = mean_intermediate_states.flatten().tolist()
        true_states = example.states.flatten().tolist()
        for j, (s, v) in enumerate(zip(raw_states, true_states)):
            data.append(
                dict(trajectory=i + 1,
                     timestep=j + 1,
                     mean_prediction=s,
                     true_value=v))

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)


if __name__ == '__main__':
    fire.Fire(generate_trajectories)
