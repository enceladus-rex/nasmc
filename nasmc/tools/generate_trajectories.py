import fire

import os
import torch

import pandas as pd

from nasmc.models import NonlinearSSM, NonlinearSSMProposal
from nasmc.datasets import NonlinearSSMDataset
from nasmc.filters import nonlinear_ssm_smc


def generate_trajectories(*, run_dir: str, csv_path: str, sequence_length: int = 1000, num_particles: int = 1000, cuda_id: int = -1, num_trajectories: int = 1):
    checkpoint_path = os.path.join(run_dir, 'checkpoint.pt')
    checkpoint = torch.load(checkpoint_path)
    
    proposal = NonlinearSSMProposal()
    model = NonlinearSSM()

    proposal.load_state_dict(checkpoint['proposal'])
    model.load_state_dict(checkpoint['model'])

    if cuda_id >= 0:
        proposal.cuda(cuda_id)
        model.cuda(cuda_id)

    data = []
    data_loader = DataLoader(NonlinearSSMDataset(model, sequence_length=sequence_length), batch_size=1, num_workers=1)
    for i, example in zip(range(num_trajectories), data_loader):
        observations = example.observations
        if cuda_id >= 0:
            observations = observations.cuda(cuda_id)

        smc_result = nonlinear_ssm_smc(proposal, model,
                                       observations, num_particles)

        mean_intermediate_states = F.sum(smc_result.intermediate_weights * smc_result.intermediate_trajectories, dim=0)[0, :, 0]
        raw_states = mean_intermediate_states.flatten().tolist()
        true_states = example.states.flatten().tolist()
        for j, (s, v) in enumerate(zip(raw_states, true_states)):
            data.append(dict(trajectory=i+1, timestep=j+1, mean_prediction=s, true_value=v))
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)


if __name__ == '__main__':
    fire.Fire(generate_trajectories)
