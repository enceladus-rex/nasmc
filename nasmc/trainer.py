import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from torch import autograd

import time
import os
import fire
import random

import numpy as np

from nasmc.models import NonlinearSSM, NonlinearSSMProposal
from nasmc.datasets import NonlinearSSMDataset
from nasmc.filters import nonlinear_ssm_smc


class NASMCTrainer:
    def run(self,
            run_dir: str = './runs/',
            lr: float = 3e-3,
            num_steps: int = 1,
            save_decimation: int = 100,
            num_particles: int = 1000,
            sequence_length: int = 50,
            batch_size: int = 1,
            cuda_id: int = -1,
            seed: int = 95):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        os.makedirs(run_dir, exist_ok=True)
        checkpoint_path = os.path.join(run_dir, 'checkpoint.pt')

        proposal = NonlinearSSMProposal()
        model = NonlinearSSM()
        optimizer = torch.optim.Adam(proposal.parameters(), lr=lr)
        step = 1

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            proposal.load_state_dict(checkpoint['proposal'])
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            step = checkpoint['step']
            num_particles = checkpoint['num_particles']
            sequence_length = checkpoint['sequence_length']

        proposal.train()
        model.train()

        if cuda_id >= 0:
            proposal.cuda(cuda_id)
            model.cuda(cuda_id)

        summary_writer = SummaryWriter(run_dir)

        dl = DataLoader(NonlinearSSMDataset(model, sequence_length),
                        batch_size=batch_size,
                        num_workers=1)
        for i, example in zip(range(num_steps), dl):
            observations = example.observations
            if cuda_id >= 0:
                observations = observations.cuda(cuda_id)

            smc_result = nonlinear_ssm_smc(proposal, model, observations,
                                           num_particles)

            loss = -torch.sum(
                smc_result.intermediate_weights *
                smc_result.intermediate_proposal_log_probs) / batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            summary_writer.add_scalar('loss/train', loss, step)

            print('time =', time.time(), '- step =', step, '- loss =', loss)

            step += 1
            if step % save_decimation == 0:
                torch.save(
                    dict(proposal=proposal.state_dict(),
                         model=model.state_dict(),
                         optimizer=optimizer.state_dict(),
                         step=step,
                         num_particles=num_particles,
                         sequence_length=sequence_length), checkpoint_path)

        summary_writer.flush()

        torch.save(
            dict(proposal=proposal.state_dict(),
                 model=model.state_dict(),
                 optimizer=optimizer.state_dict(),
                 step=step,
                 num_particles=num_particles,
                 sequence_length=sequence_length), checkpoint_path)


if __name__ == '__main__':
    fire.Fire(NASMCTrainer)
