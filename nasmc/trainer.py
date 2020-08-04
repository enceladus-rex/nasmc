import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import os
import fire

from nasmc.models import NonlinearSSM, NonlinearSSMProposal
from nasmc.datasets import NonlinearSSMDataset
from nasmc.filters import nonlinear_ssm_smc


class NASMCTrainer:
    def run(self,
            run_dir: str = './runs/',
            lr: float = 1e-3,
            num_steps: int = 1000,
            save_decimation: int = 100,
            num_particles: int = 100,
            sequence_length: int = 1000,
            batch_size: int = 32):
        os.makedirs(run_dir, exist_ok=True)
        checkpoint_path = os.path.join(run_dir, 'checkpoint.pt')

        proposal = NonlinearSSMProposal()
        model = NonlinearSSM()
        optimizer = torch.optim.Adam(proposal.parameters(), lr=lr)
        step = 1

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            proposal = checkpoint['proposal']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            step = checkpoint['step']
            num_particles = checkpoint['num_particles']
            sequence_length = checkpoint['sequence_length']

        proposal.train()
        model.train()

        summary_writer = SummaryWriter(run_dir)

        dl = DataLoader(NonlinearSSMDataset(model, sequence_length),
                        batch_size=batch_size)
        for i, example in zip(range(num_steps), dl):
            smc_result = nonlinear_ssm_smc(proposal, model,
                                           example.observations, num_particles)

            smc_result.weights.detach()
            loss = F.sum(smc_result.weights *
                         smc_result.proposal_log_probs) / batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            summary_writer.write('loss/train', loss, step)

            print('step =', step, 'loss =', loss)

            step += 1
            if step % save_decimation == 0:
                torch.save(
                    dict(proposal=proposal,
                         model=model,
                         optimizer=optimizer,
                         step=step,
                         num_particles=num_particles,
                         sequence_length=sequence_length), checkpoint_path)

        summary_writer.flush()

        torch.save(
            dict(proposal=proposal,
                 model=model,
                 optimizer=optimizer,
                 step=step,
                 num_particles=num_particles,
                 sequence_length=sequence_length), checkpoint_path)


if __name__ == '__main__':
    fire.Fire(NASMCTrainer)
