import torch
import torch.distributions as D

from math import sqrt

from typing import NamedTuple

from nasmc.models import NonlinearSSM


class NonlinearSSMExample(NamedTuple):
    states: torch.Tensor
    observations: torch.Tensor


class NonlinearSSMDataset(torch.utils.data.IterableDataset):
    def __init__(self, model: NonlinearSSM, sequence_length: int = 1000):
        super().__init__()
        assert sequence_length > 0, 'sequence length must be greater than 0'
        self.sequence_length = sequence_length
        self.model = model

    def __iter__(self):
        while True:
            state_prior = self.model.parameterize_prior_model()
            state = state_prior.sample()

            observation_sequence = []
            state_sequence = []
            for timestep in range(2, self.sequence_length + 2):
                observation_dist = self.model.parameterize_observation_model(
                    state)
                observation = observation_dist.sample()

                observation_sequence.append(observation)
                state_sequence.append(state)

                state_dist = self.model.parameterize_transition_model(
                    state, timestep)
                state = state_dist.sample()

            yield NonlinearSSMExample(
                states=torch.stack(state_sequence),
                observations=torch.stack(observation_sequence))
