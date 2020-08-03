import torch
import torch.distributions as D

from math import sqrt

from typing import NamedTuple

from nasmc.models import (nonlinear_ssm_observation_model, nonlinear_ssm_transition_model, nonlinear_ssm_transition_prior)


class NonlinearSSMExample(NamedTuple):
    states: torch.Tensor
    observations: torch.Tensor


class NonlinearSSMDataset(torch.utils.data.IterableDataset):
    def __init__(self, sigma_v: float = sqrt(10), sigma_w: float = 1., sequence_length: int = 1000):
        super().__init__()
        assert sequence_length > 0, 'sequence length must be greater than 0'
        self.state_variance = sigma_v * sigma_v
        self.observation_variance = sigma_w * sigma_w
        self.sequence_length = sequence_length

    def __iter__(self):
        while True:
            state_prior = nonlinear_ssm_transition_prior()
            state = state_prior.sample()

            observation_sequence = []
            state_sequence = []
            for timestep in range(2, self.sequence_length + 2):
                observation_dist = nonlinear_ssm_observation_model(state)
                observation = observation_dist.sample()

                observation_sequence.append(observation)
                state_sequence.append(state)

                state_dist = nonlinear_ssm_transition_model(state, timestep)
                state = state_dist.sample()

            yield NonlinearSSMExample(states=torch.stack(state_sequence), observations=torch.stack(observation_sequence))
