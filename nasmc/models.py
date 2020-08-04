import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from typing import Tuple
from math import sqrt

from nasmc.nn import batch_apply_rnn


class NonlinearSSMProposal(nn.Module):
    def __init__(self, num_mixtures: int = 3, hidden_size: int = 50):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.hidden_size = hidden_size
        self.hidden_sequential = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
        )

        self.context_sequential = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
        )

        self.mixture_linear = nn.Linear(50, num_mixtures)
        self.means_linear = nn.Linear(50, num_mixtures)
        self.stddev_linear = nn.Linear(50, num_mixtures)
        self.lstm_cell = nn.LSTMCell(2, 50)

    def _parameterize_distribution(self,
                                   hidden: torch.Tensor) -> D.Distribution:
        mixture_logits = self.mixture_linear(hidden)
        mixture = F.softmax(mixture_logits, dim=-1)

        means = self.means_linear(hidden)[..., None]

        stddev = F.softplus(self.stddev_linear(hidden))[..., None]

        c = D.Categorical(probs=mixture)
        n = D.Independent(D.Normal(means, stddev), 1)

        return D.MixtureSameFamily(c, n)

    def parameterize_prior(
        self, observation: torch.Tensor
    ) -> Tuple[D.Distribution, Tuple[torch.Tensor, torch.Tensor]]:
        h0, c0 = self.hidden_sequential(observation), self.context_sequential(
            observation)
        return self._parameterize_distribution(h0), (h0, c0)

    def parameterize_posterior(
        self, previous_state: torch.Tensor, observation: torch.Tensor,
        lstm_state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[D.Distribution, Tuple[torch.Tensor, torch.Tensor]]:
        lstm_input = torch.cat([previous_state, observation], axis=-1)
        hnext, cnext = batch_apply_rnn(lstm_input, lstm_state, self.lstm_cell)
        return self._parameterize_distribution(hnext), (hnext, cnext)


class NonlinearSSM(nn.Module):
    def parameterize_observation_model(self,
                                       state: torch.Tensor,
                                       sigma: float = 1.) -> D.Distribution:
        return D.Normal(state * state / 20., sigma)

    def parameterize_prior_model(self, device=None):
        return D.Normal(torch.tensor([0.], device=device), torch.tensor([sqrt(5)], device=device))

    def parameterize_transition_model(
        self, state: torch.Tensor, timestep: int,
        sigma: float = sqrt(10.)) -> D.Distribution:
        return D.Normal(
            state / 2. + 25 * state / (1 + state * state) +
            8 * torch.cos(1.2 * torch.as_tensor(timestep, device=state.device)), sigma)
