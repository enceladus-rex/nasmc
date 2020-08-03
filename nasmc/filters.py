import torch
import torch.distributions as D

from nasmc.models import NonlinearSSMProposal


def smc(proposal: NonlinearSSMProposal, observation_model: D.Distribution, 
        state_transition_model: D.Distribution, observations: torch.Tensor, 
        num_particles: int):
    # observations: [batch_size, seq_len]
    batch_size, seq_len = observations.shape
    weights = []
    current_weights = torch.ones(batch_size, num_particles)
    lstm_state = None
    current_states = None
    for i in range(seq_len):
        # Sample new particles.
        current_observations = observations[:, i]
        if i == 0:
            d, lstm_state = proposal.parameterize_prior(current_observations[..., None])
        else:
            d, lstm_state = proposal.parameterize_posterior(current_states, current_observations[..., None], lstm_state)

        samples = d.sample((num_particles, ))

        observation_prob = observation_model.log_prob(current_observations)
        