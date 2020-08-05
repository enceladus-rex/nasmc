import torch
import torch.distributions as D

from typing import Tuple, NamedTuple

from nasmc.models import NonlinearSSMProposal, NonlinearSSM


class SMCResult(NamedTuple):
    intermediate_weights: torch.Tensor
    intermediate_proposal_log_probs: torch.Tensor
    intermediate_trajectories: torch.Tensor
    final_trajectories: torch.Tensor
    final_weights: torch.Tensor


def nonlinear_ssm_smc(proposal: NonlinearSSMProposal, model: NonlinearSSM,
                      observations: torch.Tensor,
                      num_particles: int) -> SMCResult:
    # Dimensionality Analysis
    #
    # Observations: [batch_size, sequence_length, obs_dim]
    # Weights: [num_particles, batch_size, num_timesteps, 1]
    # Final Weights: [num_particles, batch_size, 1]
    # LSTM State: [num_particles, batch_size, hidden_dim]
    # Intermediate Trajectories: [num_particles, batch_size, num_timesteps, state_dim]
    # Final Trajectories: [num_particles, batch_size, num_timesteps, state_dim]
    # Current States: [num_particles, batch_size, state_dim]
    # Current Observations: [num_particles, batch_size, obs_dim]
    # Proposal Log Probabilities: [num_particles, batch_size, num_timesteps, 1]

    batch_size, seq_len, _ = observations.shape

    current_observations = observations[None, :,
                                        0, :].repeat(num_particles, 1, 1)
    proposal_d, lstm_state = proposal.parameterize_prior(current_observations.detach())
    transition_model = model.parameterize_prior_model(device=observations.device)
    current_states = proposal_d.sample()

    observation_model = model.parameterize_observation_model(current_states)

    transition_prob = transition_model.log_prob(current_states).exp()
    proposal_prob = proposal_d.log_prob(current_states)[..., None].exp()
    observation_prob = observation_model.log_prob(current_observations).exp()

    weights = transition_prob / (proposal_prob + 1e-8)
    weights = weights / (torch.sum(weights, dim=0, keepdims=True) + 1e-7)
    weights = weights[..., None, :]

    proposal_log_probs = proposal_d.log_prob(current_states)
    proposal_log_probs = proposal_log_probs[..., None, None]

    intermediate_trajectories = current_states[..., None, :]
    final_trajectories = current_states[..., None, :]

    for i in range(1, seq_len):
        current_observations = observations[None, :,
                                            i, :].repeat(num_particles, 1, 1)
        sampling_weights = weights[..., i - 1, 0].permute(1, 0)
        ancestor_d = D.Categorical(sampling_weights)
        ancestors = ancestor_d.sample((num_particles, ))

        resampled_trajectories = torch.gather(
            final_trajectories, 0,
            ancestors[..., None, None].repeat(1, 1, final_trajectories.shape[-2],
                                              final_trajectories.shape[-1]))
        resampled_lstm_h = torch.gather(
            lstm_state[0], 0,
            ancestors[..., None].repeat(1, 1, lstm_state[0].shape[-1]))
        resampled_lstm_c = torch.gather(
            lstm_state[1], 0,
            ancestors[..., None].repeat(1, 1, lstm_state[1].shape[-1]))
        resampled_lstm_state = (resampled_lstm_h, resampled_lstm_c)

        previous_states = resampled_trajectories[..., i - 1, :]

        proposal_d, lstm_state = proposal.parameterize_posterior(
            previous_states, current_observations.detach(), resampled_lstm_state)
        transition_model = model.parameterize_transition_model(
            previous_states, i + 1)
        current_states = proposal_d.sample()

        observation_model = model.parameterize_observation_model(
            current_states)

        transition_prob = transition_model.log_prob(current_states).exp()
        proposal_prob = proposal_d.log_prob(current_states)[..., None].exp()
        observation_prob = observation_model.log_prob(current_observations).exp()

        current_weights = transition_prob * observation_prob / (proposal_prob + 1e-8)
        current_weights = current_weights / (torch.sum(
            current_weights, dim=0, keepdims=True) + 1e-7)
        current_weights = current_weights[..., None, :]
        total_weights = torch.sum(current_weights, dim=0, keepdims=True)
        valid_weights = total_weights > 1e-2
        current_weights = current_weights.where(valid_weights, torch.ones_like(current_weights) / num_particles)

        current_trajectories = current_states[..., None, :]

        current_proposal_log_probs = proposal_d.log_prob(current_states)
        current_proposal_log_probs = current_proposal_log_probs[..., None,
                                                                None]

        weights = torch.cat([weights, current_weights], dim=-2)
        intermediate_trajectories = torch.cat(
            [intermediate_trajectories, current_trajectories], dim=-2)
        final_trajectories = torch.cat(
            [resampled_trajectories, current_trajectories], dim=-2)
        proposal_log_probs = torch.cat(
            [proposal_log_probs, current_proposal_log_probs], dim=-2)

    return SMCResult(
            intermediate_weights=weights.detach(), 
            intermediate_proposal_log_probs=proposal_log_probs, 
            intermediate_trajectories=intermediate_trajectories, 
            final_trajectories=final_trajectories,
            final_weights=weights[..., 0, :])
