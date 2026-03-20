from __future__ import annotations
import torch.nn as nn
import torch.optim as optim


class A2C:
    def __init__(
        self,
        actor_critic,
        value_loss_coef,
        entropy_coef,
        lr=None,
        eps=None,
        alpha=None,
        max_grad_norm=None,
    ):
        self.actor_critic = actor_critic
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.RMSprop(
            actor_critic.parameters(),
            lr,
            eps=eps,
            alpha=alpha,
        )

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        critic_estimates, policy_values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(
                -1,
                self.actor_critic.recurrent_hidden_state_size,
            ),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape),
        )

        critic_estimates = critic_estimates.view(num_steps, num_processes, 1)
        policy_values = policy_values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        critic_targets = rollouts.returns[:-1]
        critic_loss = (critic_targets - critic_estimates).pow(2).mean()

        if self.actor_critic.critic_type == "q":
            actor_signal = critic_estimates - policy_values
        else:
            actor_signal = critic_targets - policy_values

        action_loss = -(actor_signal.detach() * action_log_probs).mean()

        self.optimizer.zero_grad()
        (critic_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        actor_signal_variance = actor_signal.detach().var(unbiased=False).item()
        critic_target_variance = critic_targets.detach().var(unbiased=False).item()

        return {
            "critic_loss": critic_loss.item(),
            "action_loss": action_loss.item(),
            "dist_entropy": dist_entropy.item(),
            "actor_signal_variance": actor_signal_variance,
            "critic_target_variance": critic_target_variance,
        }
