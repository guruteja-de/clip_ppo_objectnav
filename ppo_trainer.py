import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from networks import ActorCritic


class RolloutBuffer:
    """
    Stores one rollout of experience for PPO update.

    One rollout = n_steps transitions of:
    (obs, action, log_prob, reward, value, done)

    After n_steps collected:
    1. Compute GAE advantages
    2. Normalize advantages
    3. Yield mini-batches for PPO update
    4. Reset buffer for next rollout
    """

    def __init__(self, n_steps: int, obs_dim: int):
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.reset()

    def reset(self):
        """Clear buffer for next rollout."""
        self.obs       = np.zeros((self.n_steps, self.obs_dim), dtype=np.float32)
        self.actions   = np.zeros(self.n_steps, dtype=np.int64)
        self.log_probs = np.zeros(self.n_steps, dtype=np.float32)
        self.rewards   = np.zeros(self.n_steps, dtype=np.float32)
        self.values    = np.zeros(self.n_steps, dtype=np.float32)
        self.dones     = np.zeros(self.n_steps, dtype=np.float32)
        self.ptr       = 0

    def store(self, obs, action, log_prob, reward, value, done):
        """Store one transition. Called every env step."""
        assert self.ptr < self.n_steps, "Buffer full!"
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr]   = reward
        self.values[self.ptr]    = value
        self.dones[self.ptr]     = float(done)
        self.ptr                += 1

    def is_full(self) -> bool:
        return self.ptr >= self.n_steps

    def _compute_gae(
        self,
        last_value: float,
        gamma:      float,
        gae_lambda: float,
    ):
        """
        Generalized Advantage Estimation (GAE).

        Why GAE?
        --------
        Simple return (MC):   high variance, unbiased
        TD error (1-step):    low variance, biased
        GAE(lambda):          balances both via lambda

        Formula:
            δ_t  = r_t + γ * V(s_{t+1}) * (1-done) - V(s_t)
            A_t  = δ_t + γλ * (1-done) * A_{t+1}
            R_t  = A_t + V(s_t)   ← critic targets

        Lambda controls tradeoff:
            λ=0 → pure TD (low variance, high bias)
            λ=1 → pure MC (high variance, low bias)
            λ=0.95 → sweet spot (our choice)
        """
        advantages = np.zeros(self.n_steps, dtype=np.float32)
        gae        = 0.0

        for t in reversed(range(self.n_steps)):
            # If done → next state is new episode → no future value
            not_terminal = 1.0 - self.dones[t]

            # Next state value
            if t == self.n_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            next_value *= not_terminal

            # TD error
            delta = self.rewards[t] + gamma * next_value - self.values[t]

            # GAE recursion
            gae          = delta + gamma * gae_lambda * not_terminal * gae
            advantages[t] = gae

        # Critic targets = advantages + values
        returns = advantages + self.values
        return advantages, returns

    def get_dataset(
        self,
        last_value: float,
        gamma:      float,
        gae_lambda: float,
        device,
    ) -> dict:
        """
        Compute GAE, normalize advantages, convert to tensors.
        Called once after buffer is full, before PPO update.
        """
        advantages, returns = self._compute_gae(last_value, gamma, gae_lambda)

        # Normalize advantages — critical for stable PPO training
        # Zero mean, unit variance
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "obs":           torch.FloatTensor(self.obs).to(device),
            "actions":       torch.LongTensor(self.actions).to(device),
            "old_log_probs": torch.FloatTensor(self.log_probs).to(device),
            "advantages":    torch.FloatTensor(advantages).to(device),
            "returns":       torch.FloatTensor(returns).to(device),
        }


class PPOTrainer:
    """
    Wraps ActorCritic + RolloutBuffer + optimizer.

    PPO Update (every n_steps):
    1. Compute GAE advantages from buffer
    2. For n_epochs:
         - Shuffle data into mini-batches
         - Compute policy ratio = π_new / π_old
         - Clip ratio to [1-ε, 1+ε]   ← "Proximal" part
         - Policy loss  = -min(ratio*A, clip(ratio)*A)
         - Value loss   = MSE(V(s), returns)
         - Entropy loss = -entropy  (encourages exploration)
         - Total loss   = policy + c1*value - c2*entropy
         - Backprop + clip gradients + step optimizer
    3. Reset buffer
    """

    def __init__(self, network: ActorCritic, config: Config, device):
        self.network = network.to(device)
        self.config  = config
        self.device  = device
        self.obs_dim = network.trunk[0].in_features

        self.optimizer = torch.optim.Adam(
            network.parameters(),
            lr  = config.lr,
            eps = 1e-5,
        )
        self.buffer = RolloutBuffer(config.n_steps, self.obs_dim)
        self._total_steps = 0

    @torch.no_grad()
    def select_action(self, obs: np.ndarray):
        """
        Sample action from current policy.
        Called every env step during rollout.

        Returns:
            action   (int)
            log_prob (float)
            value    (float)
        """
        obs_t          = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action, log_prob, _, value = self.network.get_action_and_value(obs_t)
        return action.item(), log_prob.item(), value.item()

    def update(self, last_obs: np.ndarray, last_done: bool) -> dict:
        """
        Run PPO update using current buffer contents.

        Args:
            last_obs:  observation AFTER last stored step (for bootstrapping)
            last_done: whether last step was terminal

        Returns:
            dict of training metrics
        """
        # Bootstrap value for GAE
        # If last step was terminal → future value = 0
        with torch.no_grad():
            if last_done:
                last_value = 0.0
            else:
                last_value = self.network.get_value(
                    torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
                ).item()

        # Build dataset — GAE + normalize + tensorify
        data = self.buffer.get_dataset(
            last_value,
            self.config.gamma,
            self.config.gae_lambda,
            self.device,
        )

        # PPO update over multiple epochs
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        n_updates         = 0

        for epoch in range(self.config.n_epochs):
            # Shuffle data each epoch
            indices = torch.randperm(self.config.n_steps, device=self.device)

            for start in range(0, self.config.n_steps, self.config.batch_size):
                idx = indices[start : start + self.config.batch_size]

                batch_obs      = data["obs"][idx]
                batch_actions  = data["actions"][idx]
                batch_old_lp   = data["old_log_probs"][idx]
                batch_adv      = data["advantages"][idx]
                batch_returns  = data["returns"][idx]

                # Forward pass under CURRENT policy
                _, new_log_probs, entropy, values = \
                    self.network.get_action_and_value(batch_obs, batch_actions)

                # ── Policy Loss (PPO-Clip) ────────────────────────
                # ratio = π_new(a|s) / π_old(a|s)
                ratio = torch.exp(new_log_probs - batch_old_lp)

                # Unclipped objective
                pg_loss1 = -batch_adv * ratio

                # Clipped objective — limits policy update size
                pg_loss2 = -batch_adv * torch.clamp(
                    ratio,
                    1.0 - self.config.clip_eps,
                    1.0 + self.config.clip_eps,
                )
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # ── Value Loss ────────────────────────────────────
                # MSE between critic prediction and GAE returns
                value_loss = F.mse_loss(values, batch_returns)

                # ── Total Loss ────────────────────────────────────
                # Entropy bonus → encourages exploration
                loss = (
                    policy_loss
                    + self.config.value_coef   * value_loss
                    - self.config.entropy_coef * entropy.mean()
                )

                # ── Backprop ──────────────────────────────────────
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.mean().item()
                n_updates         += 1

        # ── Linear LR annealing ───────────────────────────────────
        # Slowly reduce learning rate to 0 over total training
        self._total_steps += self.config.n_steps
        frac = 1.0 - self._total_steps / self.config.total_timesteps
        for g in self.optimizer.param_groups:
            g["lr"] = max(self.config.lr * frac, 1e-6)

        # Reset buffer for next rollout
        self.buffer.reset()

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss":  total_value_loss  / n_updates,
            "entropy":     total_entropy     / n_updates,
        }