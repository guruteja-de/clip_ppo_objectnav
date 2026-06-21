import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """
    Shared-trunk Actor-Critic network for PPO.

    Input:  1024-dim = [CLIP image embed (512) | CLIP text embed (512)]
                                    ↓
    Trunk:  Linear(1024 → 512) → LayerNorm → ReLU
            Linear(512  → 256) → LayerNorm → ReLU
                                    ↓
                           ┌────────┴────────┐
    Actor:  Linear(256→128)→ReLU→Linear(128→6)    → action logits
    Critic: Linear(256→128)→ReLU→Linear(128→1)    → state value V(s)
    """

    def __init__(
        self,
        obs_dim:     int = 1024,
        n_actions:   int = 6,
        hidden_dim:  int = 512,
        hidden_dim2: int = 256,
    ):
        super().__init__()

        # ── Shared Trunk ──────────────────────────────────────────
        # Both actor and critic share this feature extractor.
        # LayerNorm keeps CLIP embeddings stable (they are
        # L2-normalized on a hypersphere).
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.ReLU(),
        )

        # ── Actor Head ────────────────────────────────────────────
        # Outputs raw logits (unnormalized scores) for each action.
        # Categorical distribution samples from these logits.
        # Actions:
        #   0 = MoveAhead
        #   1 = RotateLeft
        #   2 = RotateRight
        #   3 = LookUp
        #   4 = LookDown
        #   5 = Stop
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim2, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

        # ── Critic Head ───────────────────────────────────────────
        # Outputs a single scalar V(s) — expected future reward
        # from current state. Used for GAE advantage estimation.
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Orthogonal initialization — standard for PPO.

        Hidden layers → gain = sqrt(2)
            Keeps gradient magnitudes stable through deep layers.

        Actor output  → gain = 0.01
            Makes initial policy near-uniform.
            Agent starts by exploring all actions equally.

        Critic output → gain = 1.0
            Standard initialization for value prediction.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Override output layers
        nn.init.orthogonal_(self.actor[-1].weight,  gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)

    def forward(self, obs: torch.Tensor):
        """
        Args:
            obs: (batch, 1024) float tensor
        Returns:
            logits: (batch, n_actions)
            value:  (batch,)
        """
        features = self.trunk(obs)
        logits   = self.actor(features)
        value    = self.critic(features).squeeze(-1)
        return logits, value

    def get_action_and_value(
        self,
        obs:    torch.Tensor,
        action: torch.Tensor = None,
    ):
        """
        Two uses:

        During ROLLOUT (action=None):
            → samples action from current policy
            → used every env step in train.py

        During PPO UPDATE (action provided):
            → evaluates log_prob of that action under NEW policy
            → used in ppo_trainer.py update loop

        Returns:
            action:   (batch,) int64
            log_prob: (batch,) log probability of action
            entropy:  (batch,) entropy of action distribution
            value:    (batch,) state value V(s)
        """
        logits, value = self.forward(obs)
        dist          = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Returns V(s) only.
        Used at end of rollout for GAE bootstrapping.
        """
        features = self.trunk(obs)
        return self.critic(features).squeeze(-1)