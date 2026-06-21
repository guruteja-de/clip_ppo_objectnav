from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:

    # ── Environment ───────────────────────────────────────────────
    # scenes: List[str] = field(default_factory=lambda: [
    #     "FloorPlan1", "FloorPlan2", "FloorPlan3",
    #     "FloorPlan4", "FloorPlan5", ])

    scenes: List[str] = field(default_factory=lambda: [f"FloorPlan{i}" for i in range(1, 31)])
   
    target_objects: List[str] = field(default_factory=lambda: [
        "refrigerator", "microwave", "sink", "toaster",
    ])
    image_size: int    = 224
    fov: int           = 90
    max_steps: int     = 500
    min_steps: int     = 15    # Stop blocked before this step
    visibility_distance: float = 1.5
    grid_size: float   = 0.25
    rotation_step: int = 90

    # ── CLIP ──────────────────────────────────────────────────────
    clip_model: str = "ViT-B/32"
    use_candidates:   bool = True    # True  = softmax over candidate list
                                     # False = open vocabulary (any text)
    goal_text:        str  = ""      # used only when use_candidates = False
                                     # e.g. "find the refrigerator"

    # ── Reward ────────────────────────────────────────────────────
    success_reward:     float = 5.0    # Big bonus when agent calls Stop AND target is truly visible.
    false_stop_penalty: float = -1.0   #Penalty when agent calls Stop but target is NOT visible. Discourages random stopping.
    step_penalty:       float = -0.01  #Small penalty every step. Encourages finding target quickly rather than wandering forever.
    clip_reward_scale:  float = 1.0    #Multiplier on the CLIP softmax probability reward. 1.0 means full scale (0–1 range).

    # ── PPO ───────────────────────────────────────────────────────
    lr:             float = 3e-4
    gamma:          float = 0.99
    gae_lambda:     float = 0.95
    clip_eps:       float = 0.2
    entropy_coef:   float = 0.05
    value_coef:     float = 0.5
    max_grad_norm:  float = 0.5
    n_steps:        int   = 2048
    batch_size:     int   = 128
    n_epochs:       int   = 4

    # ── Network ───────────────────────────────────────────────────
    hidden_dim:  int = 512
    hidden_dim2: int = 256

    # ── Training ──────────────────────────────────────────────────
    total_timesteps: int =  5_000_000 #3_000     # small for local test
    save_every:      int = 250_000 #50_000
    seed:            int = 42

    # ── Paths ─────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    log_file:       str = "training_log.csv"