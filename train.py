import os
import csv
import random
import time
import numpy as np
import torch

from config import Config
from clip_extractor import CLIPExtractor
from objectnav_env import ObjectNavEnv
from networks import ActorCritic
from ppo_trainer import PPOTrainer


def set_seed(seed: int):
    """Set seed for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train():
    # ── Setup ─────────────────────────────────────────────────────
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)

    print(f"\n{'='*55}")
    print(f"  Zero-Shot ObjectNav : CLIP + PPO")
    print(f"  Device              : {device}")
    print(f"  Scenes              : {config.scenes}")
    print(f"  Targets             : {config.target_objects}")
    print(f"  Mode                : {'Closed set' if config.use_candidates else 'Open vocabulary'}")
    print(f"  Total timesteps     : {config.total_timesteps:,}")
    print(f"{'='*55}\n")

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ── CLIP ──────────────────────────────────────────────────────
    clip = CLIPExtractor(config.clip_model, device=str(device))

    # ── Environment ───────────────────────────────────────────────
    env = ObjectNavEnv(config, clip)
    print(f"[Env] Objects in scene : {env.list_objects_in_scene()[:6]}")

    # ── Network + Trainer ─────────────────────────────────────────
    obs_dim   = env.observation_space.shape[0]   # 1024
    n_actions = env.action_space.n               # 6
    network   = ActorCritic(
        obs_dim     = obs_dim,
        n_actions   = n_actions,
        hidden_dim  = config.hidden_dim,
        hidden_dim2 = config.hidden_dim2,
    )
    trainer = PPOTrainer(network, config, device)
    print(f"[Network] Parameters   : {sum(p.numel() for p in network.parameters()):,}")

    # ── CSV Logger ────────────────────────────────────────────────
    log_file   = open(config.log_file, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "total_steps", "episode", "target", "success",
        "episode_reward", "episode_steps",
        "policy_loss", "value_loss", "entropy",
    ])

    # ── Training state ────────────────────────────────────────────
    obs, _       = env.reset()
    total_steps  = 0
    episode      = 0
    ep_reward    = 0.0
    ep_steps     = 0
    successes    = []        # recent episode outcomes for SR
    last_metrics = {}        # PPO metrics from last update
    t_start      = time.time()

    print("\nTraining started...\n")

    # ── Main Loop ─────────────────────────────────────────────────
    while total_steps < config.total_timesteps:

        # ── Step 1: Select action ──────────────────────────────
        action, log_prob, value = trainer.select_action(obs)

        # ── Step 2: Execute in environment ─────────────────────
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # ── Step 3: Store in buffer ────────────────────────────
        trainer.buffer.store(obs, action, log_prob, reward, value, done)

        # ── Step 4: Update state ───────────────────────────────
        ep_reward   += reward
        ep_steps    += 1
        total_steps += 1
        obs          = next_obs

        # ── Step 5: Episode finished ───────────────────────────
        if done:
            episode += 1
            successes.append(int(info["success"]))

            # Success rate over last 100 episodes
            sr  = np.mean(successes[-100:])
            sps = total_steps / (time.time() - t_start)

            print(
                f"Ep {episode:5d} | "
                f"Steps {total_steps:7d} | "
                f"{info['target']:12s} | "
                f"{'✓' if info['success'] else '✗'} | "
                f"Reward {ep_reward:7.2f} | "
                f"SR {sr:6.2%} | "
                f"SPS {sps:.1f}"
            )

            # Log to CSV
            log_writer.writerow([
                total_steps,
                episode,
                info["target"],
                int(info["success"]),
                f"{ep_reward:.4f}",
                ep_steps,
                f"{last_metrics.get('policy_loss', 0):.4f}",
                f"{last_metrics.get('value_loss',  0):.4f}",
                f"{last_metrics.get('entropy',     0):.4f}",
            ])
            log_file.flush()

            # Reset for next episode
            ep_reward = 0.0
            ep_steps  = 0
            obs, _    = env.reset()

        # ── Step 6: PPO update when buffer full ────────────────
        if trainer.buffer.is_full():
            last_metrics = trainer.update(obs, done)
            print(
                f"  [PPO] "
                f"policy={last_metrics['policy_loss']:+.4f}  "
                f"value={last_metrics['value_loss']:.4f}  "
                f"entropy={last_metrics['entropy']:.4f}"
            )

        # ── Step 7: Save checkpoint ────────────────────────────
        if total_steps % config.save_every == 0:
            ckpt_path = os.path.join(
                config.checkpoint_dir,
                f"policy_{total_steps:08d}.pt"
            )
            torch.save({
                "step":      total_steps,
                "network":   network.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "config":    config,
            }, ckpt_path)
            print(f"  [Saved] {ckpt_path}")

    # ── Training complete ──────────────────────────────────────────
    final_path = os.path.join(config.checkpoint_dir, "policy_final.pt")
    torch.save({
        "step":    total_steps,
        "network": network.state_dict(),
        "config":  config,
    }, final_path)

    print(f"\n{'='*55}")
    print(f"  Training complete!")
    print(f"  Total episodes : {episode}")
    print(f"  Final SR       : {np.mean(successes[-100:]):.2%}")
    print(f"  Model saved    : {final_path}")
    print(f"{'='*55}\n")

    log_file.close()
    env.close()


if __name__ == "__main__":
    train()