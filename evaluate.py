"""
evaluate.py — Evaluate a trained ObjectNav policy.

Usage:
    python evaluate.py --checkpoint checkpoints/policy_final.pt --episodes 50
    python evaluate.py --checkpoint checkpoints/policy_final.pt --episodes 50 --render

Metrics:
    SR  = Success Rate  (% of episodes where agent finds target)
    SPL = Success weighted by Path Length  (Anderson et al. 2018)
          SPL = (1/N) Σ S_i * L_i / max(p_i, L_i)
          L_i = shortest path (approx. 1 step), p_i = actual steps taken
"""

import argparse
import numpy as np
import torch

from config import Config
from clip_extractor import CLIPExtractor
from objectnav_env import ObjectNavEnv
from networks import ActorCritic


def evaluate(checkpoint_path: str, n_episodes: int = 50, render: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint ───────────────────────────────────────────
    print(f"\nLoading: {checkpoint_path}")
    ckpt   = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", Config())

    # ── Build env + network ───────────────────────────────────────
    clip    = CLIPExtractor(config.clip_model, device=str(device))
    env     = ObjectNavEnv(config, clip)
    obs_dim = env.observation_space.shape[0]
    n_acts  = env.action_space.n

    network = ActorCritic(obs_dim, n_acts, config.hidden_dim, config.hidden_dim2)
    network.load_state_dict(ckpt["network"])
    network.to(device)
    network.eval()

    print(f"Evaluating {n_episodes} episodes ...\n")
    print(f"  {'Ep':>4}  {'Target':12}  {'Result':9}  {'Steps':>5}  {'Reward':>7}")
    print(f"  {'-'*50}")

    results = []

    for ep in range(n_episodes):
        obs, _    = env.reset()         # ← Gymnasium: returns (obs, info)
        done      = False
        ep_reward = 0.0
        ep_steps  = 0

        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                # Greedy evaluation: argmax (no stochastic sampling)
                logits, _ = network(obs_t)
                action    = logits.argmax(dim=-1).item()

            obs, reward, terminated, truncated, info = env.step(action)  # ← 5 values
            done       = terminated or truncated
            ep_reward += reward
            ep_steps  += 1

            if render:
                try:
                    import cv2
                    frame = env.render()
                    cv2.imshow("ObjectNav", frame[:, :, ::-1])
                    cv2.waitKey(30)
                except ImportError:
                    print("[Render] opencv-python not installed. Skipping.")
                    render = False

        success = info["success"]
        target  = info["target"]

        # Approx SPL: if success, credit = 1/steps; else 0
        # (True SPL needs geodesic distance — approximated here)
        spl = (1.0 / max(ep_steps, 1)) if success else 0.0

        results.append({
            "target":  target,
            "success": success,
            "steps":   ep_steps,
            "reward":  ep_reward,
            "spl":     spl,
        })

        status = "✓ SUCCESS" if success else "✗ FAIL   "
        print(
            f"  {ep+1:4d}  {target:12s}  {status}  "
            f"{ep_steps:5d}  {ep_reward:7.2f}"
        )

    # ── Summary ───────────────────────────────────────────────────
    sr        = np.mean([r["success"] for r in results])
    spl       = np.mean([r["spl"]     for r in results])
    avg_steps = np.mean([r["steps"]   for r in results])

    print(f"\n{'='*55}")
    print(f"  Episodes     : {n_episodes}")
    print(f"  Success Rate : {sr:.2%}")
    print(f"  Approx SPL   : {spl:.4f}")
    print(f"  Avg Steps    : {avg_steps:.1f}")

    print(f"\n  Per-target breakdown:")
    for obj in config.target_objects:
        obj_res = [r for r in results if r["target"] == obj]
        if obj_res:
            obj_sr = np.mean([r["success"] for r in obj_res])
            print(f"    {obj:15s}: {obj_sr:.2%}  ({len(obj_res)} eps)")
    print(f"{'='*55}\n")

    env.close()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ObjectNav policy")
    parser.add_argument("--checkpoint", default="checkpoints/policy_final.pt")
    parser.add_argument("--episodes",   type=int, default=50)
    parser.add_argument("--render",     action="store_true")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.episodes, args.render)