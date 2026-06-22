import argparse, torch, numpy as np
from PIL import Image, ImageDraw
from config import Config
from clip_extractor import CLIPExtractor
from networks import ActorCritic
from objectnav_env import ObjectNavEnv

def record(checkpoint_path, target, max_attempts=50, output="episode.gif"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()
    config.target_objects = [target]
    clip = CLIPExtractor(device=device)
    policy = ActorCritic(obs_dim=1024, n_actions=6).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["network"])
    policy.eval()
    env = ObjectNavEnv(config, clip)
    action_names = ["Forward", "Back", "Rot Left", "Rot Right", "Look Up", "STOP"]
    for attempt in range(1, max_attempts + 1):
        obs, _ = env.reset()
        frames = []
        step = 0
        print(f"[Attempt {attempt}] Target: {env.target}")
        while True:
            raw_frame = env.controller.last_event.frame
            pil = Image.fromarray(raw_frame.astype(np.uint8)).resize((320, 320))
            draw = ImageDraw.Draw(pil)
            with torch.no_grad():
                state = torch.FloatTensor(obs).unsqueeze(0).to(device)
                logits, _ = policy(state)
                action = torch.argmax(logits, dim=-1).item()
            draw.rectangle([0, 0, 320, 28], fill=(0,0,0))
            draw.text((6, 6), f"Step {step+1} | {action_names[action]} | Target: {target}", fill=(255,255,255))
            frames.append(pil)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            done = terminated or truncated
            if done:
                success = info.get("success", False)
                print(f"  Steps: {step} | Success: {success}")
                if success:
                    win = frames[-1].copy()
                    d = ImageDraw.Draw(win)
                    d.rectangle([40, 120, 280, 200], fill=(0,150,0))
                    d.text((80, 140), "TARGET FOUND!", fill=(255,255,255))
                    d.text((110, 165), f"Steps: {step}", fill=(255,255,255))
                    for _ in range(12):
                        frames.append(win)
                    frames[0].save(output, save_all=True, append_images=frames[1:], duration=200, loop=0)
                    print(f"Saved: {output}")
                    env.close()
                    return
                break
    print("No success found.")
    env.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/policy_final.pt")
    p.add_argument("--target", default="microwave", choices=["refrigerator","microwave","sink","toaster"])
    p.add_argument("--output", default="episode.gif")
    p.add_argument("--attempts", type=int, default=50)
    args = p.parse_args()
    record(args.checkpoint, args.target, args.attempts, args.output)
