import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import ai2thor.controller

from config import Config
from clip_extractor import CLIPExtractor


class ObjectNavEnv(gym.Env):
    """
    AI2Thor ObjectNav environment as a Gymnasium wrapper.

    Observation:
        1024-dim float32 vector
        = [CLIP image embed (512) | CLIP text embed (512)]
        The text embed encodes the current navigation goal.

    Actions:
        0  MoveAhead   — move forward 0.25m
        1  RotateLeft  — rotate 90° left
        2  RotateRight — rotate 90° right
        3  LookUp      — tilt camera up 30°
        4  LookDown    — tilt camera down 30°
        5  Stop        — declare target found

    Reward:
        Dense:  CLIP softmax prob for target (0-1) every step
        Dense:  step_penalty (-0.01) every step
        Sparse: +5.0 when Stop called AND target truly visible
        Sparse: -1.0 when Stop called AND target NOT visible

    Episode ends:
        1. Agent calls Stop (success or failure)
        2. Max steps reached (timeout = failure)
    """

    ACTIONS = [
        "MoveAhead",
        "RotateLeft",
        "RotateRight",
        "LookUp",
        "LookDown",
        "Stop",
    ]

    def __init__(self, config: Config, clip: CLIPExtractor):
        super().__init__()
        self.config = config
        self.clip   = clip

        # ── Gymnasium spaces ──────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(1024,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.ACTIONS))

        # ── AI2Thor controller ────────────────────────────────────
        self.controller = ai2thor.controller.Controller(
            agentMode            = "default",
            visibilityDistance   = config.visibility_distance,
            scene                = config.scenes[0],
            gridSize             = config.grid_size,
            snapToGrid           = True,
            renderDepthImage     = False,
            renderInstanceSegmentation = False,
            width                = config.image_size,
            height               = config.image_size,
            fieldOfView          = config.fov,
            rotateStepDegrees    = config.rotation_step,
            headlessMode         = True,
        )

        # ── Episode state ─────────────────────────────────────────
        self.target             = None
        self._target_text_embed = None
        self._step_count        = 0
        self._episode_reward    = 0.0

    # ── Gymnasium interface ───────────────────────────────────────

    def reset(self, seed=None, options=None):
        """
        Start new episode.
        - Picks random scene from config.scenes
        - Picks random target from config.target_objects
        - Randomly spawns agent in scene
        Returns: (obs, info)
        """
        # Random scene and target each episode
        scene        = random.choice(self.config.scenes)
        self.target  = random.choice(self.config.target_objects)

        # Reset AI2Thor
        self.controller.reset(scene=scene)
        self.controller.step(
            action              = "InitialRandomSpawn",
            randomSeed          = random.randint(0, 10000),
            forceVisible        = False,
            numPlacementAttempts= 5,
        )

        # Cache target text embedding for this episode
        # encode_text() is cached so this is fast
        self._target_text_embed = self.clip.encode_text(self.target)

        # Reset episode counters
        self._step_count     = 0
        self._episode_reward = 0.0

        # Build first observation
        frame = self.controller.last_event.frame
        obs   = self._build_obs(frame)

        return obs, {"target": self.target}

    def step(self, action: int):
        """
        Execute one step.
        Returns: (obs, reward, terminated, truncated, info)

        terminated = episode ended naturally (Stop action)
        truncated  = episode ended due to timeout (max steps)
        """
        self._step_count += 1
        name = self.ACTIONS[action]

        # ── Handle Stop action ────────────────────────────────────
        if name == "Stop":
            if self._step_count < self.config.min_steps:
                # Too early — block Stop, force MoveAhead instead
                # Prevents agent from stopping immediately at start
                name = "MoveAhead"
            else:
                # Check if target is truly visible
                success = self._is_visible()
                reward  = (
                    self.config.success_reward
                    if success else
                    self.config.false_stop_penalty
                )
                self._episode_reward += reward
                obs = self._build_obs(self.controller.last_event.frame)
                return obs, reward, True, False, self._get_info(success)

        # ── Execute movement in AI2Thor ───────────────────────────
        event = self.controller.step(name)
        frame = event.frame

        # ── Single CLIP forward pass ──────────────────────────────
        # Returns BOTH image embedding (for obs) AND
        # target probability (for reward) in one pass
        if self.config.use_candidates:
            # Mode 1 — Closed set (softmax over candidates)
            img_embed, clip_prob = self.clip.encode_image_and_reward(
                frame,
                goal       = self.target,
                candidates = self.config.target_objects,
            )
        else:
            # Mode 2 — Open vocabulary (free-form text)
            img_embed, clip_prob = self.clip.encode_image_and_reward(
                frame,
                goal = self.config.goal_text or self.target,
            )

        # ── Compute reward ────────────────────────────────────────
        reward = self.config.step_penalty + self.config.clip_reward_scale * clip_prob
        self._episode_reward += reward

        # ── Build observation ─────────────────────────────────────
        obs = self._build_obs_from_embed(img_embed)

        # ── Check timeout ─────────────────────────────────────────
        truncated = self._step_count >= self.config.max_steps
        success   = False
        if truncated and self._is_visible():
            reward += self.config.success_reward
            success = True

        return obs, reward, False, truncated, self._get_info(success)

    def render(self):
        return self.controller.last_event.frame

    def close(self):
        self.controller.stop()

    # ── Internals ─────────────────────────────────────────────────

    def _build_obs(self, frame: np.ndarray) -> np.ndarray:
        """
        Full obs build — runs CLIP on frame.
        Used at reset() and after Stop action.
        """
        if self.config.use_candidates:
            img_embed, _ = self.clip.encode_image_and_reward(
                frame,
                goal       = self.target,
                candidates = self.config.target_objects,
            )
        else:
            img_embed, _ = self.clip.encode_image_and_reward(
                frame,
                goal = self.config.goal_text or self.target,
            )
        return self._build_obs_from_embed(img_embed)

    def _build_obs_from_embed(self, img_embed: np.ndarray) -> np.ndarray:
        """
        Concatenate image embed + text embed → 1024-dim obs.
        Avoids running CLIP twice per step.
        """
        return np.concatenate(
            [img_embed, self._target_text_embed]
        ).astype(np.float32)

    def _is_visible(self) -> bool:
        """
        Check AI2Thor ground truth visibility.
        Used for sparse reward and success metric.
        NOT using CLIP here — AI2Thor metadata is the oracle.
        """
        objects = self.controller.last_event.metadata.get("objects", [])
        for obj in objects:
            if (
                obj.get("objectType", "").lower() == self.target.lower()
                and obj.get("visible", False)
            ):
                return True
        return False

    def _get_info(self, success: bool) -> dict:
        return {
            "success":        success,
            "target":         self.target,
            "step_count":     self._step_count,
            "episode_reward": self._episode_reward,
        }

    def list_objects_in_scene(self) -> list:
        """Returns all object types in current scene."""
        objects = self.controller.last_event.metadata.get("objects", [])
        return sorted(set(o["objectType"] for o in objects))

    def visible_objects(self) -> list:
        """Returns currently visible object types."""
        objects = self.controller.last_event.metadata.get("objects", [])
        return [o["objectType"] for o in objects if o.get("visible", False)]