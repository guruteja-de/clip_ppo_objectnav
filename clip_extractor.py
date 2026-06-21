import numpy as np
import torch
import clip
from PIL import Image


OBJECT_PROMPTS = {
    "refrigerator": [
        "a photo of a refrigerator in a kitchen",
        "a large white refrigerator",
        "a kitchen refrigerator with a freezer",
    ],
    "microwave": [
        "a photo of a microwave oven on a kitchen counter",
        "a microwave mounted above a stove",
        "a black or silver microwave oven",
    ],
    "sink": [
        "a photo of a kitchen sink with a faucet",
        "a stainless steel kitchen sink",
        "a sink with running water in a kitchen",
    ],
    "toaster": [
        "a photo of a toaster on a kitchen counter",
        "a small silver or black toaster",
        "a bread toaster on a countertop",
    ],
}


class CLIPExtractor:
    """
    Wraps OpenAI CLIP for ObjectNav.

    Supports two modes:

    Mode 1 — Closed set (use_candidates=True):
        Softmax over predefined candidate list.
        → Sharp discrimination, stable reward.
        → Standard in ObjectNav literature.
        Usage: encode_image_and_reward(frame, "refrigerator",
                   candidates=["refrigerator","microwave","sink","toaster"])

    Mode 2 — Open vocabulary (use_candidates=False):
        Direct cosine similarity against any free-form text.
        → No predefined list needed.
        → Works with any goal text.
        Usage: encode_image_and_reward(frame, "find the fridge")
    """

    def __init__(self, model_name: str = "ViT-B/32", device: str = "cpu"):
        self.device = device
        print(f"[CLIP] Loading {model_name} on {device}...")
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self._text_cache      = {}  # cache per object name
        self._candidate_cache = {}  # cache per candidate tuple
        print("[CLIP] Ready ✓")

    def encode_image_and_reward(
        self,
        frame_np:   np.ndarray,
        goal:       str,
        candidates: list = None,
    ):
        """
        Single CLIP forward pass → (image_embed, reward_signal).

        Args:
            frame_np:   RGB numpy array (H, W, 3) from AI2Thor
            goal:       target object name OR free-form text
                        e.g. "refrigerator" or "find the fridge"
            candidates: list of candidate objects (Mode 1)
                        None = open vocabulary mode (Mode 2)

        Returns:
            image_embed:   np.ndarray (512,) — used in observation
            reward_signal: float (0-1)       — used as dense reward
        """
        image        = Image.fromarray(frame_np)
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            if candidates is not None:
                # ── Mode 1: Closed set ──────────────────────────────
                # Softmax over all candidates → sharp discrimination
                text_features = self._get_candidate_features(candidates)
                logits        = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                reward_signal = logits[0][candidates.index(goal)].item()

            else:
                # ── Mode 2: Open vocabulary ─────────────────────────
                # Direct cosine similarity → any free-form text works
                text_features = self._encode_free_text(goal)
                similarity    = (image_features @ text_features.T).item()

                # Scale raw cosine (~0.20-0.35) to cleaner 0-1 range
                reward_signal = self._scale_similarity(similarity)

        image_embed = image_features.squeeze(0).cpu().numpy()
        return image_embed, reward_signal

    def encode_text(self, object_name: str) -> np.ndarray:
        """
        Returns cached normalized CLIP text embedding (512,).
        Works for both object names and free-form text.
        Computed only once per unique string.
        """
        if object_name not in self._text_cache:
            # Use descriptive prompt if available, else use text directly
            prompts = OBJECT_PROMPTS.get(object_name.lower(), None)
            text    = prompts[0] if prompts else object_name
            tokens  = clip.tokenize([text]).to(self.device)
            with torch.no_grad():
                features = self.model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
            self._text_cache[object_name] = features.squeeze(0).cpu().numpy()
        return self._text_cache[object_name]

    # ── Internals ────────────────────────────────────────────────────

    def _get_candidate_features(self, candidates: list) -> torch.Tensor:
        """
        Mode 1 — Returns stacked text features for all candidates.
        Uses prompt ensembling — averages multiple prompts per object.
        Cached by candidate tuple.
        """
        key = tuple(candidates)
        if key not in self._candidate_cache:
            all_features = []
            for obj in candidates:
                prompts = OBJECT_PROMPTS.get(obj.lower(),
                                              [f"a photo of a {obj}"])
                tokens = clip.tokenize(prompts).to(self.device)
                with torch.no_grad():
                    feats     = self.model.encode_text(tokens)
                    feats     = feats / feats.norm(dim=-1, keepdim=True)
                    mean_feat = feats.mean(dim=0, keepdim=True)
                    mean_feat = mean_feat / mean_feat.norm(dim=-1, keepdim=True)
                all_features.append(mean_feat)
            self._candidate_cache[key] = torch.cat(all_features, dim=0)
        return self._candidate_cache[key]

    def _encode_free_text(self, text: str) -> torch.Tensor:
        """
        Mode 2 — Encodes any free-form text directly.
        No predefined prompts — uses text as-is.
        Cached by text string.
        """
        cache_key = f"__free__{text}"
        if cache_key not in self._text_cache:
            tokens = clip.tokenize([text]).to(self.device)
            with torch.no_grad():
                features = self.model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
            self._text_cache[cache_key] = features
        return self._text_cache[cache_key]

    def _scale_similarity(self, similarity: float) -> float:
        """
        Mode 2 — Scales raw cosine similarity to 0-1 range.

        Raw CLIP cosine similarity between ANY image and text
        typically falls in [0.20, 0.35]. We scale this range
        to [0, 1] so it works as a proper reward signal.

        0.20 (unrelated) → 0.0
        0.35 (matching)  → 1.0
        """
        low, high = 0.20, 0.35
        scaled = (similarity - low) / (high - low)
        return float(max(0.0, min(1.0, scaled)))