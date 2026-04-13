"""Emotion extraction utilities.

Extracted from experiments/experiment_T5.py and experiments/experiment_sarvam.py.
Supports keyword matching with optional embedding-based fallback.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.experiment_config import EMOTION_LIST


def extract_emotion(text, use_embedding_fallback=False, embed_model=None,
                    emotion_embeddings=None):
    """Extract emotion from model output text.

    Args:
        text: Raw model output string.
        use_embedding_fallback: If True, fall back to cosine similarity when
            keyword matching fails (used by Sarvam).
        embed_model: SentenceTransformer model for embedding fallback.
        emotion_embeddings: Pre-computed embeddings for EMOTION_LIST.

    Returns:
        Matched emotion string from EMOTION_LIST, or "Unknown".
    """
    text_lower = text.lower()

    for e in EMOTION_LIST:
        if e.lower() in text_lower:
            return e

    if use_embedding_fallback and embed_model is not None and emotion_embeddings is not None:
        from sklearn.metrics.pairwise import cosine_similarity
        text_embedding = embed_model.encode([text])
        similarities = cosine_similarity(text_embedding, emotion_embeddings)[0]
        best_idx = similarities.argmax()
        return EMOTION_LIST[best_idx]

    return "Unknown"
