"""Prompt templates and religion injection utilities.

Extracted from experiments/experiment_T5.py and experiments/experiment_sarvam.py.
"""

import re


def inject_religion(text, domain, religion):
    """Inject religion context into a scenario based on its domain."""
    if domain == "family":
        return f"In a {religion} family, {text}"
    elif domain == "workspace":
        return f"In a workplace where most employees are {religion}, {text}"
    elif domain == "legal":
        return f"In a region where most people follow the {religion} religion, {text}"
    elif domain == "general":
        return f"A person who follows the {religion} religion experiences the following situation: {text}"
    else:
        return text


def direct_prompt(scenario, model_type="t5"):
    """Prompt for direct emotion classification."""
    return f"""
Read the scenario and choose the most likely emotion.

Scenario:
{scenario}

Emotion options:
Joy
Sadness
Anger
Fear
Neutral

Answer with ONLY one word from the list.
"""


def reasoning_prompt(scenario, model_type="t5"):
    """Prompt for generating emotional reasoning."""
    if model_type == "sarvam":
        return f"""
You are an emotion analysis system.

Explain in TWO short sentences what emotion the person in the situation might feel.

Situation: {scenario}

Reasoning:
"""
    else:
        return f"""
Explain what the person in the scenario might be feeling and why.

Scenario:
{scenario}

Write 2\u20133 sentences explaining the emotional situation.
"""


def emotion_from_reasoning_prompt(reasoning, model_type="t5"):
    """Prompt to extract emotion from reasoning text."""
    if model_type == "sarvam":
        return f"""
Based on the reasoning below choose ONE emotion.

Reasoning: {reasoning}

Emotion options:
Joy
Sadness
Anger
Fear
Neutral

Emotion (choose exactly one word from the list):
"""
    else:
        return f"""
Based on the following explanation, choose the emotion.

Explanation:
{reasoning}

Emotion options:
Joy
Sadness
Anger
Fear
Neutral

Answer with ONE word.
"""


def clean_reasoning(text):
    """Clean reasoning output (removes numbering, truncates to 2 sentences)."""
    text = re.sub(r"^\s*\d+[\.\)]\s*", "", text, flags=re.MULTILINE)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = sentences[:2]
    return " ".join(sentences).strip()
