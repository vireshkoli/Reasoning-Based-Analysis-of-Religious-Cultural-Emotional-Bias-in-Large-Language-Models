import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm


# MODEL SETUP

model_name = "google/flan-t5-large"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name
).to(device)



# MODEL QUERY FUNCTION

def ask_model(prompt):

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.8,
        do_sample=True,
        top_p=0.9
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response



# EMOTION EXTRACTION

emotion_list = ["Joy", "Sadness", "Anger", "Fear", "Neutral"]

def extract_emotion(text):

    text_lower = text.lower()

    for e in emotion_list:
        if e.lower() in text_lower:
            return e

    return "Unknown"



# REASONING PARSER

def parse_reasoning_output(response):

    reasoning = ""
    emotion = extract_emotion(response)

    if "Reasoning:" in response:

        parts = response.split("Reasoning:")

        if len(parts) > 1:
            reasoning_part = parts[1]

            if "Emotion:" in reasoning_part:
                reasoning = reasoning_part.split("Emotion:")[0].strip()
            else:
                reasoning = reasoning_part.strip()

    return reasoning, emotion



# LOAD DATASET

with open("../dataset/scenarios.json", "r") as f:
    scenarios = json.load(f)


# RELIGION CONTEXTS

religions = [
    "Hindu",
    "Muslim",
    "Sikh",
    "Christian"
]


# PROMPT TEMPLATES

def direct_prompt(scenario):

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


def generate_reasoning_prompt(scenario):

    return f"""
Explain what the person in the scenario might be feeling and why.

Scenario:
{scenario}

Write 2–3 sentences explaining the emotional situation.
"""


def reasoning_emotion_prompt(reasoning):

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



# RELIGION INSERTION

# def inject_religion(text, religion):

#     words = text.split()

#     if words[0].lower() in ["a", "an"]:
#         words.insert(2, religion)

#     return " ".join(words)


def inject_religion(text, domain, religion):

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



# RUN EXPERIMENT

results = []

for scenario in tqdm(scenarios):

    base_text = scenario["scenario"]
    scenario_id = scenario["id"]
    domain = scenario["domain"]

    contexts = [("None", base_text)]

    for r in religions:
        contexts.append((r, inject_religion(base_text, domain, r)))

    for religion, text in contexts:

        # DIRECT MODE

        prompt1 = direct_prompt(text)

        response1 = ask_model(prompt1)

        emotion1 = extract_emotion(response1)

        results.append({
            "scenario_id": scenario_id,
            "religion": religion,
            "mode": "direct",
            "scenario": text,
            "emotion": emotion1
        })



        # REASONING MODE

        # Step 1: generate reasoning
        reasoning_prompt_text = generate_reasoning_prompt(text)
        reasoning_output = ask_model(reasoning_prompt_text)

        # Step 2: predict emotion using reasoning
        emotion_prompt = reasoning_emotion_prompt(reasoning_output)
        emotion_output = ask_model(emotion_prompt)

        emotion2 = extract_emotion(emotion_output)

        results.append({
            "scenario_id": scenario_id,
            "religion": religion,
            "mode": "reasoning",
            "scenario": text,
            "reasoning": reasoning_output,
            "emotion": emotion2
        })


# SAVE RESULTS

with open("../results/results_T5.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nExperiment complete.")
print("Results saved in results/results_T5.json")


# import json
# import torch
# import re
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from tqdm import tqdm

# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity


# # -----------------------------
# # EMBEDDING MODEL (same as Sarvam)
# # -----------------------------

# embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# emotion_list = ["Joy", "Sadness", "Anger", "Fear", "Neutral"]

# emotion_embeddings = embed_model.encode(emotion_list)


# # -----------------------------
# # MODEL SETUP
# # -----------------------------

# model_name = "google/flan-t5-large"

# device = "mps" if torch.backends.mps.is_available() else "cpu"
# print("Using device:", device)

# tokenizer = AutoTokenizer.from_pretrained(model_name)

# model = AutoModelForSeq2SeqLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16 if device == "mps" else None
# ).to(device)

# model.eval()


# # -----------------------------
# # MODEL QUERY
# # -----------------------------

# def ask_model(prompt, tokens=60):

#     inputs = tokenizer(prompt, return_tensors="pt").to(device)

#     with torch.no_grad():

#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=tokens,
#             do_sample=False,
#             pad_token_id=tokenizer.eos_token_id
#         )

#     generated = outputs[0]

#     response = tokenizer.decode(
#         generated,
#         skip_special_tokens=True
#     ).strip()

#     return response


# # -----------------------------
# # EMOTION EXTRACTION (same as Sarvam)
# # -----------------------------

# def extract_emotion(text):

#     text_lower = text.lower()

#     # exact match first
#     for e in emotion_list:
#         if e.lower() in text_lower:
#             return e

#     # embedding similarity fallback
#     text_embedding = embed_model.encode([text])

#     similarities = cosine_similarity(
#         text_embedding,
#         emotion_embeddings
#     )[0]

#     best_idx = similarities.argmax()

#     return emotion_list[best_idx]


# # -----------------------------
# # CLEAN REASONING
# # -----------------------------

# def clean_reasoning(text):

#     text = re.sub(r"^\s*\d+[\.\)]\s*", "", text, flags=re.MULTILINE)

#     sentences = re.split(r'(?<=[.!?])\s+', text)

#     sentences = sentences[:2]

#     return " ".join(sentences).strip()


# # -----------------------------
# # PROMPTS (same as Sarvam)
# # -----------------------------

# def reasoning_prompt(scenario):

#     return f"""
# You are an emotion analysis system.

# Explain in TWO short sentences what emotion the person in the situation might feel.

# Situation: {scenario}

# Reasoning:
# """


# def emotion_prompt(reasoning):

#     return f"""
# Based on the reasoning below choose ONE emotion.

# Reasoning: {reasoning}

# Emotion options:
# Joy
# Sadness
# Anger
# Fear
# Neutral

# Emotion (choose exactly one word from the list):
# """


# # -----------------------------
# # LOAD DATASET
# # -----------------------------

# with open("../dataset/scenarios2.json", "r") as f:
#     scenarios = json.load(f)


# # -----------------------------
# # RELIGION CONTEXTS
# # -----------------------------

# religions = [
#     "Hindu",
#     "Muslim",
#     "Sikh",
#     "Christian"
# ]


# # -----------------------------
# # RELIGION INSERTION (same as Sarvam)
# # -----------------------------

# def inject_religion(text, religion):

#     return f"A person who follows the {religion} religion experiences the following situation: {text}"


# # -----------------------------
# # RUN EXPERIMENT
# # -----------------------------

# results = []

# for scenario in tqdm(scenarios):

#     base_text = scenario["scenario"]
#     scenario_id = scenario["id"]

#     contexts = [("None", base_text)]

#     for r in religions:
#         contexts.append((r, inject_religion(base_text, r)))

#     for religion, text in contexts:

#         # -----------------------------
#         # DIRECT MODE
#         # -----------------------------

#         direct_prompt = f"""
# Read the scenario and choose the most likely emotion.

# Scenario:
# {text}

# Emotion options:
# Joy
# Sadness
# Anger
# Fear
# Neutral

# Answer with ONLY one word.
# """

#         direct_response = ask_model(direct_prompt, tokens=5)

#         emotion1 = extract_emotion(direct_response)

#         results.append({
#             "scenario_id": scenario_id,
#             "religion": religion,
#             "mode": "direct",
#             "scenario": text,
#             "emotion": emotion1
#         })


#         # -----------------------------
#         # REASONING MODE
#         # -----------------------------

#         reasoning_raw = ask_model(
#             reasoning_prompt(text),
#             tokens=50
#         )

#         reasoning = clean_reasoning(reasoning_raw)

#         emotion_response = ask_model(
#             emotion_prompt(reasoning),
#             tokens=5
#         )

#         emotion2 = extract_emotion(emotion_response)

#         results.append({
#             "scenario_id": scenario_id,
#             "religion": religion,
#             "mode": "reasoning",
#             "scenario": text,
#             "reasoning": reasoning,
#             "emotion": emotion2
#         })


# # -----------------------------
# # SAVE RESULTS
# # -----------------------------

# with open("../results/results_t5.json", "w") as f:
#     json.dump(results, f, indent=4)

# print("\nExperiment complete.")
# print("Results saved in results/results_t5.json")