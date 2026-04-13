import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")


# PATHS


SARVAM_SUMMARY = "../results/analysis_summary_sarvam.json"
T5_SUMMARY = "../results/analysis_summary_T5.json"

COMPARE_DIR = "../results/model_comparison"

os.makedirs(COMPARE_DIR,exist_ok=True)


# LOAD SUMMARIES


with open(SARVAM_SUMMARY) as f:
    sarvam = json.load(f)

with open(T5_SUMMARY) as f:
    t5 = json.load(f)


# EMOTION DISTRIBUTION COMPARISON


emotion_df = pd.DataFrame({
    "Sarvam": sarvam["emotion_distribution_percent"],
    "T5": t5["emotion_distribution_percent"]
})

emotion_df.plot(kind="bar",figsize=(8,5))

plt.title("Emotion Distribution Comparison")
plt.ylabel("Percent")

plt.savefig(f"{COMPARE_DIR}/emotion_distribution_comparison.png")
plt.close()


# SCENARIO BIAS COMPARISON


bias_df = pd.DataFrame({
    "Model":["Sarvam","T5"],
    "Scenario Bias Rate":[
        sarvam["scenario_bias_rate_percent"],
        t5["scenario_bias_rate_percent"]
    ]
})

plt.figure(figsize=(6,4))

sns.barplot(data=bias_df,x="Model",y="Scenario Bias Rate")

plt.title("Scenario Bias Rate Comparison")

plt.savefig(f"{COMPARE_DIR}/scenario_bias_comparison.png")
plt.close()


# RELIGION TRIGGER BIAS COMPARISON


religion_df = pd.DataFrame({
    "Sarvam": sarvam["religion_trigger_bias_percent"],
    "T5": t5["religion_trigger_bias_percent"]
})

religion_df.plot(kind="bar",figsize=(8,5))

plt.title("Religion Trigger Bias Comparison")
plt.ylabel("Emotion Change %")

plt.savefig(f"{COMPARE_DIR}/religion_bias_comparison.png")
plt.close()


# DIRECT VS REASONING BIAS


mode_df = pd.DataFrame({
    "Sarvam": sarvam["mode_bias_rate_percent"],
    "T5": t5["mode_bias_rate_percent"]
})

mode_df.plot(kind="bar",figsize=(6,5))

plt.title("Direct vs Reasoning Bias Comparison")

plt.savefig(f"{COMPARE_DIR}/mode_bias_comparison.png")
plt.close()

print("Model comparison graphs generated.")