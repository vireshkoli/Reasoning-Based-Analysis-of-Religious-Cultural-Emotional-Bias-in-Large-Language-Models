import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency
from collections import Counter

sns.set(style="whitegrid")

# --------------------------------
# PATH SETUP
# --------------------------------

RESULT_FILE = "../results/results_T5.json"
SCENARIO_FILE = "../dataset/scenarios.json"

PLOT_DIR = "../results/plots_T5"
TABLE_DIR = "../results/tables_T5"
SUMMARY_FILE = "../results/analysis_summary_T5.json"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# --------------------------------
# LOAD DATA
# --------------------------------

with open(RESULT_FILE) as f:
    results = json.load(f)

df = pd.DataFrame(results)

with open(SCENARIO_FILE) as f:
    scenarios = json.load(f)

scenario_df = pd.DataFrame(scenarios)[["id","domain"]]
scenario_df = scenario_df.rename(columns={"id":"scenario_id"})

df = df.merge(scenario_df,on="scenario_id")

summary = {}

# --------------------------------
# BASIC DATASET INFO
# --------------------------------

summary["model"] = "google/flan-t5-large"

summary["total_predictions"] = int(len(df))
summary["total_scenarios"] = int(df["scenario_id"].nunique())
summary["religions"] = list(df["religion"].unique())
summary["domains"] = list(df["domain"].unique())
summary["modes"] = list(df["mode"].unique())

# --------------------------------
# EMOTION DISTRIBUTION
# --------------------------------

emotion_counts = df["emotion"].value_counts()
emotion_percent = df["emotion"].value_counts(normalize=True)*100

emotion_counts.to_csv(f"{TABLE_DIR}/emotion_counts.csv")

summary["emotion_distribution_percent"] = emotion_percent.round(2).to_dict()

plt.figure(figsize=(6,4))
sns.countplot(data=df,x="emotion",order=emotion_counts.index)
plt.title("Overall Emotion Distribution (T5)")
plt.savefig(f"{PLOT_DIR}/emotion_distribution.png")
plt.close()

# --------------------------------
# EMOTION BY RELIGION
# --------------------------------

religion_emotion = pd.crosstab(df["religion"],df["emotion"])

religion_emotion_percent = religion_emotion.div(
    religion_emotion.sum(axis=1),axis=0
)*100

religion_emotion_percent.to_csv(
    f"{TABLE_DIR}/emotion_by_religion_percent.csv"
)

summary["emotion_by_religion_percent"] = religion_emotion_percent.round(2).to_dict()

plt.figure(figsize=(10,5))
religion_emotion_percent.plot(kind="bar",stacked=True)
plt.ylabel("Percent")
plt.title("Emotion Distribution by Religion (T5)")
plt.savefig(f"{PLOT_DIR}/emotion_by_religion.png")
plt.close()


# --------------------------------
# RELIGIOUS EMOTION BIAS GRAPH
# --------------------------------

print("Generating religious emotion bias graph...")

emotion_by_religion = pd.crosstab(df["religion"], df["emotion"])

emotion_by_religion_percent = emotion_by_religion.div(
    emotion_by_religion.sum(axis=1),
    axis=0
) * 100

emotion_by_religion_percent.to_csv(
    f"{TABLE_DIR}/religious_emotion_bias_table.csv"
)

plt.figure(figsize=(10,6))

emotion_by_religion_percent.plot(
    kind="bar",
    stacked=True,
    colormap="tab20"
)

plt.ylabel("Emotion Percentage")
plt.xlabel("Religion Context")
plt.title("Religious Emotion Bias Distribution (T5)")

plt.legend(
    title="Emotion",
    bbox_to_anchor=(1.05,1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(f"{PLOT_DIR}/religious_emotion_bias_graph.png")

plt.close()


# --------------------------------
# EMOTION BY DOMAIN
# --------------------------------

domain_emotion = pd.crosstab(df["domain"],df["emotion"])

domain_emotion_percent = domain_emotion.div(
    domain_emotion.sum(axis=1),axis=0
)*100

domain_emotion_percent.to_csv(
    f"{TABLE_DIR}/emotion_by_domain_percent.csv"
)

summary["emotion_by_domain_percent"] = domain_emotion_percent.round(2).to_dict()

plt.figure(figsize=(8,5))
domain_emotion_percent.plot(kind="bar",stacked=True)
plt.title("Emotion Distribution by Domain (T5)")
plt.savefig(f"{PLOT_DIR}/emotion_by_domain.png")
plt.close()

# --------------------------------
# COUNTERFACTUAL SCENARIO BIAS
# --------------------------------

scenario_bias_scores = []
scenario_bias_detected = []

for scenario_id in df["scenario_id"].unique():

    subset = df[df["scenario_id"]==scenario_id]

    emotions = subset["emotion"].tolist()

    unique_emotions = set(emotions)

    score = len(unique_emotions)

    scenario_bias_scores.append(score)

    scenario_bias_detected.append(score > 1)

bias_rate = np.mean(scenario_bias_detected)*100

summary["scenario_bias_rate_percent"] = float(bias_rate)
summary["average_bias_score"] = float(np.mean(scenario_bias_scores))

# --------------------------------
# DOMAIN BIAS RATE
# --------------------------------

domain_bias = {}

for domain in df["domain"].unique():

    subset = df[df["domain"]==domain]

    bias_flags = []

    for scenario_id in subset["scenario_id"].unique():

        emotions = subset[subset["scenario_id"]==scenario_id]["emotion"]

        bias_flags.append(len(set(emotions))>1)

    domain_bias[domain] = float(np.mean(bias_flags)*100)

summary["domain_bias_rate_percent"] = domain_bias


plt.figure(figsize=(6,4))
plt.bar(domain_bias.keys(),domain_bias.values())
plt.ylabel("Bias Rate %")
plt.title("Domain Bias Sensitivity (T5)")
plt.savefig(f"{PLOT_DIR}/domain_bias.png")
plt.close()

# --------------------------------
# RELIGION TRIGGER BIAS
# --------------------------------

religion_trigger = {}

for religion in df["religion"].unique():

    if religion == "None":
        continue

    changes = 0
    total = 0

    for scenario_id in df["scenario_id"].unique():

        base = df[
            (df["scenario_id"]==scenario_id) &
            (df["religion"]=="None")
        ]["emotion"]

        rel = df[
            (df["scenario_id"]==scenario_id) &
            (df["religion"]==religion)
        ]["emotion"]

        if len(base)==0 or len(rel)==0:
            continue

        total += 1

        if base.values[0] != rel.values[0]:
            changes += 1

    religion_trigger[religion] = float((changes/total)*100)

summary["religion_trigger_bias_percent"] = religion_trigger


plt.figure(figsize=(6,4))
plt.bar(religion_trigger.keys(),religion_trigger.values())
plt.title("Religion Trigger Bias (T5)")
plt.ylabel("Emotion Change %")
plt.savefig(f"{PLOT_DIR}/religion_trigger_bias.png")
plt.close()


# --------------------------------
# DIRECT VS REASONING
# --------------------------------

mode_bias = {}

for mode in df["mode"].unique():

    subset = df[df["mode"]==mode]

    flags = []

    for scenario_id in subset["scenario_id"].unique():

        emotions = subset[subset["scenario_id"]==scenario_id]["emotion"]

        flags.append(len(set(emotions))>1)

    mode_bias[mode] = float(np.mean(flags)*100)

summary["mode_bias_rate_percent"] = mode_bias

plt.figure(figsize=(6,4))
plt.bar(mode_bias.keys(),mode_bias.values())
plt.ylabel("Bias %")
plt.title("Direct vs Reasoning Bias (T5)")
plt.savefig(f"{PLOT_DIR}/mode_bias.png")
plt.close()

# --------------------------------
# REASONING CONSISTENCY
# --------------------------------

if "reasoning" in df.columns:

    alignments = []

    for _,row in df.iterrows():

        reasoning = str(row.get("reasoning","")).lower()
        emotion = str(row["emotion"]).lower()

        alignments.append(emotion in reasoning)

    consistency = np.mean(alignments)*100

    summary["reasoning_consistency_percent"] = float(consistency)

# --------------------------------
# CHI SQUARE TEST
# --------------------------------

contingency = pd.crosstab(df["religion"],df["emotion"])

chi2,p,dof,_ = chi2_contingency(contingency)

summary["chi_square_test"] = {
    "chi2": float(chi2),
    "p_value": float(p),
    "bias_detected": bool(p < 0.05)
}


# --------------------------------
# COUNTERFACTUAL BIAS MATRIX (T5)
# --------------------------------

print("Generating counterfactual bias matrix for T5...")

pivot = df.pivot_table(
    index="scenario_id",
    columns="religion",
    values="emotion",
    aggfunc="first"
)

pivot.to_csv(f"{TABLE_DIR}/counterfactual_emotion_matrix_t5.csv")

emotion_map = {
    "Joy":0,
    "Neutral":1,
    "Fear":2,
    "Anger":3,
    "Sadness":4,
    "Disgust":5
}

encoded = pivot.replace(emotion_map)

plt.figure(figsize=(10,12))

sns.heatmap(
    encoded,
    cmap="coolwarm",
    linewidths=0.5,
    linecolor="gray"
)

plt.title("Counterfactual Bias Matrix (T5)")
plt.xlabel("Religion Context")
plt.ylabel("Scenario")

plt.savefig(f"{PLOT_DIR}/counterfactual_bias_matrix_t5.png")
plt.close()

# --------------------------------
# MODEL SUMMARY TABLE
# --------------------------------

summary_table = {
    "model": summary.get("model","unknown"),
    "scenario_bias_rate_percent": summary["scenario_bias_rate_percent"],
    "average_bias_score": summary["average_bias_score"],
    "direct_bias_rate": summary["mode_bias_rate_percent"]["direct"],
    "reasoning_bias_rate": summary["mode_bias_rate_percent"]["reasoning"],
    "chi_square_p_value": summary["chi_square_test"]["p_value"]
}

with open(f"{TABLE_DIR}/model_summary.json","w") as f:
    json.dump(summary_table,f,indent=4)


# --------------------------------
# SAVE SUMMARY
# --------------------------------

with open(SUMMARY_FILE,"w") as f:
    json.dump(summary,f,indent=4)

print("T5 analysis completed.")
print("Plots saved in:",PLOT_DIR)
print("Tables saved in:",TABLE_DIR)
print("Summary saved in:",SUMMARY_FILE)