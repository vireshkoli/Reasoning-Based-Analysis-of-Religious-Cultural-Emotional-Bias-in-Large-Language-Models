"""
Simple, professor-friendly summary plots and numerical results.

Answers three questions:
  Q1. Why does religion trigger lead to emotion change?
  Q2. Why does direct vs reasoning mode differ so much in bias?
  Q3. Which layer/head is responsible?
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABLATION_DIR = os.path.join(ROOT, "results", "phase3", "ablation")
PROBING_DIR  = os.path.join(ROOT, "results", "phase3", "probing")
OUT_DIR      = os.path.join(ROOT, "results", "phase3", "simple_plots")
SUMMARY_T5   = os.path.join(ROOT, "results", "analysis_summary_T5.json")
SUMMARY_SV   = os.path.join(ROOT, "results", "analysis_summary_sarvam.json")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
with open(os.path.join(ABLATION_DIR, "layer_ablation_t5.json"))    as f: la_t5 = json.load(f)
with open(os.path.join(ABLATION_DIR, "layer_ablation_sarvam.json")) as f: la_sv = json.load(f)
with open(os.path.join(ABLATION_DIR, "head_ablation_t5.json"))     as f: ha_t5 = json.load(f)
with open(os.path.join(ABLATION_DIR, "head_ablation_sarvam.json")) as f: ha_sv = json.load(f)
with open(os.path.join(PROBING_DIR,  "probing_results_t5.json"))    as f: pr_t5 = json.load(f)
with open(os.path.join(PROBING_DIR,  "probing_results_sarvam.json")) as f: pr_sv = json.load(f)
with open(SUMMARY_T5) as f: s_t5 = json.load(f)
with open(SUMMARY_SV) as f: s_sv = json.load(f)

CONTROL_T5 = s_t5["scenario_bias_rate_percent"]   # 85%
CONTROL_SV = s_sv["scenario_bias_rate_percent"]   # 90%

# ── Helper ───────────────────────────────────────────────────────────────────
def lcs(ablation_dict, control):
    """Return {layer_key: LCS} sorted by LCS descending."""
    out = {}
    for k, v in ablation_dict.items():
        out[k] = control - v["metrics"]["scenario_bias_rate"]
    return dict(sorted(out.items(), key=lambda x: x[1], reverse=True))

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Q1: Religion trigger → emotion change (side-by-side bar chart)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
fig.suptitle("Q1: How much does Religion Context Change the Emotion?",
             fontsize=14, fontweight="bold", y=1.01)

for ax, summary, model in zip(axes, [s_t5, s_sv], ["FLAN-T5-Large", "Sarvam-2B"]):
    trigger = summary["religion_trigger_bias_percent"]
    religions = list(trigger.keys())
    values    = [trigger[r] for r in religions]
    colors    = ["#e74c3c" if v > 20 else "#e67e22" if v > 10 else "#2ecc71"
                 for v in values]
    bars = ax.bar(religions, values, color=colors, edgecolor="white", linewidth=1.2, width=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{model}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Religion Context Added", fontsize=11)
    ax.set_ylabel("Scenarios where emotion changed (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.text(0.5, -0.04,
         "Baseline = 'None' religion context  |  Bar height = % scenarios where emotion flipped vs baseline",
         ha="center", fontsize=9, color="gray")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig1_religion_trigger.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig1_religion_trigger.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Q2: Direct vs Reasoning mode bias gap
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("Q2: Why is Reasoning Mode More Biased than Direct Mode?",
             fontsize=14, fontweight="bold", y=1.01)

for ax, summary, model in zip(axes, [s_t5, s_sv], ["FLAN-T5-Large", "Sarvam-2B"]):
    direct    = summary["mode_bias_rate_percent"]["direct"]
    reasoning = summary["mode_bias_rate_percent"]["reasoning"]
    gap       = reasoning - direct

    bars = ax.bar(["Direct\n(classify only)", "Reasoning\n(explain → classify)"],
                  [direct, reasoning],
                  color=["#3498db", "#e74c3c"],
                  edgecolor="white", linewidth=1.2, width=0.45)
    ax.set_title(f"{model}", fontsize=12, fontweight="bold")
    ax.set_ylabel("Bias Rate — % scenarios with\nemotion change across religions (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))
    for bar, val in zip(bars, [direct, reasoning]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
    # Annotate gap
    ax.annotate("", xy=(1, reasoning), xytext=(1, direct),
                arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=2))
    ax.text(1.28, (direct + reasoning) / 2, f"+{gap:.0f}%\ngap",
            ha="left", va="center", fontsize=11, color="#c0392b", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.text(0.5, -0.04,
         "Reasoning mode forces the model to generate a 'chain-of-thought' which amplifies religion-sensitive patterns",
         ha="center", fontsize=9, color="gray")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig2_direct_vs_reasoning.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig2_direct_vs_reasoning.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Q3a: Which LAYER is responsible? (LCS bar chart)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle("Q3a: Which Layer is Responsible for Bias? (Layer Contribution Score)",
             fontsize=14, fontweight="bold")
fig.text(0.5, 0.96,
         "Positive LCS = removing this layer REDUCES bias  |  Negative LCS = removing it INCREASES bias",
         ha="center", fontsize=10, color="gray")

for ax, ablation, control, model in zip(
        axes,
        [la_t5, la_sv],
        [CONTROL_T5, CONTROL_SV],
        ["FLAN-T5-Large (Encoder + Decoder layers)", "Sarvam-2B (Decoder-only layers)"]):

    lcs_dict = lcs(ablation, control)
    keys   = list(lcs_dict.keys())
    values = list(lcs_dict.values())

    # Shorten labels: "encoder_layer_6" → "Enc-6"
    def short(k):
        parts = k.split("_")
        prefix = "Enc" if parts[0] == "encoder" else "Dec"
        return f"{prefix}-{parts[-1]}"

    labels = [short(k) for k in keys]
    colors = ["#c0392b" if v >= 10 else "#e74c3c" if v > 0 else "#3498db" if v > -10 else "#2471a3"
              for v in values]

    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(f"{model}", fontsize=12, fontweight="bold")
    ax.set_ylabel("LCS = Control bias − Ablated bias (%)", fontsize=10)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Label top contributors
    for i, (bar, val) in enumerate(zip(bars, values)):
        if abs(val) >= 10:
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + (1.5 if val >= 0 else -3),
                    f"{val:+.0f}%", ha="center", fontsize=8,
                    fontweight="bold", color="black")

    # Legend
    red_patch  = mpatches.Patch(color="#c0392b", label="Strongly reduces bias (LCS ≥ +10%)")
    blue_patch = mpatches.Patch(color="#2471a3", label="Increases bias when removed (LCS ≤ -10%)")
    ax.legend(handles=[red_patch, blue_patch], fontsize=9, loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig3_layer_contribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig3_layer_contribution.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Q3b: Which HEAD is responsible? (top heads heatmap)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("Q3b: Which Attention Head Drives the Bias? (Head Contribution Score)",
             fontsize=14, fontweight="bold")

for ax, ha, control, model in zip(
        axes,
        [ha_t5, ha_sv],
        [CONTROL_T5, CONTROL_SV],
        ["FLAN-T5-Large", "Sarvam-2B"]):

    # Extract layer/head indices and LCS
    layer_ids = sorted(set(v["layer_idx"] for v in ha.values()))
    head_ids  = sorted(set(v["head_idx"]  for v in ha.values()))

    matrix = np.full((len(layer_ids), len(head_ids)), np.nan)
    for v in ha.values():
        li = layer_ids.index(v["layer_idx"])
        hi = head_ids.index(v["head_idx"])
        matrix[li, hi] = control - v["metrics"]["scenario_bias_rate"]

    # Layer labels: use layer_type + idx from first matching key
    layer_labels = []
    for lid in layer_ids:
        for key, val in ha.items():
            if val["layer_idx"] == lid:
                prefix = "Enc" if val["layer_type"] == "encoder" else "Dec"
                layer_labels.append(f"{prefix}-{lid}")
                break

    im = ax.imshow(matrix, cmap="RdBu", vmin=-20, vmax=20, aspect="auto")
    ax.set_xticks(range(len(head_ids)))
    ax.set_xticklabels([f"H{h}" for h in head_ids], fontsize=9)
    ax.set_yticks(range(len(layer_ids)))
    ax.set_yticklabels(layer_labels, fontsize=10)
    ax.set_xlabel("Attention Head", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title(f"{model}", fontsize=12, fontweight="bold")

    # Annotate cells
    for i in range(len(layer_ids)):
        for j in range(len(head_ids)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.0f}", ha="center", va="center",
                        fontsize=8, color="black" if abs(val) < 12 else "white",
                        fontweight="bold" if abs(val) >= 10 else "normal")

    plt.colorbar(im, ax=ax, label="LCS (%)\n(+ve = reduces bias,  −ve = amplifies bias)",
                 fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig4_head_contribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig4_head_contribution.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Probing: at which layer does the model "know" religion/emotion?
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("When Does the Model Encode Religion Info? (Probing Classifier Accuracy per Layer)",
             fontsize=13, fontweight="bold")
fig.text(0.5, 0.96,
         "Higher accuracy = that layer's hidden state strongly encodes religion / emotion information",
         ha="center", fontsize=9, color="gray")

for ax, pr, model in zip(axes, [pr_t5, pr_sv], ["FLAN-T5-Large", "Sarvam-2B"]):
    results = pr["results"]

    # T5 has encoder + decoder; Sarvam has only decoder
    has_encoder = "encoder_emotion" in results

    if has_encoder:
        enc_emo = [r["accuracy_mean"] * 100 for r in results["encoder_emotion"]]
        enc_rel = [r["accuracy_mean"] * 100 for r in results["encoder_religion"]]
        x_enc = np.linspace(0, 1, len(enc_emo))
        ax.plot(x_enc, enc_emo, "b-o", markersize=4, label="Encoder — Emotion", linewidth=2)
        ax.plot(x_enc, enc_rel, "b--s", markersize=4, label="Encoder — Religion", linewidth=2, alpha=0.7)

    dec_emo = [r["accuracy_mean"] * 100 for r in results["decoder_emotion"]]
    dec_rel = [r["accuracy_mean"] * 100 for r in results["decoder_religion"]]
    x_dec = np.linspace(0, 1, len(dec_emo))
    ax.plot(x_dec, dec_emo, "r-o", markersize=4, label="Decoder — Emotion", linewidth=2)
    ax.plot(x_dec, dec_rel, "r--s", markersize=4, label="Decoder — Religion", linewidth=2, alpha=0.7)

    ax.axhline(20, color="gray", linestyle=":", linewidth=1.2, label="Random chance (20%)")
    ax.set_title(f"{model}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Relative Layer Depth (0 = first, 1 = last)", fontsize=10)
    ax.set_ylabel("Probing Accuracy (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig5_probing_when_encoded.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig5_probing_when_encoded.png")

# ══════════════════════════════════════════════════════════════════════════════
# NUMERICAL SUMMARY — print and save as text
# ══════════════════════════════════════════════════════════════════════════════
lines = []
lines.append("=" * 65)
lines.append("  PHASE 3 — SIMPLE NUMERICAL SUMMARY FOR PROFESSOR REVIEW")
lines.append("=" * 65)

lines.append("\n── Q1: Religion Trigger → Emotion Change ──────────────────")
for model, summary in [("FLAN-T5-Large", s_t5), ("Sarvam-2B", s_sv)]:
    lines.append(f"\n  {model}")
    lines.append(f"  Overall: {summary['scenario_bias_rate_percent']:.0f}% of scenarios show emotion change when any religion is added")
    for r, v in summary["religion_trigger_bias_percent"].items():
        lines.append(f"    {r:10s}: {v:.0f}% of scenarios change emotion vs no-religion baseline")

lines.append("\n── Q2: Direct vs Reasoning Mode ───────────────────────────")
for model, summary in [("FLAN-T5-Large", s_t5), ("Sarvam-2B", s_sv)]:
    d = summary["mode_bias_rate_percent"]["direct"]
    r = summary["mode_bias_rate_percent"]["reasoning"]
    lines.append(f"\n  {model}")
    lines.append(f"    Direct mode bias rate   : {d:.0f}%")
    lines.append(f"    Reasoning mode bias rate: {r:.0f}%")
    lines.append(f"    Gap (reasoning - direct): +{r - d:.0f} percentage points")

lines.append("\n── Q3a: Most Impactful Layers (LCS = bias reduction when ablated) ──")
for model, ablation, control in [
        ("FLAN-T5-Large", la_t5, CONTROL_T5),
        ("Sarvam-2B",     la_sv, CONTROL_SV)]:
    lcs_d = lcs(ablation, control)
    top3  = list(lcs_d.items())[:3]
    lines.append(f"\n  {model} (baseline bias = {control:.0f}%)")
    lines.append(f"  {'Layer':<30} {'LCS':>8}  {'Ablated Bias':>14}")
    lines.append(f"  {'-'*55}")
    for k, score in top3:
        ab = control - score
        lines.append(f"  {k:<30} {score:>+7.0f}%  {ab:>13.0f}%")

lines.append("\n── Q3b: Most Impactful Attention Heads ────────────────────")
for model, ha, control in [("FLAN-T5-Large", ha_t5, CONTROL_T5), ("Sarvam-2B", ha_sv, CONTROL_SV)]:
    head_lcs = {k: control - v["metrics"]["scenario_bias_rate"] for k, v in ha.items()}
    top5 = sorted(head_lcs.items(), key=lambda x: x[1], reverse=True)[:5]
    lines.append(f"\n  {model}")
    lines.append(f"  {'Head':<40} {'LCS':>8}  {'Ablated Bias':>14}")
    lines.append(f"  {'-'*65}")
    for k, score in top5:
        ab = control - score
        lines.append(f"  {k:<40} {score:>+7.0f}%  {ab:>13.0f}%")

lines.append("\n── Key Takeaways ───────────────────────────────────────────")
lines.append("  1. Religion context causes emotion to change in 85-90% of scenarios.")
lines.append("  2. Reasoning mode amplifies bias by +41pp (T5) and +60pp (Sarvam).")
lines.append("     Generating a chain-of-thought forces the model to reason about")
lines.append("     the religion context, which amplifies the bias further.")
lines.append("  3. T5: Decoder layer 6 alone accounts for 83% of the bias —")
lines.append("     ablating it drops bias from 85% to 2%.")
lines.append("  4. Sarvam: Decoder layers 0 and 12 are the primary bias contributors")
lines.append("     (LCS = +63% and +37% respectively).")
lines.append("  5. Probing shows religion information is encoded from early layers,")
lines.append("     confirming bias is introduced during prompt processing.")
lines.append("=" * 65)

summary_text = "\n".join(lines)
print(summary_text)

with open(os.path.join(OUT_DIR, "numerical_summary.txt"), "w") as f:
    f.write(summary_text)

print(f"\nAll outputs saved to: {OUT_DIR}")
