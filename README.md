# 🧠 Religion-Conditioned Emotional Bias in LLMs  
### Phase 3: Mechanistic Interpretability via Layer-wise Ablation

This repository presents a **deep mechanistic analysis of religious-context sensitivity in Large Language Models (LLMs)**, extending beyond surface-level bias evaluation into **layer-wise and head-wise interpretability**.

📄 Full ACL-style paper included: :contentReference[oaicite:0]{index=0}

---

## 🚀 Overview

Large Language Models often show **high sensitivity to religious context**, yet traditional statistical tests fail to detect significant bias.

### ⚠️ Key Paradox (Phase 2)
- **High Counterfactual Sensitivity**
  - FLAN-T5: 79%
  - Sarvam-2B: 90%
- **No Aggregate Bias**
  - Chi-square p-values: 0.21, 0.38

👉 This project answers:
> *Why do LLMs change predictions with religion but show no statistical bias?*

---

## 🔬 Phase 3 Contributions

We move beyond output-level analysis and perform **mechanistic interpretability**:

### ✅ Core Contributions
- 🔍 Layer-wise hidden state extraction
- 🧠 Attention head analysis
- ⚙️ Causal ablation (layer & head level)
- 📊 New interpretability metrics:
  - **LSS** – Layer-wise Sensitivity Score  
  - **LBS** – Layer-wise Bias Score  
  - **LCS** – Layer Contribution Score  
  - **RTAS** – Religion Token Attention Score  
- 📈 Probing classifiers (emotion & religion decoding)
- 🔗 CKA analysis for representation similarity

---

## 🧩 Key Findings

### 🧠 1. Religion is encoded early
- Encoded at **Layer 0** in both models
- Probe accuracy:
  - T5 Encoder: ~61%
  - Sarvam: ~66%

---

### ⚡ 2. Single-layer causality (T5)
- **Decoder Layer 6**
  - Responsible for **83% sensitivity**
  - Cohen’s d = **6.84 (extremely large)**

👉 Removing this layer:
- Bias drops from **85% → 2%**

---

### 🧩 3. Distributed behavior (Sarvam)
- No single dominant layer
- Multiple causal + suppressor layers

---

### 🎯 4. Specialized attention heads
- T5 Encoder **Head 9**
  - RTAS = **0.958**
  - Dedicated "religion tracker"

---

### ⚖️ 5. Final Explanation of Paradox

> Religion information is strongly encoded and amplified internally,  
> but its effects cancel out across classes → **no aggregate bias**

---

## 🏗️ Project Structure

```bash
.
├── data/                     # Scenario dataset
├── phase2/                   # Phase 2 evaluation scripts
├── phase3/
│   ├── extraction/          # Hidden state extraction
│   ├── metrics/             # LSS, LBS, LCS, RTAS
│   ├── ablation/            # Layer & head ablation
│   ├── probing/             # Classifier probes
│   ├── cka/                 # Representation similarity
│   └── visualization/       # Plots & heatmaps
├── figures/                 # All generated plots
├── report/                  # ACL paper (LaTeX)
├── run_phase3.py            # Main pipeline
└── README.md
```

# ⚙️ Setup & Installation
## Clone repo
git clone [https://github.com/your-username/religion-bias-llm.git](https://github.com/vireshkoli/Reasoning-Based-Analysis-of-Religious-Cultural-Emotional-Bias-in-Large-Language-Models.git)
cd religion-bias-llm

## Create environment
conda create -n llm_bias python=3.10
conda activate llm_bias

## Install dependencies
pip install -r requirements.txt

## Final Run
python run_phase3.py
