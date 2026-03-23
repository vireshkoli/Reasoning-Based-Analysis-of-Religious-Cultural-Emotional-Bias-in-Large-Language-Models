# 🧠 Reasoning-Based Analysis of Religious & Cultural Emotional Bias in Large Language Models

## 📌 Overview
This project investigates **religious and cultural emotional bias** in Large Language Models (LLMs) using a **scenario-based evaluation framework**.

We analyze how model outputs change when **religious context is varied**, and whether LLMs exhibit:
- Systematic bias
- Counterfactual sensitivity
- Reasoning inconsistencies

The project compares multiple LLMs to understand fairness and robustness in **emotion-based reasoning tasks**.

---

## 🎯 Objectives
- Evaluate **emotion prediction consistency** across religious contexts  
- Detect **bias in LLM-generated responses**  
- Analyze **reasoning vs non-reasoning behavior**  
- Perform **statistical validation** of bias (Chi-square tests)  
- Compare performance across different models  

---

## 🧪 Methodology

### 1. Scenario-Based Evaluation
- Created **100 real-world scenarios**
- Domains include:
  - Family
  - Career
  - Social situations

Each scenario is rewritten across **different religious contexts**.

---

### 2. Model Evaluation
We evaluate:
- **Sarvam LLM**
- **Google FLAN-T5**

Each model generates:
- Emotion prediction
- Reasoning (in reasoning mode)

---

### 3. Experiment Design
- Total predictions: **1000 per model**
- Conditions:
  - With reasoning
  - Without reasoning
  - Counterfactual variations

---

### 4. Statistical Analysis
- **Chi-Square Test** → Detect bias across religions  
- **Counterfactual Sensitivity Analysis**  
- **Reasoning Amplification Analysis**

---

## 📊 Key Results
- ❌ No statistically significant **religion-based bias** detected  
- ⚠️ High **counterfactual sensitivity** observed  
- ⚠️ Reasoning mode **amplifies variation in outputs**  

---
