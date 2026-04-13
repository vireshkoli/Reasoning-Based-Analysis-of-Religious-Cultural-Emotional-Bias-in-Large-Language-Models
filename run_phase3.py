"""
Master orchestration script for Phase 3: Layer-wise Ablation & Interpretability Study.

Runs the full pipeline in order:
  Step 1: Hidden state extraction (T5 then Sarvam)
  Step 2: Layer-wise metric computation (LSS, LBS, CKA)
  Step 3: Probing classifiers (emotion + religion per layer)
  Step 4: Ablation experiments (layer bypass + head masking)
  Step 5: Statistical validation
  Step 6: Analysis scripts (layer analysis, attention, cross-model)
  Step 7: Visualization (all plots)

"""

import argparse
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PHASE3_DIR = os.path.join(PROJECT_ROOT, "results", "phase3")
STATE_FILE = os.path.join(PHASE3_DIR, "pipeline_state.json")


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"completed_steps": [], "timestamps": {}}


def save_state(state):
    os.makedirs(PHASE3_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def mark_complete(state, step_name):
    if step_name not in state["completed_steps"]:
        state["completed_steps"].append(step_name)
    state["timestamps"][step_name] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_state(state)


def run_script(script_path, args=None, cwd=None):
    """Run a Python script as a subprocess. Raises on failure."""
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    print(f"\n  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd or PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Script failed with exit code {result.returncode}: {script_path}")


def step_extract(model="both"):
    """Step 1: Extract hidden states and attention weights."""
    print("\n" + "=" * 70)
    print("STEP 1: Hidden State Extraction")
    print("=" * 70)
    run_script(
        os.path.join(PROJECT_ROOT, "experiments", "phase3_extract.py"),
        args=["--model", model],
    )


def step_metrics(model="both"):
    """Step 2: Compute layer-wise metrics (LSS, LBS, CKA)."""
    print("\n" + "=" * 70)
    print("STEP 2: Layer-wise Metric Computation")
    print("=" * 70)
    run_script(
        os.path.join(PROJECT_ROOT, "experiments", "phase3_layerwise_metrics.py"),
        args=["--model", model, "--cross-model"],
    )


def step_probe(model="both"):
    """Step 3: Train probing classifiers per layer."""
    print("\n" + "=" * 70)
    print("STEP 3: Probing Classifiers")
    print("=" * 70)
    run_script(
        os.path.join(PROJECT_ROOT, "experiments", "phase3_probe.py"),
        args=["--model", model],
    )


def step_ablation(model, experiment="both"):
    """Step 4: Ablation experiments."""
    print("\n" + "=" * 70)
    print(f"STEP 4: Ablation Experiments ({model.upper()})")
    print("=" * 70)
    run_script(
        os.path.join(PROJECT_ROOT, "experiments", "phase3_ablation.py"),
        args=["--model", model, "--experiment", experiment],
    )


def step_statistical():
    """Step 5: Statistical validation."""
    print("\n" + "=" * 70)
    print("STEP 5: Statistical Validation")
    print("=" * 70)
    run_script(
        os.path.join(PROJECT_ROOT, "analysis", "phase3_statistical.py"),
        cwd=os.path.join(PROJECT_ROOT, "analysis"),
    )


def step_analyze():
    """Step 6: Run all analysis scripts."""
    print("\n" + "=" * 70)
    print("STEP 6: Analysis Scripts")
    print("=" * 70)
    for script in ["phase3_layer_analysis.py", "phase3_attention_analysis.py",
                   "phase3_cross_model.py"]:
        run_script(
            os.path.join(PROJECT_ROOT, "analysis", script),
            cwd=os.path.join(PROJECT_ROOT, "analysis"),
        )


def step_visualize():
    """Step 7: Generate all visualizations."""
    print("\n" + "=" * 70)
    print("STEP 7: Visualization")
    print("=" * 70)
    for script in ["layer_curves.py", "attention_heatmaps.py",
                   "hidden_projections.py", "ablation_plots.py",
                   "cross_model_plots.py"]:
        run_script(
            os.path.join(PROJECT_ROOT, "visualization", script),
            cwd=os.path.join(PROJECT_ROOT, "visualization"),
        )


def run_full_pipeline(resume=False, model="both"):
    """Run the complete Phase 3 pipeline."""
    state = load_state()

    def should_run(step_name):
        if resume and step_name in state["completed_steps"]:
            print(f"\n  [SKIP] {step_name} already completed")
            return False
        return True

    start_time = time.time()
    print("\n" + "=" * 70)
    print("PHASE 3: LAYER-WISE ABLATION & INTERPRETABILITY PIPELINE")
    print("=" * 70)

    try:
        if should_run("extract"):
            step_extract(model)
            mark_complete(state, "extract")

        if should_run("metrics"):
            step_metrics(model)
            mark_complete(state, "metrics")

        if should_run("probe"):
            step_probe(model)
            mark_complete(state, "probe")

        # Ablation: run per model (memory constraint — one model at a time)
        for m in (["t5", "sarvam"] if model == "both" else [model]):
            step_key = f"ablation_{m}"
            if should_run(step_key):
                step_ablation(m)
                mark_complete(state, step_key)

        if should_run("statistical"):
            step_statistical()
            mark_complete(state, "statistical")

        if should_run("analyze"):
            step_analyze()
            mark_complete(state, "analyze")

        if should_run("visualize"):
            step_visualize()
            mark_complete(state, "visualize")

        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"PHASE 3 COMPLETE — Total time: {elapsed/60:.1f} minutes")
        print(f"Results in: {PHASE3_DIR}")
        print(f"Dashboard: {os.path.join(PHASE3_DIR, 'plots', 'dashboard_phase3.png')}")
        print(f"{'='*70}")

    except RuntimeError as e:
        print(f"\nPIPELINE FAILED: {e}")
        print("Fix the error and re-run with --resume to skip completed steps")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Layer-wise Ablation & Interpretability Pipeline"
    )
    parser.add_argument("--step", choices=[
        "extract", "metrics", "probe", "ablation", "statistical", "analyze", "visualize"
    ], help="Run a single pipeline step")
    parser.add_argument("--model", choices=["t5", "sarvam", "both"], default="both",
                        help="Which model(s) to process")
    parser.add_argument("--experiment", choices=["layer", "head", "both"], default="both",
                        help="Ablation experiment type (used with --step ablation)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed steps")
    args = parser.parse_args()

    if args.step:
        step_fn = {
            "extract": lambda: step_extract(args.model),
            "metrics": lambda: step_metrics(args.model),
            "probe": lambda: step_probe(args.model),
            "ablation": lambda: step_ablation(args.model, args.experiment),
            "statistical": step_statistical,
            "analyze": step_analyze,
            "visualize": step_visualize,
        }
        step_fn[args.step]()
    else:
        run_full_pipeline(resume=args.resume, model=args.model)


if __name__ == "__main__":
    main()
