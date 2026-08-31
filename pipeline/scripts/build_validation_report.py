#!/usr/bin/env python3
"""Build the validation report and plots from available validation outputs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PIPELINE_ROOT.parent
sys.path.insert(0, str(PIPELINE_ROOT))

import pandas as pd  # noqa: E402

from validation.holdout import PLOTS_DIR, VALIDATION_DIR, BENCHMARK_REFERENCE_ROWS  # noqa: E402


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def make_plots() -> list[str]:
    predictions = read_csv(VALIDATION_DIR / "megastudy_holdout_predictions.csv")
    study_metrics = read_csv(VALIDATION_DIR / "megastudy_holdout_study_metrics.csv")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if predictions.empty:
        return []
    os.environ.setdefault("MPLCONFIGDIR", str(PLOTS_DIR / ".matplotlib"))
    import matplotlib.pyplot as plt

    paths: list[str] = []

    def save(name: str) -> None:
        path = PLOTS_DIR / name
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(str(path.relative_to(REPO_ROOT)))

    for name, pred_col, title in [
        ("megastudy_holdout_scatter_raw.png", "raw_f_ate_pp", "Raw F vs human"),
        ("megastudy_holdout_scatter_calibrated.png", "calibrated_f_ate_pp", "Calibrated F/C vs human"),
    ]:
        plt.figure(figsize=(5, 5))
        plt.scatter(predictions["human_ate_pp"], predictions[pred_col], alpha=0.65)
        low = min(predictions["human_ate_pp"].min(), predictions[pred_col].min())
        high = max(predictions["human_ate_pp"].max(), predictions[pred_col].max())
        plt.plot([low, high], [low, high], color="black", linewidth=1)
        plt.xlabel("Human effect pp")
        plt.ylabel(pred_col)
        plt.title(title)
        save(name)

    ordered = predictions.sort_values(["study_id", "effect_id"]).reset_index(drop=True)
    x = range(len(ordered))
    plt.figure(figsize=(12, 4))
    plt.plot(x, ordered["raw_error_pp"].abs(), label="raw abs error")
    plt.plot(x, ordered["calibrated_error_pp"].abs(), label="calibrated abs error")
    plt.legend()
    plt.title("Raw vs calibrated absolute error")
    save("raw_vs_calibrated_error.png")

    if not study_metrics.empty:
        pivot = study_metrics.pivot(index="study_id", columns="prediction", values="rmse")
        pivot.plot(kind="bar", figsize=(10, 4))
        plt.ylabel("RMSE")
        save("megastudy_rmse_by_study.png")

        pivot = study_metrics.pivot(index="study_id", columns="prediction", values="sign_accuracy")
        pivot.plot(kind="bar", figsize=(10, 4))
        plt.ylabel("Sign accuracy")
        save("megastudy_sign_accuracy_by_study.png")

        corrs = study_metrics[["study_id", "prediction", "pearson", "spearman", "n_effects"]].copy()
        corrs["label"] = corrs["study_id"] + " n=" + corrs["n_effects"].astype(str)
        plt.figure(figsize=(10, 4))
        plt.scatter(corrs["label"], corrs["pearson"], label="Pearson")
        plt.scatter(corrs["label"], corrs["spearman"], label="Spearman")
        plt.xticks(rotation=70, ha="right")
        plt.legend()
        save("megastudy_within_study_correlations.png")

    plt.figure(figsize=(6, 5))
    plt.scatter(predictions["human_ate_pp"], predictions["raw_f_ate_pp"], alpha=0.5, label="raw")
    plt.scatter(predictions["human_ate_pp"], predictions["calibrated_f_ate_pp"], alpha=0.5, label="calibrated")
    low = min(predictions["human_ate_pp"].min(), predictions["raw_f_ate_pp"].min(), predictions["calibrated_f_ate_pp"].min())
    high = max(predictions["human_ate_pp"].max(), predictions["raw_f_ate_pp"].max(), predictions["calibrated_f_ate_pp"].max())
    plt.plot([low, high], [low, high], color="black", linewidth=1)
    plt.legend()
    plt.xlabel("Human effect pp")
    plt.ylabel("Prediction pp")
    save("calibration_before_after.png")

    examples = predictions.groupby("study_id").size().sort_values(ascending=False).head(3).index.tolist()
    if examples:
        fig, axes = plt.subplots(len(examples), 1, figsize=(10, 3 * len(examples)))
        if len(examples) == 1:
            axes = [axes]
        for ax, study_id in zip(axes, examples):
            part = predictions[predictions["study_id"] == study_id].copy()
            part = part.sort_values("human_ate_pp", ascending=False).head(20)
            labels = part["effect_id"].astype(str).str[-16:]
            ax.plot(labels, part["human_ate_pp"].rank(ascending=False), label="human")
            ax.plot(labels, part["raw_f_ate_pp"].rank(ascending=False), label="raw F")
            ax.plot(labels, part["calibrated_f_ate_pp"].rank(ascending=False), label="calibrated F/C")
            ax.set_title(f"{study_id} intervention ranks")
            ax.tick_params(axis="x", labelrotation=70)
        axes[0].legend()
        save("intervention_rank_examples.png")
    return paths


def main() -> int:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    plots = make_plots()
    usage = read_csv(VALIDATION_DIR / "data_usage_audit.csv")
    split = read_csv(VALIDATION_DIR / "validation_split_manifest.csv")
    eligibility = read_csv(VALIDATION_DIR / "megastudy_holdout_eligibility.csv")
    pooled = read_csv(VALIDATION_DIR / "megastudy_holdout_pooled_metrics.csv")
    study_metrics = read_csv(VALIDATION_DIR / "megastudy_holdout_study_metrics.csv")
    comparison = read_csv(VALIDATION_DIR / "raw_vs_calibrated_holdout.csv")
    diagnostic = read_csv(VALIDATION_DIR / "holdout_diagnostic_calibration_regression.csv")
    overlap = read_json(VALIDATION_DIR / "climate_holdout_overlap_audit.json")
    holdout_status = read_json(VALIDATION_DIR / "holdout_status.json")
    frozen = read_json(VALIDATION_DIR / "frozen_method_manifest.json")

    primary = split[split["archive"] == "primary_70_study_archive"] if not split.empty else pd.DataFrame()
    secondary = split[split["archive"] == "secondary_15_megastudy_archive"] if not split.empty else pd.DataFrame()
    lines = [
        "# Validation Report",
        "",
        "## Development Data",
        "",
        f"- Primary archive studies listed: {len(primary)}",
        f"- Development/calibration studies: {int((primary.get('assigned_role', pd.Series(dtype=str)) == 'development_calibration').sum()) if not primary.empty else 0}",
        "- Uses: M0/M1/M2 selection and final C estimation only for primary-eligible effects.",
        "",
        "## Structural Holdout",
        "",
        f"- Secondary megastudies listed: {int(secondary['study_id'].nunique()) if not secondary.empty else 0}",
        f"- Secondary effect rows enumerated: {len(eligibility)}",
        f"- Eligible effect rows under current documented metadata: {int((eligibility.get('eligible', pd.Series(dtype=bool)) == True).sum()) if not eligibility.empty else 0}",
        f"- Holdout opened: {bool(holdout_status.get('holdout_opened_at'))}",
        f"- Holdout pristine: {holdout_status.get('holdout_still_pristine', 'unknown')}",
        "",
        "## Raw F Performance",
        "",
        pooled.to_markdown(index=False) if not pooled.empty else "Pending: structural holdout has not been opened.",
        "",
        "## Calibrated F/C Performance",
        "",
        comparison.to_markdown(index=False) if not comparison.empty else "Pending: structural holdout has not been opened.",
        "",
        "## Holdout Calibration Diagnostic",
        "",
        diagnostic.to_markdown(index=False) if not diagnostic.empty else "Pending: diagnostic regression not run.",
        "",
        "## Contextual Benchmark Reference",
        "",
        pd.DataFrame(BENCHMARK_REFERENCE_ROWS).to_markdown(index=False),
        "",
        "## Climate-Domain Holdout",
        "",
        f"- Status: {overlap.get('status', 'unknown')}",
        f"- Secondary archive match: {overlap.get('secondary_archive_match', 'unknown')}",
        f"- Matched study id: {overlap.get('matched_study_id', 'unknown')}",
        f"- Human outcomes already used in repo: {overlap.get('human_outcomes_already_used_in_repo', 'unknown')}",
        "",
        "## G Validation Status",
        "",
        "- G validation remains separate under `outputs/validation/g_validation/`.",
        "- Existing implementation: `submission.g_validation.validate_g_against_human`.",
        "",
        "## Holdout Integrity",
        "",
        f"- Frozen method hash: {frozen.get('method_hash', 'NOT_FROZEN')}",
        f"- Git commit in frozen manifest: {frozen.get('git_commit', 'NOT_FROZEN')}",
        f"- Method changed after holdout: {holdout_status.get('method_changed_after_holdout', 'unknown')}",
        "",
        "## Final Status",
        "",
        "- Calibration archive pipeline: PASS if `run_primary_calibration.py` completes.",
        "- Structural holdout integrity: PASS if holdout is unopened or frozen manifest exists before opening.",
        "- F raw ranking: numeric only after holdout opening; no arbitrary pass threshold.",
        "- F/C absolute calibration: numeric only after holdout opening; no arbitrary pass threshold.",
        "- Climate-domain validation: WARNING if contained in structural holdout or development-contaminated.",
        "- G external validation: PENDING until frozen G respondent-level validation datasets are run.",
        "",
        "## Plots",
        "",
        "\n".join(f"- `{plot}`" for plot in plots) if plots else "Pending: plots are generated after structural holdout predictions exist.",
        "",
    ]
    out = VALIDATION_DIR / "validation_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
