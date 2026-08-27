from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from modules.utils.statistical_tests import StatisticalDecisionSupport, WilcoxonHypothesisValidator


def load_experiment_dataframe(input_path: str) -> pd.DataFrame:
    """Load experiment data from CSV or JSON."""
    _, ext = os.path.splitext(input_path.lower())

    if ext == ".csv":
        df: pd.DataFrame = pd.read_csv(input_path)
    elif ext == ".json":
        with open(input_path, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        if not isinstance(payload, list):
            raise ValueError("JSON input must be a list of dictionaries")
        df = pd.DataFrame(payload)
    else:
        raise ValueError("Unsupported input extension. Use .csv or .json")

    return df


def validate_columns(df: pd.DataFrame, instance_col: str, method_col: str, value_col: str) -> None:
    """Validate required columns before running statistical analysis."""
    required_cols: List[str] = [instance_col, method_col, value_col]
    missing_cols: List[str] = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in input data: {missing_cols}")


def pick_reference_method(
    aggregated_df: pd.DataFrame,
    method_col: str = "centrality",
    value_col: str = "aggregated_value",
    lower_is_better: bool = True,
) -> str:
    """Select a reference method from aggregated data.

    The reference is chosen using the best mean aggregated value.
    """
    grouped: pd.DataFrame = (
        aggregated_df.groupby(method_col, as_index=False)[value_col].mean().rename(columns={value_col: "method_mean"})
    )
    grouped = grouped.sort_values("method_mean", ascending=lower_is_better).reset_index(drop=True)
    return str(grouped.loc[0, method_col])


def build_pairwise_wilcoxon_matrix(
    pivot_df: pd.DataFrame,
    validator: WilcoxonHypothesisValidator,
) -> pd.DataFrame:
    """Build matrix with pairwise Wilcoxon p-values among all methods."""
    methods: List[str] = list(pivot_df.columns)
    matrix = pd.DataFrame(np.nan, index=methods, columns=methods)

    for i, method_i in enumerate(methods):
        x: np.ndarray = pivot_df[method_i].to_numpy(dtype=float)
        matrix.loc[method_i, method_i] = 1.0
        for j in range(i + 1, len(methods)):
            method_j = methods[j]
            y: np.ndarray = pivot_df[method_j].to_numpy(dtype=float)
            result = validator.test_paired_samples(x, y)
            matrix.loc[method_i, method_j] = result.p_value
            matrix.loc[method_j, method_i] = result.p_value

    return matrix


def save_descriptive_plots(
    df: pd.DataFrame,
    descriptive_df: pd.DataFrame,
    ranks_df: pd.DataFrame,
    output_dir: str,
    method_col: str,
    value_col: str,
) -> None:
    """Save diagnostic plots used in test selection and model comparison."""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(11, 6))
    sns.boxplot(data=df, x=method_col, y=value_col)
    sns.stripplot(data=df, x=method_col, y=value_col, color="black", size=2, alpha=0.4)
    plt.title("Distribution of raw runs by method")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boxplot_raw_runs.png"), dpi=150)
    plt.close()

    stats_sorted = descriptive_df.sort_values("mean", ascending=True).reset_index(drop=True)
    plt.figure(figsize=(11, 6))
    plt.bar(stats_sorted[method_col], stats_sorted["mean"], yerr=stats_sorted["std"], capsize=4, color="steelblue")
    plt.title("Mean +/- std by method")
    plt.ylabel(value_col)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mean_std_by_method.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    rank_sorted = ranks_df.sort_values("average_rank", ascending=True).reset_index(drop=True)
    plt.bar(rank_sorted["method"], rank_sorted["average_rank"], color="darkorange")
    plt.title("Average ranks (Friedman)")
    plt.ylabel("Average rank (lower is better)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "friedman_average_ranks.png"), dpi=150)
    plt.close()


def save_pairwise_heatmap(pairwise_p_df: pd.DataFrame, output_dir: str) -> None:
    """Save a heatmap with pairwise Wilcoxon p-values."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(pairwise_p_df, annot=True, fmt=".3f", cmap="viridis_r", vmin=0.0, vmax=1.0)
    plt.title("Pairwise Wilcoxon p-values")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pairwise_wilcoxon_pvalues.png"), dpi=150)
    plt.close()


def recommendation_text(
    n_methods: int,
    n_instances: int,
    friedman_p: float,
    alpha: float,
) -> str:
    """Return a short recommendation for decision-making."""
    lines: List[str] = []
    lines.append("Recommended test flow for this dataset:")
    lines.append(f"- Paired instances available: {n_instances}")
    lines.append(f"- Methods available: {n_methods}")

    if n_methods >= 3:
        lines.append("- Use Friedman as global non-parametric paired test (k >= 3 methods).")
        lines.append(f"- Friedman p-value = {friedman_p:.6f} (alpha={alpha}).")
        if friedman_p < alpha:
            lines.append("- Global difference detected: run post-hoc Wilcoxon pairwise with Holm correction.")
        else:
            lines.append("- No global difference at alpha level: avoid strong post-hoc claims.")
    else:
        lines.append("- For two methods, use paired Wilcoxon Signed-Rank directly.")

    lines.append("- This workflow is suitable for non-normal paired observations.")
    return "\n".join(lines)


def run_statistical_pipeline(
    input_path: str,
    output_dir: str,
    alpha: float,
    aggregation: str,
    reference_method: str | None,
    lower_is_better: bool,
    instance_col: str,
    method_col: str,
    value_col: str,
) -> None:
    """Run full statistical pipeline and save tables and plots."""
    print("[INFO] Loading experiment data...")
    df: pd.DataFrame = load_experiment_dataframe(input_path)
    validate_columns(df, instance_col, method_col, value_col)

    print(f"[INFO] Input rows: {len(df)}")
    print(f"[INFO] Unique instances: {df[instance_col].nunique()}")
    print(f"[INFO] Unique methods: {df[method_col].nunique()}")

    engine = StatisticalDecisionSupport(alpha=alpha)
    validator = WilcoxonHypothesisValidator(alpha=alpha)

    print(f"[INFO] Aggregating repeated runs per pair using '{aggregation}'...")
    aggregated_df: pd.DataFrame = engine.aggregate_runs(
        df,
        instance_col=instance_col,
        method_col=method_col,
        value_col=value_col,
        aggregation=aggregation,
    )

    if reference_method is None:
        reference_method = pick_reference_method(
            aggregated_df,
            method_col=method_col,
            value_col="aggregated_value",
            lower_is_better=lower_is_better,
        )
        print(f"[INFO] Reference method auto-selected: {reference_method}")
    else:
        print(f"[INFO] Reference method (manual): {reference_method}")

    print("[INFO] Computing descriptive statistics...")
    descriptive_raw_df: pd.DataFrame = engine.descriptive_statistics(
        df,
        method_col=method_col,
        value_col=value_col,
    )
    descriptive_agg_df: pd.DataFrame = engine.descriptive_statistics(
        aggregated_df.rename(columns={"aggregated_value": value_col}),
        method_col=method_col,
        value_col=value_col,
    )

    methods: List[str] = sorted(aggregated_df[method_col].unique().tolist())
    print("[INFO] Running Friedman + Holm pipeline...")
    outputs: Dict[str, object] = engine.friedman_holm_against_reference(
        df,
        reference_method=reference_method,
        methods=methods,
        instance_col=instance_col,
        method_col=method_col,
        value_col=value_col,
        aggregation=aggregation,
        lower_is_better=lower_is_better,
    )

    friedman_result = outputs["friedman_result"]
    posthoc_df: pd.DataFrame = outputs["posthoc"]
    ranks_df: pd.DataFrame = outputs["average_ranks"]
    pivot_df: pd.DataFrame = outputs["paired_matrix"]

    print(
        f"[INFO] Friedman: statistic={friedman_result.statistic:.6f}, "
        f"p-value={friedman_result.p_value:.6f}, reject_h0={friedman_result.reject_h0}"
    )
    print("[INFO] Top methods by average rank (lower is better):")
    print(ranks_df.head(5).to_string(index=False))

    print("[INFO] Running reference-versus-others Wilcoxon summary table...")
    competitors: List[str] = [m for m in methods if m != reference_method]
    wilcoxon_vs_ref_df: pd.DataFrame = validator.compare_methods(
        df,
        reference_method=reference_method,
        competitor_methods=competitors,
        instance_col=instance_col,
        method_col=method_col,
        value_col=value_col,
        aggregation=aggregation,
        lower_is_better=lower_is_better,
    )

    print("[INFO] Building pairwise Wilcoxon p-value matrix...")
    pairwise_wilcoxon_p_df: pd.DataFrame = build_pairwise_wilcoxon_matrix(pivot_df, validator)

    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] Saving CSV outputs...")
    aggregated_df.to_csv(os.path.join(output_dir, "aggregated_runs.csv"), index=False)
    descriptive_raw_df.to_csv(os.path.join(output_dir, "descriptive_raw.csv"), index=False)
    descriptive_agg_df.to_csv(os.path.join(output_dir, "descriptive_aggregated.csv"), index=False)
    wilcoxon_vs_ref_df.to_csv(os.path.join(output_dir, "wilcoxon_vs_reference.csv"), index=False)
    posthoc_df.to_csv(os.path.join(output_dir, "friedman_holm_posthoc.csv"), index=False)
    ranks_df.to_csv(os.path.join(output_dir, "friedman_average_ranks.csv"), index=False)
    pairwise_wilcoxon_p_df.to_csv(os.path.join(output_dir, "pairwise_wilcoxon_pvalues.csv"))

    print("[INFO] Saving diagnostic plots...")
    save_descriptive_plots(
        df=df,
        descriptive_df=descriptive_raw_df,
        ranks_df=ranks_df,
        output_dir=output_dir,
        method_col=method_col,
        value_col=value_col,
    )
    save_pairwise_heatmap(pairwise_wilcoxon_p_df, output_dir)

    recommendation: str = recommendation_text(
        n_methods=friedman_result.n_methods,
        n_instances=friedman_result.n_instances,
        friedman_p=friedman_result.p_value,
        alpha=alpha,
    )
    recommendation_path: str = os.path.join(output_dir, "test_recommendation.txt")
    with open(recommendation_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(recommendation + "\n")

    metadata = {
        "alpha": alpha,
        "aggregation": aggregation,
        "reference_method": reference_method,
        "lower_is_better": lower_is_better,
        "input_path": input_path,
        "output_dir": output_dir,
        "friedman_result": asdict(friedman_result),
    }
    with open(os.path.join(output_dir, "run_metadata.json"), "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2)

    print("[INFO] Pipeline completed successfully.")
    print(f"[INFO] Outputs saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    default_input: str = os.path.join(base_dir, "data", "newdata", "global_analysis_inputs.json")
    default_output: str = os.path.join(base_dir, "data", "newdata", "statistical_results")

    parser = argparse.ArgumentParser(description="Statistical pipeline for stochastic graph experiments")
    parser.add_argument("--input", type=str, default=default_input, help="Input file (.json or .csv)")
    parser.add_argument("--output", type=str, default=default_output, help="Directory to save results")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--aggregation", type=str, default="mean", choices=["mean", "median", "min", "max"])
    parser.add_argument("--reference", type=str, default=None, help="Reference method (optional)")
    parser.add_argument("--higher-is-better", action="store_true", help="Use when larger values are better")
    parser.add_argument("--instance-col", type=str, default="Instance")
    parser.add_argument("--method-col", type=str, default="centrality")
    parser.add_argument("--value-col", type=str, default="bandwidth")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_statistical_pipeline(
        input_path=args.input,
        output_dir=args.output,
        alpha=float(args.alpha),
        aggregation=str(args.aggregation),
        reference_method=args.reference,
        lower_is_better=not bool(args.higher_is_better),
        instance_col=str(args.instance_col),
        method_col=str(args.method_col),
        value_col=str(args.value_col),
    )
