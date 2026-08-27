from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


@dataclass
class WilcoxonResult:
    """Result of a paired Wilcoxon Signed-Rank test."""

    statistic: float
    p_value: float
    alpha: float
    reject_h0: bool
    n_pairs: int
    median_difference: float
    mean_difference: float


@dataclass
class FriedmanResult:
    """Result of a Friedman test over multiple paired methods."""

    statistic: float
    p_value: float
    alpha: float
    reject_h0: bool
    n_instances: int
    n_methods: int


class WilcoxonHypothesisValidator:
    """Validate paired differences using the Wilcoxon Signed-Rank test.

    This class is meant for experiments where each instance has one paired result
    per algorithm/method and lower values are better (for example, bandwidth).

    Typical use cases:
    - Compare Q-MCH against Cuthill-McKee or any other baseline.
    - Build a summary table counting the number of instances where the reference
      method improves, ties, or worsens relative to each competitor.
    - Apply the non-parametric paired Wilcoxon test when the normality assumption
      is not satisfied.
    """

    def __init__(self, alpha: float = 0.05, zero_method: str = "pratt", alternative: str = "two-sided"):
        self.alpha = float(alpha)
        self.zero_method = zero_method
        self.alternative = alternative

    @staticmethod
    def _safe_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce")

    def paired_values(
        self,
        df: pd.DataFrame,
        *,
        instance_col: str = "Instance",
        method_col: str = "centrality",
        value_col: str = "bandwidth",
        reference_method: str,
        competitor_method: str,
        aggregation: str = "min",
    ) -> pd.DataFrame:
        """Return a paired dataframe with one row per instance.

        Parameters
        ----------
        aggregation:
            How to reduce repeated rows per instance/method. The default is
            "min", which matches the idea of "best solution".
        """
        if aggregation not in {"min", "mean", "median"}:
            raise ValueError("aggregation must be one of: min, mean, median")

        work = df[[instance_col, method_col, value_col]].copy()
        work[value_col] = self._safe_numeric(work[value_col])
        work = work.dropna(subset=[instance_col, method_col, value_col])

        grouped = work.groupby([instance_col, method_col], as_index=False)[value_col].agg(aggregation)

        ref = grouped[grouped[method_col] == reference_method][[instance_col, value_col]].rename(
            columns={value_col: f"{reference_method}_value"}
        )
        comp = grouped[grouped[method_col] == competitor_method][[instance_col, value_col]].rename(
            columns={value_col: f"{competitor_method}_value"}
        )

        paired = ref.merge(comp, on=instance_col, how="inner")
        return paired

    def test_paired_samples(self, x: Sequence[float], y: Sequence[float]) -> WilcoxonResult:
        """Run the Wilcoxon Signed-Rank test on paired samples."""
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        if x_arr.shape != y_arr.shape:
            raise ValueError("Paired samples must have the same length")
        if x_arr.size == 0:
            raise ValueError("Paired samples cannot be empty")

        diffs = x_arr - y_arr
        # If all paired differences are exactly zero, scipy.wilcoxon can fail.
        # In this edge case, there is clearly no evidence against H0.
        if np.allclose(diffs, 0.0):
            return WilcoxonResult(
                statistic=0.0,
                p_value=1.0,
                alpha=self.alpha,
                reject_h0=False,
                n_pairs=int(x_arr.size),
                median_difference=float(np.median(diffs)),
                mean_difference=float(np.mean(diffs)),
            )

        stat, p_value = wilcoxon(
            x_arr,
            y_arr,
            zero_method=self.zero_method,
            alternative=self.alternative,
            correction=False,
            mode="auto",
        )
        return WilcoxonResult(
            statistic=float(stat),
            p_value=float(p_value),
            alpha=self.alpha,
            reject_h0=bool(p_value < self.alpha),
            n_pairs=int(x_arr.size),
            median_difference=float(np.median(diffs)),
            mean_difference=float(np.mean(diffs)),
        )

    def compare_methods(
        self,
        df: pd.DataFrame,
        *,
        reference_method: str,
        competitor_methods: Iterable[str],
        instance_col: str = "Instance",
        method_col: str = "centrality",
        value_col: str = "bandwidth",
        aggregation: str = "min",
        lower_is_better: bool = True,
    ) -> pd.DataFrame:
        """Compare a reference method against multiple competitors.

        Returns a DataFrame with one row per competitor containing the Wilcoxon
        statistics and a small decision summary.
        """
        rows = []
        for competitor_method in competitor_methods:
            paired = self.paired_values(
                df,
                instance_col=instance_col,
                method_col=method_col,
                value_col=value_col,
                reference_method=reference_method,
                competitor_method=competitor_method,
                aggregation=aggregation,
            )

            ref_col = f"{reference_method}_value"
            comp_col = f"{competitor_method}_value"
            ref_values = paired[ref_col].to_numpy(dtype=float)
            comp_values = paired[comp_col].to_numpy(dtype=float)
            result = self.test_paired_samples(ref_values, comp_values)

            if lower_is_better:
                ref_better = int(np.sum(ref_values < comp_values))
                comp_better = int(np.sum(ref_values > comp_values))
            else:
                ref_better = int(np.sum(ref_values > comp_values))
                comp_better = int(np.sum(ref_values < comp_values))
            ties = int(np.sum(ref_values == comp_values))

            rows.append(
                {
                    "reference_method": reference_method,
                    "competitor_method": competitor_method,
                    "n_pairs": result.n_pairs,
                    "statistic": result.statistic,
                    "p_value": result.p_value,
                    "alpha": result.alpha,
                    "reject_h0": result.reject_h0,
                    "median_difference": result.median_difference,
                    "mean_difference": result.mean_difference,
                    "reference_better": ref_better,
                    "competitor_better": comp_better,
                    "ties": ties,
                }
            )

        return pd.DataFrame(rows)

    def build_improvement_table(
        self,
        df: pd.DataFrame,
        *,
        reference_method: str,
        competitor_methods: Sequence[str],
        instance_col: str = "Instance",
        method_col: str = "centrality",
        value_col: str = "bandwidth",
        aggregation: str = "min",
        lower_is_better: bool = True,
    ) -> pd.DataFrame:
        """Build a table counting how many instances the reference method wins.

        This matches the paper-style table where each cell stores the number of
        instances in which the reference method improves the solution relative to
        the method in that column.
        """
        rows = []
        for competitor_method in competitor_methods:
            paired = self.paired_values(
                df,
                instance_col=instance_col,
                method_col=method_col,
                value_col=value_col,
                reference_method=reference_method,
                competitor_method=competitor_method,
                aggregation=aggregation,
            )
            ref_col = f"{reference_method}_value"
            comp_col = f"{competitor_method}_value"
            ref_values = paired[ref_col].to_numpy(dtype=float)
            comp_values = paired[comp_col].to_numpy(dtype=float)

            if lower_is_better:
                wins = int(np.sum(ref_values < comp_values))
                losses = int(np.sum(ref_values > comp_values))
            else:
                wins = int(np.sum(ref_values > comp_values))
                losses = int(np.sum(ref_values < comp_values))
            ties = int(np.sum(ref_values == comp_values))

            rows.append(
                {
                    "competitor_method": competitor_method,
                    "instances_compared": int(len(paired)),
                    "reference_wins": wins,
                    "competitor_wins": losses,
                    "ties": ties,
                    "win_rate": (wins / len(paired)) if len(paired) else np.nan,
                }
            )

        return pd.DataFrame(rows)

    def validate_from_dataframe(
        self,
        df: pd.DataFrame,
        *,
        reference_method: str,
        competitor_method: str,
        instance_col: str = "Instance",
        method_col: str = "centrality",
        value_col: str = "bandwidth",
        aggregation: str = "min",
        lower_is_better: bool = True,
    ) -> WilcoxonResult:
        """Convenience wrapper to test a single method pair from a DataFrame."""
        paired = self.paired_values(
            df,
            instance_col=instance_col,
            method_col=method_col,
            value_col=value_col,
            reference_method=reference_method,
            competitor_method=competitor_method,
            aggregation=aggregation,
        )
        ref_values = paired[f"{reference_method}_value"].to_numpy(dtype=float)
        comp_values = paired[f"{competitor_method}_value"].to_numpy(dtype=float)
        if lower_is_better:
            return self.test_paired_samples(ref_values, comp_values)
        return self.test_paired_samples(comp_values, ref_values)


class StatisticalDecisionSupport:
    """Helper class with descriptive and non-parametric decision tools.

    Designed for stochastic experiments where each instance can have multiple
    executions per method. The usual workflow is:
    1) Aggregate repeated runs per (instance, method) using mean/min/median.
    2) Inspect descriptive statistics.
    3) Run Friedman (global) + Holm-corrected pairwise Wilcoxon (post-hoc).
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = float(alpha)
        self.wilcoxon_validator = WilcoxonHypothesisValidator(alpha=alpha)

    @staticmethod
    def _to_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce")

    def aggregate_runs(
        self,
        df: pd.DataFrame,
        *,
        instance_col: str = "Instance",
        method_col: str = "centrality",
        value_col: str = "bandwidth",
        aggregation: str = "mean",
    ) -> pd.DataFrame:
        """Aggregate repeated stochastic runs per (instance, method)."""
        if aggregation not in {"mean", "median", "min", "max"}:
            raise ValueError("aggregation must be one of: mean, median, min, max")

        work = df[[instance_col, method_col, value_col]].copy()
        work[value_col] = self._to_numeric(work[value_col])
        work = work.dropna(subset=[instance_col, method_col, value_col])

        grouped = (
            work.groupby([instance_col, method_col], as_index=False)
            .agg(aggregated_value=(value_col, aggregation), n_runs=(value_col, "count"))
        )
        return grouped

    def descriptive_statistics(
        self,
        df: pd.DataFrame,
        *,
        method_col: str = "centrality",
        value_col: str = "bandwidth",
    ) -> pd.DataFrame:
        """Compute per-method descriptive statistics."""
        work = df[[method_col, value_col]].copy()
        work[value_col] = self._to_numeric(work[value_col])
        work = work.dropna(subset=[method_col, value_col])

        stats_df = work.groupby(method_col, as_index=False)[value_col].agg(
            n_samples="count",
            mean="mean",
            std="std",
            median="median",
            min="min",
            max="max",
            q1=lambda s: s.quantile(0.25),
            q3=lambda s: s.quantile(0.75),
        )
        stats_df["iqr"] = stats_df["q3"] - stats_df["q1"]
        return stats_df

    def build_paired_matrix(
        self,
        aggregated_df: pd.DataFrame,
        *,
        instance_col: str = "Instance",
        method_col: str = "centrality",
        value_col: str = "aggregated_value",
        methods: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Create an instance x method matrix and keep only fully paired rows."""
        pivot = aggregated_df.pivot(index=instance_col, columns=method_col, values=value_col)
        if methods is not None:
            pivot = pivot.reindex(columns=list(methods))
        pivot = pivot.dropna(axis=0, how="any")
        return pivot

    def friedman_test(
        self,
        pivot_df: pd.DataFrame,
        *,
        lower_is_better: bool = True,
    ) -> Tuple[FriedmanResult, pd.Series]:
        """Run Friedman test and return average ranks per method."""
        if pivot_df.empty:
            raise ValueError("Friedman test requires a non-empty paired matrix")
        if pivot_df.shape[1] < 3:
            raise ValueError("Friedman test requires at least 3 methods")

        arrays = [pivot_df[col].to_numpy(dtype=float) for col in pivot_df.columns]
        statistic, p_value = friedmanchisquare(*arrays)

        values = pivot_df.to_numpy(dtype=float)
        ranks = np.zeros_like(values, dtype=float)
        for i in range(values.shape[0]):
            row = values[i, :]
            if lower_is_better:
                ranks[i, :] = rankdata(row, method="average")
            else:
                ranks[i, :] = rankdata(-row, method="average")

        average_ranks = pd.Series(np.mean(ranks, axis=0), index=pivot_df.columns, name="average_rank")
        result = FriedmanResult(
            statistic=float(statistic),
            p_value=float(p_value),
            alpha=self.alpha,
            reject_h0=bool(p_value < self.alpha),
            n_instances=int(pivot_df.shape[0]),
            n_methods=int(pivot_df.shape[1]),
        )
        return result, average_ranks.sort_values()

    @staticmethod
    def holm_adjust(p_values: Sequence[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Holm-Bonferroni correction to a list of p-values.

        Returns
        -------
        adjusted_pvalues, reject_flags
        """
        p = np.asarray(p_values, dtype=float)
        m = p.size
        if m == 0:
            return np.array([], dtype=float), np.array([], dtype=bool)

        order = np.argsort(p)
        p_sorted = p[order]

        adjusted_sorted = np.empty(m, dtype=float)
        running_max = 0.0
        for i, p_i in enumerate(p_sorted):
            adjusted = (m - i) * p_i
            adjusted = min(1.0, adjusted)
            running_max = max(running_max, adjusted)
            adjusted_sorted[i] = running_max

        adjusted = np.empty(m, dtype=float)
        adjusted[order] = adjusted_sorted
        reject = adjusted < float(alpha)
        return adjusted, reject

    def friedman_holm_against_reference(
        self,
        df: pd.DataFrame,
        *,
        reference_method: str,
        methods: Optional[Sequence[str]] = None,
        instance_col: str = "Instance",
        method_col: str = "centrality",
        value_col: str = "bandwidth",
        aggregation: str = "mean",
        lower_is_better: bool = True,
    ) -> Dict[str, object]:
        """Run Friedman global test + Holm-corrected Wilcoxon post-hoc tests."""
        aggregated = self.aggregate_runs(
            df,
            instance_col=instance_col,
            method_col=method_col,
            value_col=value_col,
            aggregation=aggregation,
        )

        if methods is None:
            methods = sorted(aggregated[method_col].dropna().unique().tolist())
        if reference_method not in methods:
            raise ValueError(f"reference_method '{reference_method}' not found in methods")

        pivot = self.build_paired_matrix(
            aggregated,
            instance_col=instance_col,
            method_col=method_col,
            value_col="aggregated_value",
            methods=methods,
        )
        friedman_result, average_ranks = self.friedman_test(pivot, lower_is_better=lower_is_better)

        posthoc_rows: List[Dict[str, object]] = []
        raw_p_values: List[float] = []
        competitors: List[str] = []

        ref_values = pivot[reference_method].to_numpy(dtype=float)
        for method in methods:
            if method == reference_method:
                continue
            comp_values = pivot[method].to_numpy(dtype=float)
            if lower_is_better:
                w_result = self.wilcoxon_validator.test_paired_samples(ref_values, comp_values)
                ref_better = int(np.sum(ref_values < comp_values))
                comp_better = int(np.sum(ref_values > comp_values))
            else:
                w_result = self.wilcoxon_validator.test_paired_samples(comp_values, ref_values)
                ref_better = int(np.sum(ref_values > comp_values))
                comp_better = int(np.sum(ref_values < comp_values))

            ties = int(np.sum(ref_values == comp_values))
            raw_p_values.append(float(w_result.p_value))
            competitors.append(method)
            posthoc_rows.append(
                {
                    "reference_method": reference_method,
                    "competitor_method": method,
                    "n_pairs": int(w_result.n_pairs),
                    "statistic": float(w_result.statistic),
                    "raw_p_value": float(w_result.p_value),
                    "median_difference": float(w_result.median_difference),
                    "mean_difference": float(w_result.mean_difference),
                    "reference_better": ref_better,
                    "competitor_better": comp_better,
                    "ties": ties,
                }
            )

        adjusted_p_values, reject_flags = self.holm_adjust(raw_p_values, alpha=self.alpha)
        for i, row in enumerate(posthoc_rows):
            row["holm_adjusted_p_value"] = float(adjusted_p_values[i])
            row["reject_h0_holm"] = bool(reject_flags[i])

        posthoc_df = pd.DataFrame(posthoc_rows)
        posthoc_df = posthoc_df.sort_values(by="holm_adjusted_p_value", ascending=True).reset_index(drop=True)

        ranks_df = average_ranks.reset_index()
        ranks_df.columns = ["method", "average_rank"]

        return {
            "aggregated": aggregated,
            "paired_matrix": pivot,
            "friedman_result": friedman_result,
            "average_ranks": ranks_df,
            "posthoc": posthoc_df,
        }
