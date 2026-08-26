from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


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
