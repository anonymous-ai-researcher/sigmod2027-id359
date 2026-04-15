"""Statistical significance tests (Appendix E.14)."""

import numpy as np
from scipy import stats


def paired_test(results_a: np.ndarray, results_b: np.ndarray, test: str = "wilcoxon"):
    """Paired significance test between two methods."""
    if test == "wilcoxon":
        stat, p = stats.wilcoxon(results_a, results_b)
    elif test == "ttest":
        stat, p = stats.ttest_rel(results_a, results_b)
    return stat, p
