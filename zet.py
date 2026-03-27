#!/usr/bin/env python3
"""
ZET Imputer — missing values imputation for 2D tables.

Based on the ZET algorithm by Zagoruiko, Elkina, Timerkaev (1975).

References:
    Загоруйко Н.Г., Елкина В.Н., Тимеркаев В.С.
    «Алгоритм заполнения пропусков в эмпирических таблицах (алгоритм Zet)»
    Вычислительные системы, вып. 61. Новосибирск, 1975. С. 3–27.
"""

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)

# Type alias for input data
ArrayLike = Union[np.ndarray, pd.DataFrame]


def _row_competence(x: np.ndarray, y: np.ndarray) -> float:
    """Compute row competence (similarity) between two rows.

    Uses inverse Euclidean distance on non-missing overlapping values,
    multiplied by the number of overlapping values.

    Args:
        x: First row (may contain NaN).
        y: Second row (may contain NaN).

    Returns:
        Competence score (higher = more similar). Returns 0 if no overlap.
    """
    mask = ~np.isnan(x) & ~np.isnan(y)
    n_overlap = mask.sum()
    if n_overlap == 0:
        return 0.0
    dist = euclidean(x[mask], y[mask])
    if dist == 0:
        return float("inf")  # identical rows
    return (1.0 / dist) * n_overlap


def _col_competence(x: np.ndarray, y: np.ndarray) -> float:
    """Compute column competence (correlation) between two columns.

    Uses absolute Pearson correlation on non-missing overlapping values,
    multiplied by the number of overlapping values.

    Args:
        x: First column (may contain NaN).
        y: Second column (may contain NaN).

    Returns:
        Competence score. Returns 0 if insufficient overlap.
    """
    mask = ~np.isnan(x) & ~np.isnan(y)
    n_overlap = mask.sum()
    if n_overlap <= 1:
        return 1.0e-15 if n_overlap == 1 else 0.0
    corr = abs(np.corrcoef(x[mask], y[mask])[0, 1])
    if np.isnan(corr):
        return 1.0e-15
    return corr * n_overlap


def _linear_predict(x: np.ndarray, y: np.ndarray, x_target: float) -> float:
    """Predict y[target] using linear regression from x -> y.

    Args:
        x: Predictor values (may contain NaN).
        y: Target values (may contain NaN).
        x_target: The x value at which to predict.

    Returns:
        Predicted y value, or 0.0 if regression impossible.
    """
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() == 0:
        return 0.0
    x_fit = x[mask].reshape(-1, 1)
    y_fit = y[mask]
    lr = LinearRegression()
    lr.fit(x_fit, y_fit)
    return lr.predict([[x_target]])[0]


def zet_fill(
    df: ArrayLike,
    scale: str = "standard",
    cmshape: int = 5,
    alpha: float = 3.0,
    use_imputed: bool = True,
    max_iter: int = 1,
) -> np.ndarray:
    """Fill missing values in a 2D table using the ZET algorithm.

    The algorithm works in two phases for each missing value:
    1. **Row phase**: Find the most similar rows (competent rows) and
       predict the missing value using weighted linear regression across rows.
    2. **Column phase**: Transpose the compact submatrix, find competent
       columns, and predict using weighted linear regression across columns.
    3. **Combine**: Average the row and column predictions.

    Args:
        df: Input data as numpy array or pandas DataFrame with NaN for
            missing values.
        scale: Feature scaling method:
            - 'standard': StandardScaler (zero mean, unit variance)
            - 'minmax': MinMaxScaler (0-1 range)
            - 'none': No scaling
        cmshape: Size of the compact submatrix (cmshape × cmshape).
            Larger values use more data but are slower.
        alpha: Exponent for competence weighting. Higher values give
            more weight to the most similar rows/columns.
        use_imputed: If True, use previously imputed values when
            computing competence for subsequent missing values.
            If False, always use original data (parallel imputation).
        max_iter: Number of imputation iterations. >1 enables iterative
            refinement (impute → re-compute competences → re-impute).

    Returns:
        numpy array of same shape as input with imputed missing values.

    Raises:
        ValueError: If input is not 2D or has no missing values.

    Example:
        >>> import numpy as np
        >>> data = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])
        >>> filled = zet_fill(data)
        >>> print(filled)
    """
    # Input validation
    df_arr = np.array(df, dtype=float)
    if df_arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {df_arr.ndim}D")

    nan_mask = np.isnan(df_arr)
    if not nan_mask.any():
        logger.info("No missing values found, returning copy")
        return df_arr.copy()

    n_missing = nan_mask.sum()
    logger.info(
        f"ZET imputation: {df_arr.shape[0]}×{df_arr.shape[1]} table, "
        f"{n_missing} missing values ({100 * n_missing / df_arr.size:.1f}%)"
    )

    # Scaling
    scaler = None
    if scale == "standard":
        scaler = StandardScaler()
        # Fill NaN with column means for fitting scaler
        df_filled = pd.DataFrame(df_arr).fillna(pd.DataFrame(df_arr).mean())
        dfs = scaler.fit_transform(df_filled)
        dfs[nan_mask] = np.nan
    elif scale == "minmax":
        scaler = MinMaxScaler()
        df_filled = pd.DataFrame(df_arr).fillna(pd.DataFrame(df_arr).mean())
        dfs = scaler.fit_transform(df_filled)
        dfs[nan_mask] = np.nan
    elif scale == "none":
        dfs = df_arr.copy()
    else:
        raise ValueError(f"Unknown scale method: {scale!r}. Use 'standard', 'minmax', or 'none'")

    N, M = dfs.shape
    dfs_copy = dfs.copy()

    for iteration in range(max_iter):
        if max_iter > 1:
            logger.debug(f"Iteration {iteration + 1}/{max_iter}")

        nan_args = np.argwhere(np.isnan(dfs_copy))
        for row_idx, col_idx in nan_args:
            # --- ROW PHASE ---
            # Build competence scores for all rows
            source = dfs_copy if use_imputed else dfs
            row_a = source[row_idx]

            comp_rows = np.zeros(N)
            for i in range(N):
                if i != row_idx and not np.isnan(source[i, col_idx]):
                    comp_rows[i] = _row_competence(row_a, source[i])

            # Select top-k competent rows
            valid_rows = comp_rows > 0
            n_valid = valid_rows.sum()
            k = min(n_valid, cmshape - 1)

            if k == 0:
                logger.warning(
                    f"No competent rows for ({row_idx}, {col_idx})"
                )
                dfs_copy[row_idx, col_idx] = np.nan
                continue

            # Get indices of top-k rows + target row
            top_row_indices = np.argsort(comp_rows)[::-1][:k]
            comp_row_indices = np.append(top_row_indices, row_idx)

            # --- COLUMN PHASE ---
            # Extract compact submatrix and compute column competences
            submatrix = dfs_copy[np.ix_(comp_row_indices, range(M))]
            sub_target_row = submatrix[-1]  # target row is last

            comp_cols = np.zeros(M)
            for j in range(M):
                if j != col_idx and not np.isnan(submatrix[-1, j]):
                    comp_cols[j] = _col_competence(
                        submatrix[:, col_idx], submatrix[:, j]
                    )

            valid_cols = comp_cols > 0
            n_valid_cols = valid_cols.sum()
            k_col = min(n_valid_cols, cmshape - 1)

            if k_col == 0:
                logger.warning(
                    f"No competent columns for ({row_idx}, {col_idx})"
                )
                dfs_copy[row_idx, col_idx] = np.nan
                continue

            top_col_indices = np.argsort(comp_cols)[::-1][:k_col]
            comp_col_indices = np.append(top_col_indices, col_idx)

            # Final compact submatrix
            compact = submatrix[np.ix_(range(len(comp_row_indices)), comp_col_indices)]

            if compact.shape[0] <= 1 or compact.shape[1] <= 1:
                logger.warning(
                    f"Insufficient data in compact submatrix for "
                    f"({row_idx}, {col_idx})"
                )
                dfs_copy[row_idx, col_idx] = np.nan
                continue

            # --- ROW REGRESSION ---
            pred_r = _weighted_row_predict(
                compact, comp_row_indices, row_idx, comp_rows, alpha
            )

            # --- COLUMN REGRESSION ---
            compact_T = compact.T
            pred_c = _weighted_col_predict(
                compact_T, comp_col_indices, col_idx, comp_cols, alpha
            )

            # Combine predictions
            pred = 0.5 * (pred_r + pred_c)
            dfs_copy[row_idx, col_idx] = pred

    # Inverse scaling
    if scaler is not None:
        return scaler.inverse_transform(dfs_copy)
    return dfs_copy


def _weighted_row_predict(
    compact: np.ndarray,
    comp_row_indices: np.ndarray,
    target_row_idx: int,
    comp_rows: np.ndarray,
    alpha: float,
) -> float:
    """Predict target value using weighted regression across rows."""
    y_col = compact[:, -1]  # target column (last)
    sum_pred = 0.0
    sum_weight = 0.0

    for i, ri in enumerate(comp_row_indices):
        if ri == target_row_idx:
            continue
        x_row = compact[i]
        weight = comp_rows[ri] ** alpha
        pred = _linear_predict(x_row, y_col, len(x_row) - 1)
        sum_pred += pred * weight
        sum_weight += weight

    return sum_pred / sum_weight if sum_weight > 0 else 0.0


def _weighted_col_predict(
    compact_T: np.ndarray,
    comp_col_indices: np.ndarray,
    target_col_idx: int,
    comp_cols: np.ndarray,
    alpha: float,
) -> float:
    """Predict target value using weighted regression across columns."""
    y_row = compact_T[:, -1]  # target row (last)
    sum_pred = 0.0
    sum_weight = 0.0

    for j, cj in enumerate(comp_col_indices):
        if cj == target_col_idx:
            continue
        x_col = compact_T[j]
        weight = comp_cols[cj] ** alpha
        pred = _linear_predict(x_col, y_row, len(x_col) - 1)
        sum_pred += pred * weight
        sum_weight += weight

    return sum_pred / sum_weight if sum_weight > 0 else 0.0


# Convenience alias for sklearn-style usage
class ZETImputer:
    """Sklearn-compatible wrapper for ZET imputation.

    Example:
        >>> imputer = ZETImputer(scale="standard", cmshape=5, alpha=3)
        >>> X_filled = imputer.fit_transform(X)
    """

    def __init__(
        self,
        scale: str = "standard",
        cmshape: int = 5,
        alpha: float = 3.0,
        use_imputed: bool = True,
        max_iter: int = 1,
    ):
        self.scale = scale
        self.cmshape = cmshape
        self.alpha = alpha
        self.use_imputed = use_imputed
        self.max_iter = max_iter

    def fit(self, X: ArrayLike, y=None):
        """No-op for compatibility. Returns self."""
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Impute missing values."""
        return zet_fill(
            X,
            scale=self.scale,
            cmshape=self.cmshape,
            alpha=self.alpha,
            use_imputed=self.use_imputed,
            max_iter=self.max_iter,
        )

    def fit_transform(self, X: ArrayLike, y=None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
