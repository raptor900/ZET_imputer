# ZET Imputer

Missing values imputation for 2D tables based on the **ZET algorithm** (Zagoruiko, Elkina, Timerkaev, 1975).

## Algorithm

The ZET algorithm fills missing values in empirical tables by finding **competent (similar) rows and columns** and using weighted linear regression to predict missing entries.

### How it works

For each missing value at position `(i, j)`:

1. **Row competence**: Find rows most similar to row `i` using inverse Euclidean distance on non-missing overlapping values.
2. **Compact submatrix**: Select top-k competent rows and form a compact submatrix.
3. **Column competence**: Within the submatrix, find columns most similar to column `j` using absolute Pearson correlation.
4. **Row regression**: Predict the missing value using weighted linear regression across competent rows.
5. **Column regression**: Predict using weighted linear regression across competent columns.
6. **Combine**: Average the two predictions.

Key properties:
- Uses **only local data** (competent subset), not the full table
- Does **not require pre-filling** missing values (unlike EM/SVD)
- Adapts to non-linear patterns better than global regression
- Supports **iterative refinement** (re-impute with updated values)

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scale` | `"standard"` | Feature scaling: `"standard"`, `"minmax"`, or `"none"` |
| `cmshape` | `5` | Compact submatrix size (cmshape × cmshape) |
| `alpha` | `3.0` | Competence weighting exponent (higher → more weight to best matches) |
| `use_imputed` | `True` | Use previously imputed values for subsequent predictions |
| `max_iter` | `1` | Number of imputation iterations |

## Usage

### Basic

```python
import numpy as np
from zet import zet_fill

data = np.array([
    [1.0, 2.0, np.nan],
    [4.0, np.nan, 6.0],
    [7.0, 8.0, 9.0],
])

filled = zet_fill(data)
print(filled)
```

### With pandas DataFrame

```python
import pandas as pd
from zet import zet_fill

df = pd.DataFrame({
    "a": [1, 4, 7],
    "b": [2, np.nan, 8],
    "c": [np.nan, 6, 9],
})

result = zet_fill(df, scale="minmax", cmshape=3)
```

### Sklearn-style API

```python
from zet import ZETImputer

imputer = ZETImputer(scale="standard", cmshape=5, alpha=3)
X_filled = imputer.fit_transform(X)
```

### Iterative refinement

```python
# Run 3 iterations for better convergence
filled = zet_fill(data, max_iter=3)
```

## Reference

Загоруйко Н.Г., Елкина В.Н., Тимеркаев В.С.
*«Алгоритм заполнения пропусков в эмпирических таблицах (алгоритм Zet)»*
Вычислительные системы, вып. 61. Новосибирск, 1975. С. 3–27.

## Requirements

- Python ≥ 3.8
- numpy
- pandas
- scipy
- scikit-learn
