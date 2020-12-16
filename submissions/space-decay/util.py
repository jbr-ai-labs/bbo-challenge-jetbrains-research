import numpy as np
import scipy.stats as ss


def order_stats(X):
  _, idx, cnt = np.unique(X, return_inverse=True, return_counts=True)
  obs = np.cumsum(cnt)  # Need to do it this way due to ties
  o_stats = obs[idx]
  return o_stats


def copula_standardize(X):
  X = np.nan_to_num(np.asarray(X))  # Replace inf by something large
  assert X.ndim == 1 and np.all(np.isfinite(X))
  o_stats = order_stats(X)
  quantile = np.true_divide(o_stats, len(X) + 1)
  X_ss = ss.norm.ppf(quantile)
  return X_ss
