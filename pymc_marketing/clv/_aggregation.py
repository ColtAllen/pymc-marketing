#   Copyright 2022 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Weighted-observation aggregation shared by the CLV models.

A likelihood whose per-customer contribution depends only on a small set of
columns (for example ``(cohort, recency, T)`` in the sBG model) can be evaluated
once per *unique* combination of those columns and weighted by the number of
customers that share it. This module deduplicates the customer-level frame,
keeps the inverse map back to customers, and expands per-row posterior
quantities back to the customer level.

The module is model-agnostic: it imports no model and makes no assumption about
which columns form the key. Each model's distribution applies the weights
inside its own ``logp``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import arviz_stats as az
import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import xarray as xr
from arviz_stats.utils import ELPDData

__all__ = [
    "UNIQUE_DIM",
    "AggregatedObservations",
    "aggregate_observations",
    "customer_loo",
    "expand_to_customers",
    "identity_aggregation",
    "survival_counts_to_sbg_data",
]

#: Dimension name of the deduplicated likelihood rows in the model graph.
UNIQUE_DIM = "unique_obs"

#: ``unique rows / customers`` above which aggregation buys little and a warning is emitted.
LOW_COMPRESSION_RATIO = 0.5

_ROW = "_row"
_UNIQUE_IDX = "_unique_idx"


@dataclass(frozen=True)
class AggregatedObservations:
    """Deduplicated observations plus the map back to customers.

    Attributes
    ----------
    unique : pandas.DataFrame
        One row per unique key combination with the columns ``key_cols`` and
        ``"weight"``. Row ``i`` is likelihood row ``i`` (coordinate ``i`` of
        :data:`UNIQUE_DIM`). Rows are sorted by ``key_cols`` so the graph is
        reproducible for a given data set.
    inverse : numpy.ndarray
        ``int64`` array of length ``n_customers``: the row of ``unique`` each
        customer belongs to.
    customer_id : numpy.ndarray
        Customer identifiers, aligned with ``inverse``.
    key_cols : tuple of str
        Columns that define a unique likelihood row.
    """

    unique: pd.DataFrame
    inverse: np.ndarray
    customer_id: np.ndarray
    key_cols: tuple[str, ...]

    @property
    def weights(self) -> np.ndarray:
        """Weight of each unique row as ``float64``."""
        return self.unique["weight"].to_numpy(dtype="float64")

    @property
    def n_unique(self) -> int:
        """Number of unique likelihood rows."""
        return len(self.unique)

    @property
    def n_customers(self) -> int:
        """Number of customers (rows of the input frame)."""
        return len(self.customer_id)

    @property
    def total_weight(self) -> float:
        """Sum of the row weights: the number of customers represented."""
        return float(self.weights.sum())

    @property
    def compression_ratio(self) -> float:
        """``n_unique / total_weight``; lower is better.

        Equals ``n_unique / n_customers`` for unweighted input. For pre-aggregated input
        the denominator is the number of customers the weights stand for, not the number
        of rows passed in.
        """
        return self.n_unique / self.total_weight


def _check_columns(schema: nw.Schema, required: Sequence[str]) -> None:
    missing = [col for col in required if col not in schema]
    if missing:
        raise KeyError(
            f"Column(s) {missing} not found in data. Available columns: {schema.names()}."
        )


def aggregate_observations(
    data: Any,
    key_cols: Sequence[str],
    weight_col: str | None = None,
    *,
    warn_ratio: float | None = LOW_COMPRESSION_RATIO,
) -> AggregatedObservations:
    """Deduplicate ``data`` on ``key_cols`` and count (or sum) the rows of each group.

    Parameters
    ----------
    data : dataframe
        Any Narwhals-compatible frame (pandas, Polars eager or lazy, PyArrow, DuckDB,
        ...) with a ``customer_id`` column, ``key_cols`` and optionally ``weight_col``.
    key_cols : sequence of str
        Columns that define a unique likelihood row. Grouping is exact; values are never
        rounded.
    weight_col : str, optional
        Column of strictly positive weights to sum within each group. When omitted every
        row weighs ``1`` and the weight of a group is its row count. Summing lets an
        already aggregated frame be aggregated again.
    warn_ratio : float, optional
        Emit a :class:`UserWarning` when ``n_unique / n_customers`` exceeds this value.
        ``None`` disables the warning.

    Returns
    -------
    AggregatedObservations

    Notes
    -----
    Eager inputs keep their row order in ``customer_id`` and ``inverse``. Lazy inputs
    (Polars ``LazyFrame``, DuckDB) have no row order, so they are ordered by
    ``customer_id`` instead.
    """
    key_cols = tuple(key_cols)
    frame = nw.from_native(data)
    eager = isinstance(frame, nw.DataFrame)
    schema = frame.collect_schema()
    _check_columns(
        schema, ["customer_id", *key_cols, *([weight_col] if weight_col else [])]
    )
    keys = list(key_cols)

    if eager:
        rows = frame.with_row_index(_ROW).lazy()
    else:
        rows = frame.lazy().with_row_index(_ROW, order_by="customer_id")

    weight = nw.col(weight_col).sum() if weight_col else nw.len()
    unique_lf = (
        rows.group_by(keys)
        .agg(weight.alias("weight"))
        .with_columns(nw.col("weight").cast(nw.Float64))
        .sort(keys)
        .with_row_index(_UNIQUE_IDX, order_by=keys)
    )
    unique = unique_lf.collect().to_pandas().reset_index(drop=True)

    if unique[keys].isna().any().any():
        raise ValueError(
            f"Columns {keys} contain null values; drop or fill them first."
        )
    if weight_col is not None:
        min_weight = rows.select(nw.col(weight_col).min()).collect().item()
        if not (min_weight is not None and np.isfinite(min_weight) and min_weight > 0):
            raise ValueError(
                f"Column {weight_col!r} must be finite and strictly positive."
            )
        n_null = rows.select(nw.col(weight_col).is_null().sum()).collect().item()
        if n_null:
            raise ValueError(
                f"Column {weight_col!r} must be finite and strictly positive."
            )

    mapped = (
        rows.select(_ROW, "customer_id", *keys)
        .join(unique_lf.select(*keys, _UNIQUE_IDX), on=keys, how="left")
        .sort(_ROW)
        .select("customer_id", _UNIQUE_IDX)
        .collect()
        .to_pandas()
    )
    if mapped[_UNIQUE_IDX].isna().any():
        raise ValueError(
            f"Columns {keys} contain null values; drop or fill them first."
        )

    agg = AggregatedObservations(
        unique=unique.drop(columns=_UNIQUE_IDX),
        inverse=mapped[_UNIQUE_IDX].to_numpy(dtype="int64"),
        customer_id=mapped["customer_id"].to_numpy(),
        key_cols=key_cols,
    )
    if warn_ratio is not None and agg.compression_ratio > warn_ratio:
        warnings.warn(
            f"Aggregation keeps {agg.n_unique} rows for {agg.total_weight:g} customers "
            f"(compression ratio {agg.compression_ratio:.2f} > {warn_ratio}). "
            "Continuous or high-cardinality key columns defeat aggregation; "
            "consider aggregate=False or coarser covariates.",
            UserWarning,
            stacklevel=2,
        )
    return agg


def identity_aggregation(
    data: pd.DataFrame, key_cols: Sequence[str]
) -> AggregatedObservations:
    """Treat every customer as its own likelihood row with weight ``1``.

    This is the ``aggregate=False`` path: ``unique`` is the input key columns in input
    order, ``inverse`` is ``arange(n)`` and all weights are ``1``.
    """
    key_cols = tuple(key_cols)
    unique = data[list(key_cols)].reset_index(drop=True).copy()
    unique["weight"] = 1.0
    return AggregatedObservations(
        unique=unique,
        inverse=np.arange(len(data), dtype="int64"),
        customer_id=data["customer_id"].to_numpy(),
        key_cols=key_cols,
    )


def expand_to_customers(
    da: xr.DataArray,
    agg: AggregatedObservations,
    dim: str = UNIQUE_DIM,
    *,
    max_bytes: int = 2 * 1024**3,
) -> xr.DataArray:
    """Gather a per-unique-row array back to one entry per customer.

    ``dim`` is replaced by ``customer_id`` in place; every other dimension and every
    non-dimension coordinate attached to ``dim`` is kept, so ``swap_dims`` /
    ``sel(cohort=...)`` patterns keep working on the result.

    Parameters
    ----------
    da : xarray.DataArray
        Array with a ``dim`` dimension of length ``agg.n_unique``.
    agg : AggregatedObservations
    dim : str
        Name of the unique-row dimension in ``da``.
    max_bytes : int
        Refuse to build an output larger than this (default 2 GiB). Posterior-sized
        arrays expanded to millions of customers do not fit in memory.
    """
    if dim not in da.dims:
        raise ValueError(f"DataArray has no {dim!r} dimension; dims are {da.dims}.")
    if da.sizes[dim] != agg.n_unique:
        raise ValueError(
            f"DataArray {dim!r} has length {da.sizes[dim]}, expected {agg.n_unique}."
        )
    estimated = da.nbytes // max(da.sizes[dim], 1) * agg.n_customers
    if estimated > max_bytes:
        raise MemoryError(
            f"Expanding to {agg.n_customers} customers needs about {estimated / 1024**3:.1f} GiB, "
            f"above max_bytes={max_bytes}. Reduce the array first (e.g. average over draws)."
        )
    idx = xr.DataArray(
        agg.inverse, dims="customer_id", coords={"customer_id": agg.customer_id}
    )
    out = da.isel({dim: idx})
    return out.drop_vars(dim, errors="ignore")


def survival_counts_to_sbg_data(
    df: Any,
    cohort_col: str = "cohort",
    period_col: str = "period",
    alive_col: str = "n_alive",
) -> pd.DataFrame:
    r"""Convert a cohort survival table into weighted sBG input rows.

    Parameters
    ----------
    df : dataframe
        One row per ``(cohort, period)`` with the number of customers still active at
        the end of that period. Periods must run ``0, 1, ..., T - 1`` within each cohort,
        with ``period == 0`` the cohort size, and ``n_alive`` must be non-increasing.
    cohort_col, period_col, alive_col : str
        Column names.

    Returns
    -------
    pandas.DataFrame
        Columns ``customer_id`` (a synthetic configuration id), ``cohort``, ``recency``,
        ``T`` and ``weight``. Customers who churned at period ``t`` get ``recency = t``
        (``t = 1, ..., T - 1``); those still active after period ``T - 1`` get
        ``recency = T``. Zero-weight rows are dropped.

    Notes
    -----
    ``recency == T`` is right-censored: the sBG likelihood evaluates
    :math:`P(X \\geq T) = S(T - 1)`, which is what the customer-level model computes for
    an active customer. ``T`` for a cohort is its number of observed periods.
    """
    frame = nw.from_native(df)
    if isinstance(frame, nw.LazyFrame):
        frame = frame.collect()
    _check_columns(frame.collect_schema(), [cohort_col, period_col, alive_col])
    table = (
        frame.select(cohort_col, period_col, alive_col)
        .sort([cohort_col, period_col])
        .to_pandas()
        .reset_index(drop=True)
    )

    out = []
    for cohort, group in table.groupby(cohort_col, sort=True):
        periods = group[period_col].to_numpy()
        alive = group[alive_col].to_numpy(dtype="float64")
        T = len(alive)
        if not np.array_equal(periods, np.arange(T)):
            raise ValueError(
                f"Cohort {cohort!r}: {period_col} must be 0, 1, ..., T - 1 without gaps; "
                f"got {periods.tolist()}."
            )
        if np.any(np.diff(alive) > 0):
            raise ValueError(
                f"Cohort {cohort!r}: {alive_col} must be non-increasing over {period_col}."
            )
        churned = -np.diff(
            alive
        )  # churn at period t = alive[t-1] - alive[t], t = 1..T-1
        recency = np.concatenate([np.arange(1, T), [T]])
        weight = np.concatenate([churned, [alive[-1]]])
        out.append(
            pd.DataFrame(
                {"cohort": cohort, "recency": recency, "T": T, "weight": weight}
            )
        )

    result = pd.concat(out, ignore_index=True)
    result = result[result["weight"] > 0].reset_index(drop=True)
    result.insert(0, "customer_id", np.arange(1, len(result) + 1))
    return result


def customer_loo(
    data: xr.DataTree,
    agg: AggregatedObservations,
    var_name: str | None = None,
    dim: str = UNIQUE_DIM,
    **loo_kwargs: Any,
) -> ELPDData:
    """Customer-level PSIS-LOO from a model fitted on aggregated rows.

    Every customer in unique row ``k`` has the same likelihood ``p(y_k | theta)``, hence
    the same PSIS importance ratios, ``elpd_loo_i`` and Pareto ``k``. Customer-level LOO
    is therefore an exact weighted reduction of the row-level result; no
    ``draws x n_customers`` array is ever built.

    Parameters
    ----------
    data : xarray.DataTree
        Inference data with a ``posterior`` group and a ``log_likelihood`` group whose
        ``var_name`` variable holds the **unweighted** per-row log-likelihood over ``dim``.
    agg : AggregatedObservations
        The aggregation the model was built with.
    var_name : str, optional
        Log-likelihood variable; required when the group holds several.
    dim : str
        Row dimension of the log-likelihood (default :data:`UNIQUE_DIM`).
    **loo_kwargs
        Forwarded to :func:`arviz_stats.loo` (for example ``reff``).

    Returns
    -------
    arviz_stats.utils.ELPDData
        ``elpd``, ``se``, ``p`` and ``n_data_points`` refer to customers; ``elpd_i`` and
        ``pareto_k`` have a ``customer_id`` dimension. The object works directly in
        :func:`arviz_stats.compare`.
    """
    loo_kwargs.pop("pointwise", None)
    rows = az.loo(data, var_name=var_name, pointwise=True, **loo_kwargs)
    if rows.elpd_i.dims != (dim,):
        raise ValueError(
            f"Expected pointwise results over {dim!r}, got dims {rows.elpd_i.dims}."
        )
    if rows.elpd_i.sizes[dim] != agg.n_unique:
        raise ValueError(
            f"log-likelihood has {rows.elpd_i.sizes[dim]} rows, aggregation has {agg.n_unique}."
        )

    log_lik_ds = data["log_likelihood"].to_dataset()
    if var_name is None:
        var_name = next(iter(log_lik_ds.data_vars))
    log_lik = log_lik_ds[var_name]
    sample = log_lik.stack(__sample__=("chain", "draw"))
    shift = sample.max("__sample__")
    lppd_i = shift + np.log(np.exp(sample - shift).mean("__sample__"))

    weights = xr.DataArray(agg.weights, dims=dim)
    n_customers = float(weights.sum())
    elpd = float((weights * rows.elpd_i).sum())
    mean = elpd / n_customers
    se = float(np.sqrt((weights * (rows.elpd_i - mean) ** 2).sum()))
    p_loo = float((weights * (lppd_i - rows.elpd_i)).sum())
    n_data_points = (
        round(n_customers)
        if np.isclose(n_customers, round(n_customers))
        else n_customers
    )

    return ELPDData(
        kind="loo",
        elpd=elpd,
        se=se,
        p=p_loo,
        n_samples=rows.n_samples,
        n_data_points=n_data_points,
        scale="log",
        warning=rows.warning,
        good_k=rows.good_k,
        elpd_i=expand_to_customers(rows.elpd_i, agg, dim=dim).rename("elpd_i"),
        pareto_k=expand_to_customers(rows.pareto_k, agg, dim=dim).rename("pareto_k"),
        approx_posterior=False,
        log_weights=rows.log_weights,
    )
