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
"""Plotting functions for the CLV module."""

import warnings
from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
from arviz_stats.ecdf_utils import (
    compute_ecdf,
    get_pointwise_confidence_band,
    simulate_confidence_bands,
)
from matplotlib.lines import Line2D

from pymc_marketing.clv import BetaGeoModel, ParetoNBDModel
from pymc_marketing.clv.utils import (
    _expected_cumulative_transactions,
    _find_first_transactions,
)

__all__ = [
    "plot_cohorts",
    "plot_customer_exposure",
    "plot_expected_purchases_over_time",
    "plot_expected_purchases_ppc",
    "plot_frequency_recency_matrix",
    "plot_probability_alive_matrix",
]

# Predictive samples per observation below which the ECDF confidence band is too narrow, because
# the reference CDF is estimated rather than known. Simulated coverage of a nominal 95% band, on
# negative-binomial counts with n = 2357: 0.28-0.41 at a ratio of 1, 0.83 at 10, 0.89 at 20, and
# indistinguishable from nominal (0.90-0.96, within Monte Carlo error at these replicate counts)
# from 50 upwards.
_MIN_PPC_RATIO = 20

# Predictive draws per customer used to estimate the ECDF reference CDF on a point-estimate fit,
# whose posterior holds a single draw. Bounded rather than taken from `samples`, because the cost is
# linear in the draws while the coverage above saturates well before this: at 20k customers a ratio
# of 1000 costs 646 MB and 171 s, against 38 MB and 9 s at 50, for no measurable gain.
_REFERENCE_DRAWS = 100

# Cell count above which per-cell annotations on a cohort heatmap become unreadable, so they
# are suppressed unless the caller asks for them explicitly. A daily transaction log easily
# produces hundreds of cohorts by hundreds of periods.
_MAX_ANNOT_CELLS = 200


def plot_customer_exposure(
    df: pd.DataFrame,
    linewidth: float | None = None,
    size: float | None = None,
    labels: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
    padding: float = 0.25,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the recency and T of DataFrame of customers.

    Plots customers as horizontal lines with markers representing their recency and T starting.
    Order is the same as the DataFrame and plotted from the bottom up.

    The lines are colored by recency and T.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with columns "recency" and "T" representing the recency and age of customers.
    linewidth : float, optional
        The width of the horizontal lines in the plot.
    size : float, optional
        The size of the markers in the plot.
    labels : Sequence[str], optional
        A sequence of labels for the legend. Default is ["Recency", "T"].
    colors : Sequence[str], optional
        A sequence of colors for the legend. Default is ["C0", "C1"].
    padding : float, optional
        The padding around the plot. Default is 0.25.
    ax : plt.Axes, optional
        A matplotlib axes instance to plot on. If None, a new figure and axes is created.

    Returns
    -------
    plt.Axes
        The matplotlib axes instance.

    Examples
    --------
    Plot customer exposure

    .. code-block:: python

        df = pd.DataFrame({"recency": [0, 1, 2, 3, 4], "T": [5, 5, 5, 5, 5]})

        plot_customer_exposure(df)

    Plot customer exposure ordered by recency and T

    .. code-block:: python

        (df.sort_values(["recency", "T"]).pipe(plot_customer_exposure))

    Plot exposure for only those with time until last purchase is less than 3

    .. code-block:: python

        (df.query("T - recency < 3").pipe(plot_customer_exposure))

    """
    if padding < 0:
        raise ValueError("padding must be non-negative")

    if size is not None and size < 0:
        raise ValueError("size must be non-negative")

    if linewidth is not None and linewidth < 0:
        raise ValueError("linewidth must be non-negative")

    if ax is None:
        ax = plt.gca()

    n = len(df)
    customer_idx = np.arange(1, n + 1)

    recency = df["recency"].to_numpy()
    T = df["T"].to_numpy()

    if colors is None:
        colors = ["C0", "C1"]

    if len(colors) != 2:
        raise ValueError("colors must be a sequence of length 2")

    recency_color, T_color = colors

    ax.hlines(
        y=customer_idx, xmin=0, xmax=recency, linewidth=linewidth, color=recency_color
    )
    ax.hlines(y=customer_idx, xmin=recency, xmax=T, linewidth=linewidth, color=T_color)

    ax.scatter(x=recency, y=customer_idx, linewidth=linewidth, s=size, c=recency_color)
    ax.scatter(x=T, y=customer_idx, linewidth=linewidth, s=size, c=T_color)

    ax.set(
        xlabel="Time since first purchase",
        ylabel="Customer",
        xlim=(0 - padding, T.max() + padding),
        ylim=(1 - padding, n + padding),
        title="Customer Exposure",
    )

    if labels is None:
        labels = ["Recency", "T"]

    if len(labels) != 2:
        raise ValueError("labels must be a sequence of length 2")

    recency_label, T_label = labels

    legend_elements = [
        Line2D([0], [0], color=recency_color, label=recency_label),
        Line2D([0], [0], color=T_color, label=T_label),
    ]

    ax.legend(handles=legend_elements, loc="best")

    return ax


def _cohorts_from_transactions(
    data: pd.DataFrame,
    customer_id_col: str,
    datetime_col: str,
    time_unit: str,
    datetime_format: str | None,
    sort_transactions: bool | None,
) -> pd.DataFrame:
    """Derive customer-level cohort labels and activity ages from a transaction log.

    A customer's cohort is the period of their first transaction, and their *age of last
    activity* is the number of periods between that and their final transaction.
    """
    transactions = _find_first_transactions(
        data,
        customer_id_col,
        datetime_col,
        datetime_format=datetime_format,
        time_unit=time_unit,
        sort_transactions=sort_transactions,
    )

    spans = transactions.groupby(customer_id_col)[datetime_col].agg(["min", "max"])
    cohort_ordinal = np.asarray(pd.PeriodIndex(spans["min"]).astype("int64"))
    last_ordinal = np.asarray(pd.PeriodIndex(spans["max"]).astype("int64"))
    observation_end = pd.PeriodIndex(transactions[datetime_col]).max()

    return pd.DataFrame(
        {
            "cohort": spans["min"].to_numpy(),
            "age_last": last_ordinal - cohort_ordinal,
            "n_ages": observation_end.ordinal - cohort_ordinal,
        }
    )


def _cohorts_from_summary(data: pd.DataFrame, cohort_col: str | None) -> pd.DataFrame:
    """Derive customer-level cohort labels and activity ages from a summary DataFrame.

    *recency* is taken as the age of last activity. When no cohort column is available the
    acquisition cohort is recovered from *T*: a larger *T* means an earlier acquisition.
    """
    missing = [col for col in ("recency", "T") if col not in data.columns]
    if missing:
        raise ValueError(
            f"Summary data must contain 'recency' and 'T' columns; missing {missing}. "
            "Pass 'datetime_col' to supply a raw transaction log instead."
        )

    if cohort_col is None and "cohort" in data.columns:
        cohort_col = "cohort"

    if cohort_col is not None:
        if cohort_col not in data.columns:
            raise ValueError(f"'{cohort_col}' is not a column of the provided data.")
        labels = data[cohort_col].to_numpy()
    else:
        T = data["T"].to_numpy(dtype=float)
        labels = np.rint(T.max() - T).astype(np.int64)
        if len(np.unique(labels)) == 1:
            warnings.warn(
                "All customers share the same 'T', so they collapse into a single cohort. "
                "Supply 'cohort_col' to group customers explicitly.",
                UserWarning,
                stacklevel=3,
            )

    return pd.DataFrame(
        {
            "cohort": labels,
            "age_last": data["recency"].to_numpy(),
            "n_ages": data["T"].to_numpy(),
        }
    )


def _cohort_retention_table(customers: pd.DataFrame, show_pct: bool) -> pd.DataFrame:
    """Count customers still active at each cohort age, one row per cohort and age."""
    frames = []

    for label, group in customers.groupby("cohort", sort=True):
        n_ages = int(np.floor(group["n_ages"].max()))
        if n_ages < 1:
            continue
        ages = np.arange(n_ages)
        # A customer is retained at age ``a`` when their last observed activity falls
        # strictly after it, which makes the resulting curve non-increasing in age.
        surviving = (group["age_last"].to_numpy()[:, None] > ages[None, :]).sum(axis=0)
        value = 100 * surviving / len(group) if show_pct else surviving
        frames.append(
            pd.DataFrame({"cohort": label, "cohort_age": ages, "value": value})
        )

    if not frames:
        raise ValueError(
            "No cohort has a full period of follow-up, so there is nothing to plot."
        )

    return pd.concat(frames, ignore_index=True)


def _cohort_periods(labels: pd.Series, time_unit: str) -> dict | None:
    """Map cohort labels onto periods, or return ``None`` when they are not dates."""
    unique = pd.Index(pd.unique(labels))

    if isinstance(unique, pd.PeriodIndex):
        return {label: label for label in unique}
    if unique.dtype.kind in "iuf":
        return None

    try:
        with warnings.catch_warnings():
            # Mixed or unusual label formats warn about falling back to dateutil; the
            # except branch below already handles labels that are not dates at all.
            warnings.simplefilter("ignore", UserWarning)
            periods = pd.to_datetime(unique).to_period(time_unit)
    except (ValueError, TypeError):
        return None

    return dict(zip(unique, periods, strict=True))


def _warn_on_cohort_spacing(period_map: dict) -> None:
    """Warn when *time_unit* is finer than the spacing between consecutive cohorts."""
    periods = pd.PeriodIndex(sorted(set(period_map.values())))
    if len(periods) < 2:
        return

    step = np.diff(np.asarray(periods.astype("int64"))).min()
    if step > 1:
        warnings.warn(
            f"Consecutive cohorts are {step} '{periods.freqstr}' periods apart. 'time_unit' "
            "likely does not match the cohort spacing, which will stretch the calendar axis.",
            UserWarning,
            stacklevel=3,
        )


def plot_cohorts(
    data: pd.DataFrame,
    *,
    cohort_col: str | None = None,
    customer_id_col: str = "customer_id",
    datetime_col: str | None = None,
    time_unit: str = "D",
    datetime_format: str | None = None,
    sort_transactions: bool | None = True,
    show_pct: bool = False,
    x_axis: Literal["auto", "calendar", "cohort_age"] = "auto",
    annot: bool | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "Cohort",
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Plot observed cohort retention as a heatmap.

    Each row is a cohort and each column a point in time, with cells reporting how many of
    that cohort were still active. This is a description of the observed data only; no model
    is fitted or consulted, so it accepts the data shapes used by every CLV transaction
    model.

    Data can be supplied in either of two forms:

    * a **summary DataFrame** with *recency* and *T*, as consumed by ``BetaGeoModel``,
      ``ModifiedBetaGeoModel``, ``ParetoNBDModel``, ``BetaGeoBetaBinomModel`` and
      ``ShiftedBetaGeoModel``;
    * a **raw transaction log**, by passing *datetime_col*.

    A customer counts as retained at cohort age ``a`` when their last observed activity falls
    strictly after ``a``. Note what this means for the first column, because it differs by
    model family. ``ShiftedBetaGeoModel`` records *recency* as the one-indexed period of the
    last contract renewal, so *recency* is always at least 1 and age 0 is 100% by
    construction. In RFM data *recency* is elapsed time since the first purchase, which is 0
    for one-time buyers, so age 0 sits below 100% and reads as the repeat-purchase rate.
    Percentages are always relative to the full cohort size.

    Parameters
    ----------
    data : ~pandas.DataFrame
        Either a summary DataFrame containing *recency* and *T*, or a raw transaction log
        containing *customer_id_col* and *datetime_col*.
    cohort_col : str, optional
        Column holding the cohort label. Defaults to ``"cohort"`` when that column is
        present. If neither is available the acquisition cohort is derived from *T*, since a
        larger *T* implies an earlier acquisition. Ignored when *datetime_col* is given.
    customer_id_col : str, optional
        Column denoting the customer ID. Only used for a transaction log. Default:
        ``"customer_id"``.
    datetime_col : str, optional
        Column denoting the transaction datetimes. Supplying it switches *data* from summary
        to transaction-log interpretation.
    time_unit : str, optional
        Length of one time period, both for resampling a transaction log and for advancing
        the calendar axis. Default: ``'D'`` for days. Possible values listed here:
        https://numpy.org/devdocs/reference/arrays.datetime.html#datetime-units
    datetime_format : str, optional
        A string that represents the timestamp format. Useful if Pandas doesn't recognize
        the provided format.
    sort_transactions : bool, optional
        Default: *True*. If a transaction log is already sorted in chronological order, set
        to *False* to improve computational efficiency.
    show_pct : bool, optional
        Default: *False*. Show retention as a percentage of the cohort's starting size rather
        than as raw customer counts.
    x_axis : str, optional
        ``'calendar'`` places cohorts on a shared absolute timeline, giving the familiar
        upper-triangular chart. ``'cohort_age'`` indexes columns by periods since the cohort
        started, giving a rectangular chart, and is the only option when cohort labels are
        not dates. Default: ``'auto'``, which uses ``'calendar'`` where the labels permit it.
    annot : bool, optional
        Write the value into each cell. Defaults to *True* for small matrices and *False*
        beyond 200 cells, where the annotations stop being readable.
    title : str, optional
        Figure title
    xlabel : str, optional
        Figure xlabel
    ylabel : str, optional
        Figure ylabel
    ax : matplotlib.Axes, optional
        A matplotlib Axes instance. Creates new axes instance by default.
    kwargs
        Passed into the seaborn.heatmap command.

    Returns
    -------
    axes : matplotlib.AxesSubplot

    Raises
    ------
    ValueError
        If *x_axis* is not a recognized option, if ``'calendar'`` is requested for cohort
        labels that are not dates, if a summary DataFrame lacks *recency* or *T*, or if no
        cohort has a full period of follow-up.

    Examples
    --------
    Retention percentages from a summary DataFrame:

    .. code-block:: python

        from pymc_marketing.clv import plot_cohorts

        plot_cohorts(rfm_data, show_pct=True)

    A monthly cohort chart built straight from a transaction log:

    .. code-block:: python

        plot_cohorts(
            transactions,
            customer_id_col="id",
            datetime_col="date",
            time_unit="M",
        )

    """
    if x_axis not in ("auto", "calendar", "cohort_age"):
        raise ValueError("'x_axis' must be one of 'auto', 'calendar' or 'cohort_age'.")

    if datetime_col is not None:
        customers = _cohorts_from_transactions(
            data,
            customer_id_col,
            datetime_col,
            time_unit,
            datetime_format,
            sort_transactions,
        )
    else:
        customers = _cohorts_from_summary(data, cohort_col)

    table = _cohort_retention_table(customers, show_pct)
    period_map = _cohort_periods(table["cohort"], time_unit)

    if period_map is None:
        if x_axis == "calendar":
            raise ValueError(
                "Cohort labels could not be interpreted as dates, so 'calendar' is not "
                "available for 'x_axis'. Use 'cohort_age' instead."
            )
        use_calendar = False
    else:
        use_calendar = x_axis != "cohort_age"

    if period_map is not None and use_calendar:
        starts = table["cohort"].map(period_map)
        columns = pd.PeriodIndex(
            [
                start + age
                for start, age in zip(starts, table["cohort_age"], strict=True)
            ]
        )
        table = table.assign(column=columns.astype(str))
        _warn_on_cohort_spacing(period_map)
    else:
        table = table.assign(column=table["cohort_age"])

    pivot = table.pivot(index="cohort", columns="column", values="value")
    pivot = pivot.sort_index()[sorted(pivot.columns)]

    if annot is None:
        annot = pivot.size <= _MAX_ANNOT_CELLS

    if ax is None:
        ax = plt.subplot(111)

    if show_pct:
        fmt = ".0f"
        cbar_format = mtick.FuncFormatter(lambda y, _: f"{y:.0f}%")
        default_title = "Cohort Retention Rate (%)"
    else:
        fmt = ",.0f"
        cbar_format = mtick.FuncFormatter(lambda y, _: f"{y:,.0f}")
        default_title = "Cohort Customer Counts"

    kwargs.setdefault("cmap", "viridis_r")
    kwargs.setdefault("linewidths", 0.2)
    kwargs.setdefault("linecolor", "black")
    kwargs.setdefault("cbar_kws", {"format": cbar_format})

    sns.heatmap(pivot, annot=annot, fmt=fmt, ax=ax, **kwargs)

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set(
        title=default_title if title is None else title,
        xlabel=("Time Period" if use_calendar else "Cohort Age")
        if xlabel is None
        else xlabel,
        ylabel=ylabel,
    )

    return ax


def _create_frequency_recency_meshes(
    max_frequency: int,
    max_recency: int,
) -> tuple[np.ndarray, np.ndarray]:
    frequency = np.arange(max_frequency + 1)
    recency = np.arange(max_recency + 1)
    mesh_frequency, mesh_recency = np.meshgrid(frequency, recency)

    return mesh_frequency, mesh_recency


def plot_frequency_recency_matrix(
    model: BetaGeoModel | ParetoNBDModel,
    future_t: int = 1,
    max_frequency: int | None = None,
    max_recency: int | None = None,
    title: str | None = None,
    xlabel: str = "Customer's Historical Frequency",
    ylabel: str = "Customer's Recency",
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Plot expected purchases in *future_t* time periods as a heatmap based on customer population *frequency* and *recency*.

    Parameters
    ----------
    model: CLV model
        A fitted CLV model.
    future_t: float, optional
        Future time periods over which to run predictions.
    max_frequency: int, optional
        The maximum *frequency* to plot. Defaults to max observed *frequency*.
    max_recency: int, optional
        The maximum *recency* to plot. This also determines the age of the customer. Defaults to max observed *recency*.
    title: str, optional
        Figure title
    xlabel: str, optional
        Figure xlabel
    ylabel: str, optional
        Figure ylabel
    ax: plt.Axes, optional
        A matplotlib axes instance. Creates new axes instance by default.
    kwargs
        Passed into the matplotlib.imshow command.

    Returns
    -------
    axes: matplotlib.AxesSubplot

    """  # noqa: E501
    if max_frequency is None:
        max_frequency = int(model.data["frequency"].max())

    if max_recency is None:
        max_recency = int(model.data["recency"].max())

    mesh_frequency, mesh_recency = _create_frequency_recency_meshes(
        max_frequency=max_frequency,
        max_recency=max_recency,
    )

    # create dataframe for model input
    transaction_data = pd.DataFrame(
        {
            "customer_id": np.arange(mesh_recency.size),  # placeholder
            "frequency": mesh_frequency.ravel(),
            "recency": mesh_recency.ravel(),
            "T": max_recency,
        }
    )

    # run model predictions to create heatmap values
    Z = (
        model.expected_purchases(
            data=transaction_data,
            future_t=future_t,
        )
        .mean(("draw", "chain"))
        .values.reshape(mesh_recency.shape)
    )

    if ax is None:
        ax = plt.subplot(111)

    pcm = ax.imshow(Z, **kwargs)
    if title is None:
        title = (
            "Expected Number of Future Purchases for {} Unit{} of Time,".format(
                future_t, "s"[future_t == 1 :]
            )
            + "\nby Frequency and Recency of a Customer"
        )

    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
    )

    _force_aspect(ax)

    # plot colorbar beside matrix
    plt.colorbar(pcm, ax=ax)

    return ax


def plot_probability_alive_matrix(
    model: BetaGeoModel | ParetoNBDModel,
    max_frequency: int | None = None,
    max_recency: int | None = None,
    title: str = "Probability Customer is Alive,\nby Frequency and Recency of a Customer",
    xlabel: str = "Customer's Historical Frequency",
    ylabel: str = "Customer's Recency",
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Plot probability alive matrix as a heatmap based on customer population *frequency* and *recency*.

    Parameters
    ----------
    model: CLV model
        A fitted CLV model.
    max_frequency: int, optional
        The maximum *frequency* to plot. Defaults to max observed *frequency*.
    max_recency: int, optional
        The maximum *recency* to plot. This also determines the age of the customer. Defaults to max observed *recency*.
    title: str, optional
        Figure title
    xlabel: str, optional
        Figure xlabel
    ylabel: str, optional
        Figure ylabel
    ax: plt.Axes, optional
        A matplotlib axes instance. Creates new axes instance by default.
    kwargs
        Passed into the matplotlib.imshow command.

    Returns
    -------
    axes: matplotlib.AxesSubplot

    """
    if max_frequency is None:
        max_frequency = int(model.data["frequency"].max())

    if max_recency is None:
        max_recency = int(model.data["recency"].max())

    mesh_frequency, mesh_recency = _create_frequency_recency_meshes(
        max_frequency=max_frequency,
        max_recency=max_recency,
    )

    # create dataframe for model input
    transaction_data = pd.DataFrame(
        {
            "customer_id": np.arange(mesh_recency.size),  # placeholder
            "frequency": mesh_frequency.ravel(),
            "recency": mesh_recency.ravel(),
            "T": max_recency,
        }
    )

    # run model predictions to create heatmap values
    Z = (
        model.expected_probability_alive(data=transaction_data)
        .mean(("draw", "chain"))
        .values.reshape(mesh_recency.shape)
    )

    interpolation = kwargs.pop("interpolation", "none")

    if ax is None:
        ax = plt.subplot(111)

    pcm = ax.imshow(Z, interpolation=interpolation, **kwargs)

    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
    )
    _force_aspect(ax)

    # plot colorbar beside matrix
    plt.colorbar(pcm, ax=ax)

    return ax


def plot_expected_purchases_over_time(
    model,
    purchase_history: pd.DataFrame,
    customer_id_col: str,
    datetime_col: str,
    t: int,
    plot_cumulative: bool = True,
    t_start_eval: int | None = None,
    datetime_format: str | None = None,
    time_unit: str = "D",
    time_scaler: float | None = 1,
    sort_purchases: bool | None = True,
    set_index_date: bool | None = False,
    title: str | None = None,
    xlabel: str = "Time Periods",
    ylabel: str = "Purchases",
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Plot actual and expected purchases over time for a fitted ``BetaGeoModel`` or ``ParetoNBDModel``.

    This function is based on the formulation on page 8 of [1]_. Specifically, we take only customers who have made
    their first purchase before the specified number of ``t`` time periods, and run
    ``expected_purchases_new_customer()`` for all remaining time periods. Results can be either cumulative or
    incremental.

    Adapted from the legacy ``lifetimes`` library:
    https://github.com/CamDavidsonPilon/lifetimes/blob/master/lifetimes/plotting.py#L392

    Parameters
    ----------
    model :
        A fitted ``BetaGeoModel`` or ``ParetoNBDModel``.
    purchase_history : ~pandas.DataFrame
        A Pandas DataFrame containing *customer_id_col* and *datetime_col*.
    customer_id_col : string
        Column in the *purchases* DataFrame denoting the *customer_id*.
    datetime_col :  string
        Column in the *purchases* DataFrame denoting datetimes purchase were made.
    t : int
        Number of time units since earliest purchase to include in plot.
    plot_cumulative : bool
        Default: *True*
        Plot cumulative purchases over time. Set to *False* to plot incremental purchases.
    t_start_eval : int, optional
        If testing model on unobserved data, specify number of time units in training data to add an indicator for
        the start of the testing period.
    datetime_format : string, optional
        A string that represents the timestamp format. Useful if Pandas doesn't recognize the provided format.
    time_unit : string, optional
        Time granularity for study.
        Default: 'D' for days. Possible values listed here:
        https://numpy.org/devdocs/reference/arrays.datetime.html#datetime-units
    time_scaler : int, optional
        Default: 1. Scales *recency* & *T* to a different time granularity.
        This is useful for datasets spanning many years, and running predictions in different time scales.
    sort_purchases : bool, optional
        Default: *True*
        If *purchase_history* DataFrame is already sorted in chronological order,
        set to *False* to improve computational efficiency.
    set_index_date : bool, optional
        Set to True to return a dataframe with a datetime index.
    title : str, optional
        Figure title
    xlabel : str, optional
        Figure xlabel
    ylabel : str, optional
        Figure ylabel
    ax : matplotlib.Axes, optional
        A matplotlib Axes instance. Creates new axes instance by default.
    kwargs
        Additional arguments to pass into the pandas.DataFrame.plot command.

    Returns
    -------
    axes: matplotlib.AxesSubplot

    References
    ----------
    .. [1] Fader, Peter S., Bruce G.S. Hardie, and Ka Lok Lee (2005),
       A Note on Implementing the Pareto/NBD Model in MATLAB.
       http://brucehardie.com/notes/008/
    """
    if ax is None:
        ax = plt.subplot(111)

    df_cum_purchases = _expected_cumulative_transactions(
        model=model,
        transactions=purchase_history,
        customer_id_col=customer_id_col,
        datetime_col=datetime_col,
        t=t,
        datetime_format=datetime_format,
        time_unit=time_unit,
        time_scaler=time_scaler,
        sort_transactions=sort_purchases,
        set_index_date=set_index_date,
    )

    if not plot_cumulative:
        df_cum_purchases = df_cum_purchases.diff()
        if title is None:
            title = "Tracking Incremental Transactions"
    else:
        if title is None:
            title = "Tracking Cumulative Transactions"

    # TODO: After utility func supports xarrays, refactor this for matplotlib API.
    ax = df_cum_purchases.plot(ax=ax, title=title, **kwargs)

    if t_start_eval:
        if set_index_date:
            x_vline = df_cum_purchases.index[int(t_start_eval)]
        else:
            x_vline = t_start_eval
        ax.axvline(x=x_vline, color="r", linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_expected_purchases_ppc(
    model,
    ppc: str = "posterior",
    max_purchases: int = 10,
    samples: int = 1000,
    random_seed: int = 45,
    ax: plt.Axes | Sequence[plt.Axes] | None = None,
    plot_type: str = "hist",
    **kwargs,
) -> plt.Axes | tuple[plt.Axes, plt.Axes]:
    """Plot a prior or posterior predictive check for the customer purchase frequency distribution.

    ``ParetoNBDModel``, ``BetaGeoBetaBinomModel``, ``BetaGeoModel`` and ``ModifiedBetaGeoModel`` are supported.

    Adapted from legacy ``lifetimes`` library:
    https://github.com/CamDavidsonPilon/lifetimes/blob/master/lifetimes/plotting.py#L25

    Parameters
    ----------
    model : CLVModel
        A built CLV model is required for prior predictive checks, and a fitted model for posterior predictive checks.
    ppc : string, optional
        Type of predictive check to perform. Options are 'prior' or 'posterior'; defaults to 'posterior'.
    max_purchases : int, optional
        Cutoff for bars of purchase counts to plot. Only used when ``plot_type`` is 'hist'. Default is 10.
    samples : int, optional
        Number of samples to draw for prior predictive checks. This is not used for posterior predictive checks.
    random_seed : int, optional
        Random seed to fix sampling results
    ax : matplotlib.Axes or sequence of matplotlib.Axes, optional
        A matplotlib Axes instance, or a pair of Axes when ``plot_type`` is 'ecdf'. Creates new axes
        instance(s) by default.
    plot_type : string, optional
        Type of plot to produce. Options are 'hist' for a bar chart of estimated vs observed purchase
        counts, or 'ecdf' for an ECDF plot with a 95% simultaneous confidence band and a companion
        difference plot.
        Defaults to 'hist'.
    **kwargs
        Additional arguments to pass into the pandas.DataFrame.plot command when ``plot_type`` is
        'hist'. When ``plot_type`` is 'ecdf', ``num_trials`` and ``confidence_level`` can be passed
        to control the confidence band.

    Returns
    -------
    axes : matplotlib.AxesSubplot, or tuple of two matplotlib.AxesSubplot when ``plot_type`` is 'ecdf'

    Notes
    -----
    The confidence band of the 'ecdf' plot assumes a continuous reference CDF, which does not hold
    for integer purchase counts, so its coverage is only approximately the nominal level. It also
    assumes that CDF is known rather than estimated, so it warns when the predictive samples behind
    it are too few relative to the number of customers.
    """
    if plot_type not in ("hist", "ecdf"):
        raise ValueError("Specify 'hist' or 'ecdf' for 'plot_type' parameter.")

    if plot_type == "ecdf":
        if ax is not None and not (
            np.shape(ax) == (2,) and all(isinstance(panel, plt.Axes) for panel in ax)
        ):
            raise ValueError(
                "Specify a sequence of two matplotlib Axes for 'ax' when 'plot_type' is 'ecdf'."
            )
        if max_purchases != 10:
            warnings.warn(
                "'max_purchases' is ignored when 'plot_type' is 'ecdf'.",
                UserWarning,
                stacklevel=2,
            )
    elif ax is None:
        ax = plt.subplot(111)
    elif not isinstance(ax, plt.Axes):
        raise ValueError(
            "Specify a single matplotlib Axes for 'ax' when 'plot_type' is 'hist'."
        )

    match ppc:
        # TODO: Revisit prior logic after adding PPC support for CLVModels in ModelBuilder
        case "prior":
            prior_idata = pm.sample_prior_predictive(
                draws=samples,
                model=model.model,
                random_seed=random_seed,
            )

            # obs_var must be retrieved from prior_idata if model has not been fit
            obs_freq = prior_idata.observed_data["recency_frequency"].sel(
                obs_var="frequency"
            )
            ppc_freq = prior_idata.prior_predictive["recency_frequency"].sel(
                obs_var="frequency"
            )
            title = "Prior Predictive Check for Customer Frequency"
            title_prefix = "Prior Predictive"
        case "posterior":
            obs_freq = model.idata.observed_data["recency_frequency"].sel(
                obs_var="frequency"
            )
            # n_samples only takes effect for a point-estimate fit, where the posterior holds a
            # single draw; a sampled posterior already provides (chain * draw * customer) samples.
            # Only the ECDF band needs a precise reference CDF, so only it pays for the extra draws.
            ppc_freq = model.distribution_new_customer_recency_frequency(
                random_seed=random_seed,
                n_samples=_REFERENCE_DRAWS if plot_type == "ecdf" else 1,
            ).sel(obs_var="frequency")
            title = "Posterior Predictive Check for Customer Frequency"
            title_prefix = "Posterior Predictive"
        case _:
            raise NameError("Specify 'prior' or 'posterior' for 'ppc' parameter.")

    if plot_type == "ecdf":
        return _plot_expected_purchases_ecdf(
            observed=obs_freq.values.ravel(),
            ppc=ppc_freq.values.ravel(),
            title_prefix=title_prefix,
            random_seed=random_seed,
            ax=ax,
            **kwargs,
        )

    # convert estimated and observed xarrays into dataframes for plotting
    estimated = ppc_freq.to_dataframe().value_counts(normalize=True).sort_index()
    observed = obs_freq.to_dataframe().value_counts(normalize=True).sort_index()

    # PPC histogram plot
    ax = pd.DataFrame(
        {
            "Estimated": estimated.reset_index()["proportion"].head(max_purchases),
            "Observed": observed.reset_index()["proportion"].head(max_purchases),
        },
    ).plot(
        kind="bar",
        ax=ax,
        title=title,
        xlabel="Repeat Purchases",
        ylabel="% of Customer Population",
        rot=0.0,
        **kwargs,
    )
    return ax


def _plot_expected_purchases_ecdf(
    observed: np.ndarray,
    ppc: np.ndarray,
    title_prefix: str,
    random_seed: int,
    ax: plt.Axes | Sequence[plt.Axes] | None = None,
    num_trials: int = 500,
    confidence_level: float = 0.95,
) -> tuple[plt.Axes, plt.Axes]:
    """Plot an ECDF plot and companion difference plot against a simultaneous confidence band.

    The band of Sailynoja et al. (2021) assumes a continuous reference CDF, which does not hold for
    integer purchase counts, so its coverage is only approximately the nominal level.

    It also assumes the reference CDF is known rather than estimated. ``ppc`` therefore has to hold
    many more samples than ``observed``; otherwise both curves carry sampling error of the same
    order and the band comes out too narrow, failing models that fit perfectly well. Warns rather
    than raising when it does not, since the plot is still readable and a short fit is a legitimate
    thing to want to look at.
    """
    if observed.min() == observed.max():
        raise ValueError(
            "An ECDF plot requires more than one distinct observed purchase count."
        )

    if len(ppc) < _MIN_PPC_RATIO * len(observed):
        warnings.warn(
            "The confidence band treats the predictive CDF as known, which needs at least "
            f"{_MIN_PPC_RATIO} predictive samples per observation; got "
            f"{len(ppc) / len(observed):.3g}. The band is too narrow, so a well-fitting model can "
            "fall outside it. Increase 'samples' for a prior check, or fit the model with more "
            "draws for a posterior check.",
            UserWarning,
            stacklevel=3,
        )

    if ax is None:
        _, (ax_ecdf, ax_diff) = plt.subplots(2, 1, layout="constrained")
    else:
        ax_ecdf, ax_diff = ax

    rng = np.random.default_rng(random_seed)
    # purchase counts are integers, so evaluate on the attainable counts
    x = np.arange(observed.min(), observed.max() + 1)
    # reference cdf estimated from the (prior or posterior) predictive samples
    z = compute_ecdf(ppc, x)
    n = len(observed)
    # simulation-based simultaneous confidence band of Sailynoja et al. (2021)
    gamma = simulate_confidence_bands(
        n_draws=n,
        n_chains=1,
        eval_points=z,
        prob=confidence_level,
        n_simulations=num_trials,
        rng=rng,
    )
    lower, upper = get_pointwise_confidence_band(gamma, n, z)

    band_label = f"{confidence_level * 100:.4g}% simultaneous band"
    observed_ecdf = compute_ecdf(observed, x)

    ax_ecdf.step(x, observed_ecdf, where="post", label="Observed")
    ax_ecdf.fill_between(x, lower, upper, step="post", alpha=0.2, label=band_label)
    ax_ecdf.set_title(f"{title_prefix} ECDF Plot")
    ax_ecdf.set_xlabel("Purchases per Customer")
    ax_ecdf.set_ylabel("Proportion of Customers")
    ax_ecdf.legend()

    ax_diff.step(x, observed_ecdf - z, where="post", label="Observed")
    ax_diff.fill_between(
        x, lower - z, upper - z, step="post", alpha=0.2, label=band_label
    )
    ax_diff.set_title(f"{title_prefix} Difference Plot")
    ax_diff.set_xlabel("Purchases per Customer")
    ax_diff.set_ylabel("Deviation from Expected Proportion")
    ax_diff.legend()

    return ax_ecdf, ax_diff


def _force_aspect(ax: plt.Axes, aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)
