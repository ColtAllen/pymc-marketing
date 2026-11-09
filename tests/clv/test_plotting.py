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
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pytensor.tensor import TensorVariable

from pymc_marketing.clv import (
    plot_cohorts,
    plot_customer_exposure,
    plot_expected_purchases_over_time,
    plot_expected_purchases_ppc,
    plot_frequency_recency_matrix,
    plot_probability_alive_matrix,
)
from pymc_marketing.clv.plotting import _plot_expected_purchases_ecdf


class MockModel:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self._model_type = None

    def _mock_posterior(self, data: pd.DataFrame) -> xr.DataArray:
        n_customers = len(data)
        n_chains = 4
        n_draws = 10
        chains = np.arange(n_chains)
        draws = np.arange(n_draws)
        return xr.DataArray(
            data=np.ones((n_customers, n_chains, n_draws)),
            coords={"customer_id": data["customer_id"], "chain": chains, "draw": draws},
            dims=["customer_id", "chain", "draw"],
        )

    def expected_probability_alive(self, data: np.ndarray | pd.Series):
        return self._mock_posterior(data)

    def expected_purchases(
        self,
        data: pd.DataFrame,
        *,
        future_t: np.ndarray | pd.Series | TensorVariable,
    ):
        return self._mock_posterior(data)

    def expected_purchases_new_customer(
        self,
        data: pd.DataFrame,
    ):
        return self._mock_posterior(data)


@pytest.fixture
def mock_model(test_summary_data) -> MockModel:
    return MockModel(test_summary_data)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"colors": ["blue", "red"]},
        {"labels": ["Customer Recency", "Customer T"]},
    ],
)
def test_plot_customer_exposure(test_summary_data, kwargs) -> None:
    ax: plt.Axes = plot_customer_exposure(test_summary_data, **kwargs)

    assert isinstance(ax, plt.Axes)


def test_plot_customer_exposure_with_ax(test_summary_data) -> None:
    ax = plt.subplot()
    plot_customer_exposure(test_summary_data, ax=ax)

    assert ax.get_title() == "Customer Exposure"
    assert ax.get_xlabel() == "Time since first purchase"
    assert ax.get_ylabel() == "Customer"


@pytest.mark.parametrize(
    "kwargs",
    [
        # More labels or colors
        {"labels": [0, 1, 2]},
        {"colors": ["blue", "red", "green"]},
        # Negative Values
        {"padding": -1},
        {"linewidth": -1},
        {"size": -1},
    ],
)
def test_plot_customer_exposure_invalid_args(test_summary_data, kwargs) -> None:
    with pytest.raises(ValueError):
        plot_customer_exposure(test_summary_data, **kwargs)


def test_plot_frequency_recency_matrix(mock_model) -> None:
    ax: plt.Axes = plot_frequency_recency_matrix(mock_model)

    assert isinstance(ax, plt.Axes)


def test_plot_frequency_recency_matrix_bounds(mock_model) -> None:
    max_recency = 10
    max_frequency = 10
    ax: plt.Axes = plot_frequency_recency_matrix(
        mock_model, max_recency=max_recency, max_frequency=max_frequency
    )

    assert isinstance(ax, plt.Axes)


def test_plot_frequency_recency_matrix_with_ax(mock_model) -> None:
    ax = plt.subplot()
    plot_frequency_recency_matrix(mock_model, ax=ax)

    assert ax.get_xlabel() == "Customer's Historical Frequency"
    assert ax.get_ylabel() == "Customer's Recency"


def test_plot_probability_alive_matrix(mock_model) -> None:
    ax: plt.Axes = plot_probability_alive_matrix(mock_model)

    assert isinstance(ax, plt.Axes)


def test_plot_probability_alive_matrix_bounds(mock_model) -> None:
    max_recency = 10
    max_frequency = 10
    ax: plt.Axes = plot_probability_alive_matrix(
        mock_model, max_recency=max_recency, max_frequency=max_frequency
    )

    assert isinstance(ax, plt.Axes)


def test_plot_probability_alive_matrix_with_ax(mock_model) -> None:
    ax = plt.subplot()
    plot_probability_alive_matrix(mock_model, ax=ax)

    assert ax.get_xlabel() == "Customer's Historical Frequency"
    assert ax.get_ylabel() == "Customer's Recency"


@pytest.mark.parametrize(
    "plot_cumulative, set_index_date, subplot",
    [(True, False, None), (False, True, plt.subplot())],
)
def test_plot_expected_purchases_over_time(
    mock_model, cdnow_trans, plot_cumulative, set_index_date, subplot
) -> None:
    ax = plot_expected_purchases_over_time(
        model=mock_model,
        purchase_history=cdnow_trans,
        customer_id_col="id",
        datetime_col="date",
        datetime_format="%Y%m%d",
        time_unit="D",
        plot_cumulative=plot_cumulative,
        set_index_date=set_index_date,
        t=10,
        t_start_eval=8,
        ax=subplot,
    )

    assert isinstance(ax, plt.Axes)

    # clear any existing pyplot figures
    plt.clf()


def test_plot_expected_purchases_ppc_exceptions(fitted_model):
    with pytest.raises(
        NameError, match=r"Specify 'prior' or 'posterior' for 'ppc' parameter."
    ):
        plot_expected_purchases_ppc(fitted_model, ppc="ppc")

    with pytest.raises(
        ValueError, match=r"Specify 'hist' or 'ecdf' for 'plot_type' parameter."
    ):
        plot_expected_purchases_ppc(fitted_model, plot_type="bar")

    with pytest.raises(ValueError, match=r"sequence of two matplotlib Axes"):
        plot_expected_purchases_ppc(
            fitted_model, plot_type="ecdf", ax=plt.subplots(1, 1)[1]
        )

    with pytest.raises(ValueError, match=r"sequence of two matplotlib Axes"):
        plot_expected_purchases_ppc(
            fitted_model, plot_type="ecdf", ax=plt.subplots(2, 2)[1]
        )

    # anything that is not a two-element container of Axes, sized or not
    for bad_ax in (42, plt.figure(), set(plt.subplots(2, 1)[1]), [1, 2]):
        with pytest.raises(ValueError, match=r"sequence of two matplotlib Axes"):
            plot_expected_purchases_ppc(fitted_model, plot_type="ecdf", ax=bad_ax)

    with pytest.raises(ValueError, match=r"single matplotlib Axes"):
        plot_expected_purchases_ppc(fitted_model, ax=plt.subplots(2, 1)[1])

    with pytest.raises(ValueError, match=r"distinct observed purchase count"):
        _plot_expected_purchases_ecdf(
            observed=np.ones(10),
            ppc=np.ones(10),
            title_prefix="Posterior Predictive",
            random_seed=45,
        )

    plt.close("all")


def test_plot_expected_purchases_ecdf_warns_on_small_ppc():
    with pytest.warns(UserWarning, match=r"predictive samples per observation"):
        _plot_expected_purchases_ecdf(
            observed=np.arange(10),
            ppc=np.arange(10),
            title_prefix="Posterior Predictive",
            random_seed=45,
        )

    plt.close("all")


@pytest.mark.parametrize(
    "ppc, max_purchases, samples, use_ax",
    [("prior", 10, 100, False), ("posterior", 20, 50, True)],
)
def test_plot_expected_purchases_ppc(fitted_model, ppc, max_purchases, samples, use_ax):
    subplot = plt.subplots(1, 1)[1] if use_ax else None
    ax = plot_expected_purchases_ppc(
        model=fitted_model,
        ppc=ppc,
        max_purchases=max_purchases,
        samples=samples,
        ax=subplot,
    )

    # the default plot_type is 'hist', which returns a single Axes
    assert isinstance(ax, plt.Axes)
    if use_ax:
        assert ax is subplot

    # clear any existing pyplot figures
    plt.close("all")


@pytest.mark.parametrize("ppc", ["prior", "posterior"])
@pytest.mark.parametrize("use_ax", [False, True])
def test_plot_expected_purchases_ppc_ecdf(fitted_model, ppc, use_ax):
    subplots = plt.subplots(2, 1)[1] if use_ax else None
    ax_ecdf, ax_diff = plot_expected_purchases_ppc(
        model=fitted_model,
        ppc=ppc,
        plot_type="ecdf",
        samples=100,
        ax=subplots,
        confidence_level=0.9,
        num_trials=50,
    )

    assert isinstance(ax_ecdf, plt.Axes)
    assert isinstance(ax_diff, plt.Axes)
    if use_ax:
        assert (ax_ecdf, ax_diff) == tuple(subplots)
    else:
        assert ax_ecdf.figure is ax_diff.figure

    # 'confidence_level' reached the band through **kwargs, rather than being silently swallowed
    assert "90% simultaneous band" in [
        text.get_text() for text in ax_ecdf.get_legend().get_texts()
    ]

    observed = (
        fitted_model.idata.observed_data["recency_frequency"]
        .sel(obs_var="frequency")
        .values.ravel()
    )
    grid = np.arange(observed.min(), observed.max() + 1)

    # the ECDF panel plots the observed ECDF, on one point per attainable purchase count
    assert np.array_equal(ax_ecdf.lines[0].get_xdata(), grid)
    np.testing.assert_allclose(
        ax_ecdf.lines[0].get_ydata(), [(observed <= point).mean() for point in grid]
    )

    # the band spans the same grid as the curve it is drawn around
    band_x = ax_ecdf.collections[0].get_paths()[0].vertices[:, 0]
    assert (band_x.min(), band_x.max()) == (grid.min(), grid.max())

    # the difference panel is the same ECDF minus a reference CDF, on the same grid
    assert np.array_equal(ax_diff.lines[0].get_xdata(), grid)
    reference = ax_ecdf.lines[0].get_ydata() - ax_diff.lines[0].get_ydata()
    assert np.all(np.diff(reference) >= 0)
    assert reference.min() >= 0.0
    assert reference.max() <= 1.0

    # clear any existing pyplot figures
    plt.close("all")


def test_plot_expected_purchases_ppc_ecdf_map_fit(map_fitted_bg):
    """A point-estimate fit has a single posterior draw, so the reference CDF has to come from
    extra predictive draws per customer rather than from the posterior."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ax_ecdf, ax_diff = plot_expected_purchases_ppc(
            model=map_fitted_bg,
            plot_type="ecdf",
        )

    # the reference CDF must not fall back to one predictive sample per customer
    assert not [w for w in caught if "predictive samples" in str(w.message)]
    assert isinstance(ax_ecdf, plt.Axes)
    assert isinstance(ax_diff, plt.Axes)

    plt.close("all")


def test_plot_expected_purchases_ppc_ecdf_ignores_max_purchases(fitted_model):
    with pytest.warns(UserWarning, match=r"'max_purchases' is ignored"):
        plot_expected_purchases_ppc(
            model=fitted_model,
            plot_type="ecdf",
            max_purchases=20,
        )

    plt.close("all")


@pytest.fixture(scope="module")
def sbg_cohort_data() -> pd.DataFrame:
    """Contractual sBG data whose cohort labels are segments, not dates."""
    return pd.read_csv("data/sbg_cohorts.csv")


@pytest.fixture
def dated_cohort_data() -> pd.DataFrame:
    """Summary data with staggered monthly cohorts, as produced by an sBG study."""
    rng = np.random.default_rng(42)
    frames = []
    for offset, cohort in enumerate(["2025-01", "2025-02", "2025-03"]):
        T = 6 - offset
        frames.append(
            pd.DataFrame(
                {
                    "customer_id": range(offset * 100, (offset + 1) * 100),
                    "cohort": cohort,
                    "recency": rng.integers(1, T + 1, 100),
                    "T": T,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_plot_cohorts_retention_values_are_exact():
    # last activity at ages 0, 1, 1 and 3 within a single four-period cohort
    data = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "cohort": ["A"] * 4,
            "recency": [0, 1, 1, 3],
            "T": [4] * 4,
        }
    )

    ax = plot_cohorts(data)

    # retained at age a == customers whose last activity falls strictly after a
    assert [text.get_text() for text in ax.texts] == ["3", "1", "1", "0"]

    plt.close("all")


@pytest.mark.parametrize("show_pct", [False, True])
def test_plot_cohorts_summary_data(sbg_cohort_data, show_pct) -> None:
    ax = plot_cohorts(sbg_cohort_data, show_pct=show_pct)

    assert isinstance(ax, plt.Axes)
    # 'highend' and 'regular' are not dates, so the calendar axis is unavailable
    assert ax.get_xlabel() == "Cohort Age"
    assert ax.get_ylabel() == "Cohort"
    expected_title = (
        "Cohort Retention Rate (%)" if show_pct else "Cohort Customer Counts"
    )
    assert ax.get_title() == expected_title

    plt.close("all")


def test_plot_cohorts_sbg_retention_starts_at_full_cohort(sbg_cohort_data) -> None:
    """sBG *recency* is one-indexed, so no customer has churned by age 0."""
    ax = plot_cohorts(sbg_cohort_data, show_pct=True)

    first_column = [text.get_text() for text in ax.texts][::8]

    assert first_column == ["100", "100"]

    plt.close("all")


def test_plot_cohorts_derives_cohorts_from_T(test_summary_data) -> None:
    ax = plot_cohorts(test_summary_data)

    assert isinstance(ax, plt.Axes)
    # a larger T means an earlier acquisition, so cohorts are integer offsets
    assert ax.get_xlabel() == "Cohort Age"
    assert len(ax.get_yticklabels()) > 1

    plt.close("all")


def test_plot_cohorts_warns_on_homogeneous_T() -> None:
    data = pd.DataFrame(
        {"customer_id": [1, 2, 3], "recency": [0, 1, 2], "T": [3, 3, 3]}
    )

    with pytest.warns(UserWarning, match=r"same 'T'"):
        ax = plot_cohorts(data)

    assert len(ax.get_yticklabels()) == 1

    plt.close("all")


@pytest.mark.parametrize("x_axis", ["auto", "calendar", "cohort_age"])
def test_plot_cohorts_dated_cohorts(dated_cohort_data, x_axis) -> None:
    ax = plot_cohorts(dated_cohort_data, time_unit="M", x_axis=x_axis, show_pct=True)

    columns = [text.get_text() for text in ax.get_xticklabels()]

    if x_axis == "cohort_age":
        assert ax.get_xlabel() == "Cohort Age"
        assert columns == ["0", "1", "2", "3", "4", "5"]
    else:
        # cohorts are staggered onto a shared calendar, spanning 2025-01 to 2025-06
        assert ax.get_xlabel() == "Time Period"
        assert columns == [f"2025-0{month}" for month in range(1, 7)]

    plt.close("all")


def test_plot_cohorts_warns_when_time_unit_misses_cohort_spacing(
    dated_cohort_data,
) -> None:
    with pytest.warns(UserWarning, match=r"'time_unit' likely does not match"):
        plot_cohorts(dated_cohort_data, time_unit="D")

    plt.close("all")


@pytest.mark.parametrize("time_unit", ["W", "M"])
def test_plot_cohorts_transaction_log(cdnow_trans, time_unit) -> None:
    ax = plot_cohorts(
        cdnow_trans,
        customer_id_col="id",
        datetime_col="date",
        datetime_format="%Y%m%d",
        time_unit=time_unit,
    )

    cohorts = [text.get_text() for text in ax.get_yticklabels()]

    assert isinstance(ax, plt.Axes)
    assert ax.get_xlabel() == "Time Period"
    # CDNOW customers were all acquired in the first quarter of 1997
    assert all(label.startswith(("1996-12", "1997-0")) for label in cohorts)
    if time_unit == "M":
        assert cohorts == ["1997-01", "1997-02", "1997-03"]
    else:
        assert len(cohorts) > 3

    plt.close("all")


def test_plot_cohorts_retention_is_non_increasing(cdnow_trans) -> None:
    ax = plot_cohorts(
        cdnow_trans,
        customer_id_col="id",
        datetime_col="date",
        datetime_format="%Y%m%d",
        time_unit="M",
        show_pct=True,
    )

    values = ax.collections[0].get_array().reshape(3, -1)

    for cohort in np.ma.filled(values, np.nan):
        observed = cohort[~np.isnan(cohort)]
        assert np.all(np.diff(observed) <= 0)

    plt.close("all")


def test_plot_cohorts_annotates_small_matrices_only(cdnow_trans, sbg_cohort_data):
    small = plot_cohorts(sbg_cohort_data)
    assert len(small.texts) == 16
    plt.close("all")

    large = plot_cohorts(
        cdnow_trans,
        customer_id_col="id",
        datetime_col="date",
        datetime_format="%Y%m%d",
        time_unit="W",
    )
    assert len(large.texts) == 0
    plt.close("all")

    forced = plot_cohorts(
        cdnow_trans,
        customer_id_col="id",
        datetime_col="date",
        datetime_format="%Y%m%d",
        time_unit="W",
        annot=True,
    )
    assert len(forced.texts) > 0
    plt.close("all")


def test_plot_cohorts_with_ax(sbg_cohort_data) -> None:
    ax = plt.subplot()

    assert plot_cohorts(sbg_cohort_data, ax=ax, title="Custom", ylabel="Segment") is ax
    assert ax.get_title() == "Custom"
    assert ax.get_ylabel() == "Segment"

    plt.close("all")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"x_axis": "quarterly"}, r"'x_axis' must be one of"),
        ({"x_axis": "calendar"}, r"could not be interpreted as dates"),
        ({"cohort_col": "segment"}, r"not a column of the provided data"),
    ],
)
def test_plot_cohorts_invalid_arguments(sbg_cohort_data, kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        plot_cohorts(sbg_cohort_data, **kwargs)


def test_plot_cohorts_requires_summary_columns() -> None:
    with pytest.raises(ValueError, match=r"missing \['recency', 'T'\]"):
        plot_cohorts(pd.DataFrame({"customer_id": [1, 2]}))
