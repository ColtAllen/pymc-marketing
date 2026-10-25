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
"""Tests for the shared weighted-observation aggregation helpers (A8, A9)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_marketing.clv._aggregation import (
    UNIQUE_DIM,
    AggregatedObservations,
    aggregate_observations,
    expand_to_customers,
    identity_aggregation,
    survival_counts_to_sbg_data,
)

KEY_COLS = ("cohort", "recency", "T")


@pytest.fixture(scope="module")
def customer_data() -> pd.DataFrame:
    """Two cohorts, T = 4, with duplicated (cohort, recency, T) rows and shuffled ids."""
    rng = np.random.default_rng(7)
    rows = []
    for cohort, n in [("b", 30), ("a", 20)]:
        recency = rng.integers(1, 5, size=n)
        rows.append(pd.DataFrame({"cohort": cohort, "recency": recency, "T": 4}))
    data = pd.concat(rows, ignore_index=True)
    data["customer_id"] = rng.permutation(np.arange(100, 100 + len(data)))
    data["channel"] = rng.integers(0, 2, size=len(data))
    return data[["customer_id", "cohort", "recency", "T", "channel"]]


def _by_customer(agg: AggregatedObservations) -> pd.DataFrame:
    """Customer-level reconstruction of the keys from ``unique`` and ``inverse``."""
    out = agg.unique.iloc[agg.inverse][list(agg.key_cols)].reset_index(drop=True)
    out.insert(0, "customer_id", agg.customer_id)
    return out.sort_values("customer_id").reset_index(drop=True)


class TestAggregateObservations:
    def test_inverse_map_recovers_every_customer(self, customer_data, to_backend):
        agg = aggregate_observations(to_backend(customer_data), KEY_COLS)

        expected = (
            customer_data[["customer_id", *KEY_COLS]]
            .sort_values("customer_id")
            .reset_index(drop=True)
        )
        got = _by_customer(agg)
        pd.testing.assert_frame_equal(
            got.astype(expected.dtypes.to_dict()), expected, check_dtype=False
        )
        assert agg.key_cols == KEY_COLS
        assert agg.inverse.dtype == np.int64
        assert len(agg.customer_id) == len(customer_data)
        assert set(agg.customer_id) == set(customer_data["customer_id"])

    def test_weights_are_row_counts(self, customer_data, to_backend):
        agg = aggregate_observations(to_backend(customer_data), KEY_COLS)

        counts = (
            customer_data.groupby(list(KEY_COLS)).size().rename("weight").reset_index()
        )
        got = agg.unique[[*KEY_COLS, "weight"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            got.astype(counts.dtypes.to_dict()), counts, check_dtype=False
        )
        np.testing.assert_array_equal(agg.weights, counts["weight"].to_numpy())
        assert agg.weights.dtype == np.float64
        assert agg.weights.sum() == len(customer_data)

    def test_unique_rows_sorted_and_order_independent(self, customer_data):
        agg = aggregate_observations(customer_data, KEY_COLS)
        shuffled = customer_data.sample(frac=1.0, random_state=3).reset_index(drop=True)
        agg_shuffled = aggregate_observations(shuffled, KEY_COLS)

        pd.testing.assert_frame_equal(agg.unique, agg_shuffled.unique)
        sort_keys = agg.unique[list(KEY_COLS)]
        assert sort_keys.equals(sort_keys.sort_values(list(KEY_COLS)))
        # eager input keeps the customer order of the input frame
        np.testing.assert_array_equal(agg.customer_id, customer_data["customer_id"])
        np.testing.assert_array_equal(agg_shuffled.customer_id, shuffled["customer_id"])

    def test_covariate_keys_split_rows(self, customer_data):
        agg = aggregate_observations(customer_data, (*KEY_COLS, "channel"))
        plain = aggregate_observations(customer_data, KEY_COLS)

        assert len(agg.unique) > len(plain.unique)
        assert agg.weights.sum() == plain.weights.sum() == len(customer_data)
        assert "channel" in agg.unique.columns

    def test_weight_column_is_summed(self, customer_data, to_backend):
        plain = aggregate_observations(customer_data, KEY_COLS)
        pre = plain.unique.copy()
        pre["customer_id"] = np.arange(len(pre))
        # split one configuration across two rows so the sum matters
        extra = pre.iloc[[0]].copy()
        extra["customer_id"] = len(pre)
        extra["weight"] = 2.5
        pre.loc[0, "weight"] -= 2.5
        pre = pd.concat([pre, extra], ignore_index=True)

        agg = aggregate_observations(to_backend(pre), KEY_COLS, weight_col="weight")

        pd.testing.assert_frame_equal(
            agg.unique.reset_index(drop=True).astype(plain.unique.dtypes.to_dict()),
            plain.unique.reset_index(drop=True),
            check_dtype=False,
        )
        assert agg.total_weight == pytest.approx(len(customer_data))
        assert agg.compression_ratio == pytest.approx(plain.compression_ratio)

    def test_null_key_rejected(self, customer_data):
        bad = customer_data.copy()
        bad.loc[3, "cohort"] = None
        with pytest.raises(ValueError, match="null"):
            aggregate_observations(bad, KEY_COLS)

    @pytest.mark.parametrize("bad_weight", [0.0, -1.0, np.nan])
    def test_non_positive_weight_rejected(self, customer_data, bad_weight):
        bad = customer_data.assign(weight=1.0)
        bad.loc[5, "weight"] = bad_weight
        with pytest.raises(ValueError, match="positive"):
            aggregate_observations(bad, KEY_COLS, weight_col="weight")

    def test_missing_column_rejected(self, customer_data):
        with pytest.raises(KeyError, match="missing_col"):
            aggregate_observations(customer_data, (*KEY_COLS, "missing_col"))

    def test_low_compression_warns(self, customer_data):
        data = customer_data.assign(noise=np.arange(len(customer_data)))
        with pytest.warns(UserWarning, match="compression"):
            agg = aggregate_observations(data, (*KEY_COLS, "noise"))
        assert agg.compression_ratio == 1.0

    def test_compression_warning_threshold(self, customer_data):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            agg = aggregate_observations(customer_data, KEY_COLS)
        assert agg.compression_ratio < 0.5


class TestIdentityAggregation:
    def test_identity(self, customer_data):
        agg = identity_aggregation(customer_data, KEY_COLS)

        np.testing.assert_array_equal(agg.inverse, np.arange(len(customer_data)))
        np.testing.assert_array_equal(agg.customer_id, customer_data["customer_id"])
        np.testing.assert_array_equal(agg.weights, 1.0)
        assert agg.compression_ratio == 1.0
        pd.testing.assert_frame_equal(
            agg.unique[list(KEY_COLS)].reset_index(drop=True),
            customer_data[list(KEY_COLS)].reset_index(drop=True),
        )


class TestExpandToCustomers:
    def test_expand_replaces_unique_dim(self, customer_data):
        agg = aggregate_observations(customer_data, KEY_COLS)
        n_unique = len(agg.unique)
        da = xr.DataArray(
            np.arange(2 * 3 * n_unique, dtype="float64").reshape(2, 3, n_unique),
            dims=("chain", "draw", UNIQUE_DIM),
            coords={
                UNIQUE_DIM: np.arange(n_unique),
                "cohort": (UNIQUE_DIM, agg.unique["cohort"].to_numpy()),
            },
        )

        out = expand_to_customers(da, agg)

        assert out.dims == ("chain", "draw", "customer_id")
        np.testing.assert_array_equal(out["customer_id"], agg.customer_id)
        np.testing.assert_array_equal(
            out.values, da.values[:, :, agg.inverse]
        )  # vectorized gather
        # non-dim coords ride along, so cohort selection keeps working
        np.testing.assert_array_equal(
            out["cohort"].values, agg.unique["cohort"].to_numpy()[agg.inverse]
        )
        assert out.swap_dims({"customer_id": "cohort"}).sel(cohort="a").sizes[
            "cohort"
        ] == int((customer_data["cohort"] == "a").sum())

    def test_expand_memory_guard(self, customer_data):
        agg = aggregate_observations(customer_data, KEY_COLS)
        da = xr.DataArray(
            np.zeros((4, 1000, len(agg.unique))),
            dims=("chain", "draw", UNIQUE_DIM),
        )
        with pytest.raises(MemoryError, match="max_bytes"):
            expand_to_customers(da, agg, max_bytes=1024)


# Published Fader & Hardie (2007) survival table, periods 0..7 (A9).
FH_HIGHEND = [1000, 869, 743, 653, 593, 551, 517, 491]
FH_REGULAR = [1000, 631, 468, 382, 326, 289, 262, 241]


@pytest.fixture(scope="module")
def survival_table() -> pd.DataFrame:
    return pd.concat(
        [
            pd.DataFrame(
                {"cohort": name, "period": range(len(alive)), "n_alive": alive}
            )
            for name, alive in [("highend", FH_HIGHEND), ("regular", FH_REGULAR)]
        ],
        ignore_index=True,
    )


class TestSurvivalCountsConverter:
    """A9: the published Fader & Hardie (2007) survival table."""

    def test_matches_customer_level_fixture(self, survival_table, to_backend):
        converted = survival_counts_to_sbg_data(to_backend(survival_table))

        customer_level = pd.read_csv("data/sbg_cohorts.csv")
        expected = aggregate_observations(customer_level, KEY_COLS).unique
        got = aggregate_observations(converted, KEY_COLS, weight_col="weight").unique

        pd.testing.assert_frame_equal(
            got.reset_index(drop=True).astype(expected.dtypes.to_dict()),
            expected.reset_index(drop=True),
            check_dtype=False,
        )
        assert converted["customer_id"].is_unique
        assert (converted["weight"] > 0).all()
        assert set(converted.columns) >= {
            "customer_id",
            "cohort",
            "recency",
            "T",
            "weight",
        }

    def test_boundary_convention(self, survival_table):
        """Churn at period t -> recency t; survivors of the last observed period -> recency T."""
        converted = survival_counts_to_sbg_data(survival_table)
        highend = converted.query("cohort == 'highend'").set_index("recency")["weight"]

        assert highend[1] == 1000 - 869
        assert highend[7] == 517 - 491
        assert highend[8] == 491
        assert (converted["T"] == 8).all()

    def test_zero_weight_rows_dropped_and_increasing_alive_rejected(self):
        table = pd.DataFrame(
            {"cohort": "x", "period": [0, 1, 2, 3], "n_alive": [10, 10, 4, 4]}
        )
        converted = survival_counts_to_sbg_data(table)
        assert sorted(converted["recency"]) == [2, 4]
        assert converted.set_index("recency")["weight"].to_dict() == {2: 6, 4: 4}

        bad = table.assign(n_alive=[10, 11, 4, 4])
        with pytest.raises(ValueError, match="non-increasing"):
            survival_counts_to_sbg_data(bad)

    def test_periods_must_be_contiguous_from_zero(self):
        table = pd.DataFrame(
            {"cohort": "x", "period": [1, 2, 3], "n_alive": [10, 8, 4]}
        )
        with pytest.raises(ValueError, match="period"):
            survival_counts_to_sbg_data(table)
