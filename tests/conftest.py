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
from collections.abc import Callable
from typing import Any

import narwhals.stable.v2 as nw
import pandas as pd
import pytest
from pymc.testing import mock_sample_setup_and_teardown


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="also run slow tests"
    )
    parser.addoption(
        "--only-slow", action="store_true", default=False, help="only run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line(
        "markers", "requires_polars: mark test as requiring polars library"
    )
    config.addinivalue_line(
        "markers", "requires_pyspark: mark test as requiring pyspark library"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        # --run-slow given in cli: do not need to skip any tests
        return

    elif config.getoption("--only-slow"):
        # --only-slow given in cli: need to skip non-slow tests
        skip_fast = pytest.mark.skip(reason="Fast test")
        for item in items:
            if "slow" not in item.keywords:
                item.add_marker(skip_fast)

    else:
        # Default: skip slow tests
        skip_slow = pytest.mark.skip(reason="Slow test, use --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


mock_pymc_sample = pytest.fixture(scope="module")(mock_sample_setup_and_teardown)


# Dataframe libraries the Narwhals-based code paths are tested against.
BACKENDS = ["pandas", "pandas_pyarrow", "polars", "polars_lazy", "pyarrow", "duckdb"]


@pytest.fixture(params=BACKENDS)
def to_backend(request) -> Callable[[pd.DataFrame], Any]:
    """Convert a pandas frame into the native frame of the parametrised backend."""
    name = request.param

    def convert(df: pd.DataFrame):
        if name == "pandas":
            return df
        if name == "pandas_pyarrow":
            # What pd.read_parquet(..., dtype_backend="pyarrow") returns.
            pytest.importorskip("pyarrow")
            return df.convert_dtypes(dtype_backend="pyarrow")
        if name in ("polars", "polars_lazy"):
            pl = pytest.importorskip("polars")
            out = pl.from_pandas(df)
            return out.lazy() if name == "polars_lazy" else out
        pa = pytest.importorskip("pyarrow")
        table = pa.Table.from_pandas(df, preserve_index=False)
        if name == "pyarrow":
            return table
        duckdb = pytest.importorskip("duckdb")
        return duckdb.from_arrow(table)

    convert.backend_name = name  # type: ignore[attr-defined]
    return convert


def collect_sorted(result, by: str | list[str] = "customer_id") -> pd.DataFrame:
    """Materialise any native frame into a pandas frame sorted by ``by``."""
    frame = nw.from_native(result)
    if isinstance(frame, nw.LazyFrame):
        frame = frame.collect()
    result = frame.sort(by).to_pandas().reset_index(drop=True)
    if any(isinstance(dtype, pd.ArrowDtype) for dtype in result.dtypes):
        # Compare values, not the PyArrow dtypes of the pandas_pyarrow backend.
        result = frame.sort(by).to_arrow().replace_schema_metadata(None).to_pandas()
    return result
