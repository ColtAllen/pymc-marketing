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
"""CLV Model base class."""

import warnings
from collections.abc import Sequence
from typing import Any

import arviz as az
import narwhals.stable.v2 as nw
import xarray as xr
from narwhals.stable.v2.typing import IntoFrame
from pydantic import ConfigDict, InstanceOf, validate_call

from pymc_marketing.model_builder import (
    DifferentModelError,
    ModelBuilder,
    _to_pandas,
)
from pymc_marketing.model_config import (
    ModelConfig,
    _reject_removed_parameters,
    parse_model_config,
)
from pymc_marketing.names import Attr
from pymc_marketing.names.clv import Col


class CLVModel(ModelBuilder):
    """CLV Model base class."""

    _model_type = "CLVModel"

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        *,
        model_config: InstanceOf[ModelConfig] | None = None,
        sampler_config: dict | None = None,
        **kwargs: Any,
    ):
        # ``non_distributions`` used to be accepted here; ``**kwargs`` keeps it
        # failing with a migration hint rather than a bare TypeError.
        _reject_removed_parameters(**kwargs)

        model_config = model_config or {}

        super().__init__(model_config, sampler_config)

        # Parse model config after merging with defaults
        self.model_config = parse_model_config(self.model_config)

    @staticmethod
    def _any_row(data: IntoFrame, **checks: nw.Expr) -> dict[str, Any]:
        """Evaluate aggregate expressions over ``data`` in a single query.

        A lazy table is scanned once, where it lives, and only the one-row result
        is collected.
        """
        frame = nw.from_native(data).lazy()
        result = frame.select(*(expr.alias(name) for name, expr in checks.items()))
        return result.collect().rows(named=True)[0]

    @staticmethod
    def _validate_cols(
        data: IntoFrame,
        required_cols: Sequence[str],
        must_be_unique: Sequence[str] = (),
        must_be_homogenous: Sequence[str] = (),
    ):
        missing = set(required_cols).difference(
            nw.from_native(data).collect_schema().names()
        )
        if missing:
            raise ValueError(
                "The following required columns are missing from the "
                f"input data: {sorted(list(missing))}"
            )

        counted = [
            col
            for col in required_cols
            if col in must_be_unique or col in must_be_homogenous
        ]
        if not counted:
            return
        # Suffixed so no count can shadow a data column.
        counts = CLVModel._any_row(
            data,
            _n_rows=nw.len(),
            **{f"{col}_n_unique": nw.col(col).n_unique() for col in counted},
        )

        for col in counted:
            n_unique = counts[f"{col}_n_unique"]
            if col in must_be_unique and n_unique != counts["_n_rows"]:
                raise ValueError(f"Column {col} has duplicate entries")
            if col in must_be_homogenous and n_unique != 1:
                raise ValueError(f"Column {col} has non-homogeneous entries")

    def _validate_frequency(self, data: IntoFrame) -> None:
        """Validate frequency column values shared across BTYD and Gamma-Gamma models."""
        frequency = nw.col(Col.FREQUENCY)
        found = self._any_row(
            data,
            negative=(frequency < 0).any(),
            # No modulo: pandas columns with a PyArrow dtype do not implement it.
            fractional=(frequency != frequency.round()).any(),
        )
        if found["negative"]:
            raise ValueError("Column frequency has negative values")
        if found["fractional"]:
            raise ValueError("frequency column must contain only integer values")

    def _validate_rfm_data(self, data: IntoFrame) -> None:
        """Validate frequency, recency, and T columns shared by BTYD transaction models."""
        self._validate_cols(data, required_cols=[Col.FREQUENCY, Col.RECENCY, Col.T])
        self._validate_frequency(data)

        frequency, recency, T = (
            nw.col(Col.FREQUENCY),
            nw.col(Col.RECENCY),
            nw.col(Col.T),
        )
        found = self._any_row(
            data,
            negative_recency=(recency < 0).any(),
            negative_T=(T < 0).any(),
            recency_without_purchase=((frequency == 0) & (recency > 0)).any(),
            recency_after_T=(recency > T).any(),
            zero_T=(T == 0).any(),
        )
        if found["negative_recency"]:
            raise ValueError("Column recency has negative values")
        if found["negative_T"]:
            raise ValueError("Column T has negative values")
        if found["recency_without_purchase"]:
            raise ValueError("recency cannot be greater than 0 if frequency is 0")
        if found["recency_after_T"]:
            raise ValueError("recency cannot be greater than T")

        if found["zero_T"]:
            warnings.warn(
                "T=0 is uninformative for model fitting. This occurs when a customer "
                "makes a purchase in the final observed time period.",
                UserWarning,
                stacklevel=2,
            )

    def _validate_data(self, data: IntoFrame) -> None:
        """Validate model input data. Child classes should override and call super()."""
        self._validate_rfm_data(data)

    def __repr__(self) -> str:
        """Representation of the model."""
        if not hasattr(self, "model"):
            return self._model_type
        else:
            return f"{self._model_type}\n{self.model.str_repr()}"

    def _prepare_fit(self, data: IntoFrame | None = None) -> None:
        """Build the model on first fit, and reject a second fit on different data.

        CLV models bake the training data into the model graph, so refitting a built
        model with new data would silently sample the wrong likelihood.
        """
        if not hasattr(self, "model"):
            if data is None and getattr(self, "data", None) is None:
                raise ValueError(
                    "data is required to build the model. Pass it to `fit(data)` "
                    "or call `build_model(data)` first."
                )
            self.build_model(data)  # type: ignore[call-arg]
        elif (
            data is not None
            and data is not self.data  # type: ignore[attr-defined]
            # Compared as pandas so the same values in another library still match.
            and not _to_pandas(self.data).equals(_to_pandas(data))  # type: ignore[attr-defined]
        ):
            raise ValueError(
                "The model was built with different data. "
                "Create a new model instance to fit new data."
            )

    @classmethod
    def build_from_idata(cls, idata: xr.DataTree) -> None:
        """Build the model from the DataTree object."""
        kwargs = cls.idata_to_init_kwargs(idata)
        model = cls(**kwargs)

        model.idata = idata
        model.data = idata.fit_data.dataset.to_dataframe()

        model.build_model(model.data)  # type: ignore
        if model.id != idata.attrs[Attr.ID]:
            msg = (
                "The model id in the DataTree does not match the model id. "
                "There was no error loading the inference data, but the model may "
                "be different. "
                "Investigate if the model structure or configuration has changed."
            )
            raise DifferentModelError(msg)
        return model

    def thin_fit_result(self, keep_every: int):
        """Return a copy of the model with a thinned fit result.

        This is useful when computing summary statistics that may require too much memory per posterior draw.

        Examples
        --------

        .. code-block:: python

            fitted_gg = ...
            fitted bg = ...

            fitted_gg_thinned = fitted_gg.thin_fit_result(keep_every=10)
            fitted_bg_thinned = fitted_bg.thin_fit_result(keep_every=10)

            clv_thinned = fitted_gg_thinned.expected_customer_lifetime_value(
                transaction_model=fitted_bg_thinned,
                customer_id=t.index,
                frequency=t["frequency"],
                recency=t["recency"],
                T=t["T"],
                mean_transaction_value=t["monetary_value"],
            )

        """
        self.fit_result  # noqa: B018 (Raise Error if fit didn't happen yet)
        assert self.idata is not None  # noqa: S101
        new_idata = self.idata.isel(draw=slice(None, None, keep_every)).copy()
        return self.build_from_idata(new_idata)

    @property
    def default_sampler_config(self) -> dict:
        """Default sampler configuration."""
        return {}

    @property
    def _serializable_model_config(self) -> dict:
        return self.model_config

    def fit_summary(self, **kwargs):
        """Compute the summary of the fit result."""
        res = self.fit_result
        # Map fitting only gives one value, so we return it. We use arviz
        # just to get it nicely into a DataFrame
        if res.chain.size == 1 and res.draw.size == 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = az.summary(self.fit_result, **kwargs, kind="stats")
            return res["mean"].rename("value")
        else:
            return az.summary(self.fit_result, **kwargs)
