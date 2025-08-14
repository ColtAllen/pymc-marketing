#   Copyright 2022 - 2025 The PyMC Labs Developers
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

import arviz as az
import pandas as pd
from pydantic import ConfigDict, InstanceOf, validate_call

from pymc_marketing.model_builder import DifferentModelError, ModelBuilder
from pymc_marketing.model_config import ModelConfig, parse_model_config


class CLVModel(ModelBuilder):
    """CLV Model base class."""

    _model_type = "CLVModel"

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        data: pd.DataFrame,
        *,
        model_config: InstanceOf[ModelConfig] | None = None,
        sampler_config: dict | None = None,
        non_distributions: list[str] | None = None,
    ):
        self.data = data
        model_config = model_config or {}

        deprecated_keys = [key for key in model_config if key.endswith("_prior")]
        for key in deprecated_keys:
            new_key = key.replace("_prior", "")
            warnings.warn(
                f"The key '{key}' in model_config is deprecated and will be removed in future versions."
                f"Use '{new_key}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

            model_config[new_key] = model_config.pop(key)

        super().__init__(model_config, sampler_config)

        # Parse model config after merging with defaults
        self.model_config = parse_model_config(
            self.model_config,
            non_distributions=non_distributions,
        )

        self.data = data

    @staticmethod
    def _validate_cols(
        data: pd.DataFrame,
        required_cols: Sequence[str],
        must_be_unique: Sequence[str] = (),
        must_be_homogenous: Sequence[str] = (),
    ):
        missing = set(required_cols).difference(data.columns)
        if missing:
            raise ValueError(
                "The following required columns are missing from the "
                f"input data: {sorted(list(missing))}"
            )

        n = data.shape[0]

        for col in required_cols:
            if col in must_be_unique:
                if data[col].nunique() != n:
                    raise ValueError(f"Column {col} has duplicate entries")
            if col in must_be_homogenous:
                if data[col].nunique() != 1:
                    raise ValueError(f"Column {col} has non-homogeneous entries")

    def __repr__(self) -> str:
        """Representation of the model."""
        if not hasattr(self, "model"):
            return self._model_type
        else:
            return f"{self._model_type}\n{self.model.str_repr()}"

    @classmethod
    def idata_to_init_kwargs(cls, idata: az.InferenceData) -> dict:
        """Create the initialization kwargs from an InferenceData object."""
        kwargs = cls.attrs_to_init_kwargs(idata.attrs)
        kwargs["data"] = idata.fit_data.to_dataframe()

        return kwargs

    @classmethod
    def build_from_idata(cls, idata: az.InferenceData) -> None:
        """Build the model from the InferenceData object."""
        kwargs = cls.idata_to_init_kwargs(idata)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
            )
            model = cls(**kwargs)

        model.idata = idata
        model._rename_posterior_variables()

        model.build_model()  # type: ignore
        if model.id != idata.attrs["id"]:
            msg = (
                "The model id in the InferenceData does not match the model id. "
                "There was no error loading the inference data, but the model may "
                "be different. "
                "Investigate if the model structure or configuration has changed."
            )
            raise DifferentModelError(msg)
        return model

    # TODO: Remove in 2026Q1?
    def _rename_posterior_variables(self):
        """Rename variables in the posterior group to remove the _prior suffix.

        This is used to support the old model configuration format, which used
        to include a _prior suffix for each parameter.
        """
        prior_vars = [
            var for var in self.idata.posterior.data_vars if var.endswith("_prior")
        ]
        rename_dict = {var: var.replace("_prior", "") for var in prior_vars}
        self.idata.posterior = self.idata.posterior.rename(rename_dict)
        return self.idata.posterior

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
