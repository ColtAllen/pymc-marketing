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
import arviz_stats as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from pymc_extras.prior import Prior
from pytensor.compile import ViewOp
from pytensor.tensor.elemwise import Elemwise

from pymc_marketing.clv import ShiftedBetaGeoModel
from pymc_marketing.clv._aggregation import UNIQUE_DIM, expand_to_customers
from tests.clv.conftest import mock_sample


class TestShiftedBetaGeoModel:
    @classmethod
    def setup_class(cls):
        # Set random seed
        cls.seed = 42
        cls.rng = np.random.default_rng(cls.seed)

        # Test parameters for dual cohort MCMC fit of expected_probability_alive, expected_retention_rate
        # Highend and regular parameters from pg(7) of paper: https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_jim_07.pdf
        cls.alpha_hi_reg = [0.668, 0.704]
        cls.beta_hi_reg = [3.806, 1.182]

        # Test parameters for expected_residual_lifetime, expected_retention_elasticity
        # Both parameter sets from from pg(4) of paper: https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_contractual_mksc_10.pdf
        cls.alpha_case1_case2 = [3.80, 0.067]
        cls.beta_case1_case2 = [15.20, 0.267]

        # Instantiate model with research paper data
        cls.data = pd.read_csv("data/sbg_cohorts.csv").query("T <= 8")
        cls.model = ShiftedBetaGeoModel()
        cls.model.build_model(data=cls.data)

        # Mock an idata object for tests requiring a fitted model
        cls.N = len(cls.data)
        cls.chains = 2
        cls.draws = 50
        cls.cohorts = ["highend", "regular"]

        # Mock a fitted model with [highend, regular] cohort parameters
        cls.model = cls.mock_cohort_fit(
            cls.model,
            [cls.alpha_hi_reg, cls.beta_hi_reg],
            cls.chains,
            cls.draws,
            cls.cohorts,
        )

        # Create separate model for expected_residual_lifetime tests with case1/case2 cohorts
        # Create mock data for case1/case2 cohorts (representing different renewal periods)
        cls.erl_data = pd.DataFrame(
            {
                "customer_id": np.arange(1, 11),
                "recency": np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5]),
                "T": np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5]),
                "cohort": np.array(
                    [
                        "case1",
                        "case1",
                        "case1",
                        "case1",
                        "case1",
                        "case2",
                        "case2",
                        "case2",
                        "case2",
                        "case2",
                    ]
                ),
            }
        )
        cls.erl_model = ShiftedBetaGeoModel()
        cls.erl_model.build_model(data=cls.erl_data)
        cls.erl_cohorts = ["case1", "case2"]
        # Mock a fitted model with [case1, case2] cohort parameters
        # for testing expected_residual_lifetime, expected_retention_elasticity
        cls.erl_model = cls.mock_cohort_fit(
            cls.erl_model,
            [cls.alpha_case1_case2, cls.beta_case1_case2],
            cls.chains,
            cls.draws,
            cls.erl_cohorts,
        )

    # TODO: Generalize and move into conftest? create_mock_fit doesn't support multi-dim parameters
    @classmethod
    def mock_cohort_fit(cls, model, params, chains, draws, cohorts):
        """Mock a fitted model with multi-dim parameters."""
        # Generate arrays from true parameters for alpha and beta
        alpha_beta_sim = [
            cls.rng.normal(
                param,
                1e-3,
                size=(chains, draws, len(cohorts)),
            )
            for param in params
        ]
        # Mock posterior
        param_arrays = [
            xr.DataArray(
                param[0],
                dims=("chain", "draw", "cohort"),
                coords={
                    "chain": np.arange(chains),
                    "draw": np.arange(draws),
                    "cohort": cohorts,
                },
                name=param[1],
            )
            for param in zip(alpha_beta_sim, ["alpha", "beta"], strict=False)
        ]
        posterior = xr.DataTree.from_dict({"/posterior": xr.merge(param_arrays)})
        # Set idata and add fit data group
        model.idata = posterior
        model.set_idata_attrs(model.idata)
        fit_data = model.create_fit_data_group()
        if fit_data is not None:
            model.idata["/fit_data"] = fit_data

        return model

    @pytest.fixture(scope="class")
    def covariate_test_data(self):
        """Test data with covariates for testing covariate functionality."""
        return pd.DataFrame(
            {
                "customer_id": np.arange(1, 21),
                "recency": np.array(
                    [3, 4, 5, 5, 5, 3, 4, 5, 5, 5, 3, 4, 5, 5, 5, 3, 4, 5, 5, 5]
                ),
                "T": np.array(
                    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
                ),
                "cohort": np.array(
                    [
                        "A",
                        "A",
                        "A",
                        "A",
                        "A",
                        "B",
                        "B",
                        "B",
                        "B",
                        "B",
                        "A",
                        "A",
                        "A",
                        "A",
                        "A",
                        "B",
                        "B",
                        "B",
                        "B",
                        "B",
                    ]
                ),
                "channel": np.array(
                    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
                ),
                "tier": np.array(
                    [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
                ),
            }
        )

    @pytest.fixture(scope="class")
    def covariate_pred_data(self):
        """Prediction data with covariates for testing predictions."""
        return pd.DataFrame(
            {
                "customer_id": np.arange(1, 6),
                "T": np.array([5, 5, 5, 5, 5]),
                "cohort": np.array(["A", "A", "B", "B", "A"]),
                "channel": np.array([1, 0, 1, 0, 1]),
                "tier": np.array([2, 1, 2, 1, 2]),
            }
        )

    @pytest.fixture(scope="class")
    def custom_model_config(self):
        return {
            "alpha": Prior("HalfNormal", sigma=10, dims="cohort"),
            "beta": Prior("HalfStudentT", nu=4, sigma=10, dims="cohort"),
        }

    def test_model(self, custom_model_config):
        default_model = ShiftedBetaGeoModel()
        custom_model = ShiftedBetaGeoModel(
            model_config=custom_model_config,
        )

        for model in (default_model, custom_model):
            model.build_model(data=self.data)
            assert isinstance(
                model.model["alpha"].owner.op,
                ViewOp | Elemwise
                if "alpha" not in model.model_config
                else model.model_config["alpha"].pymc_distribution,
            )
            assert isinstance(
                model.model["beta"].owner.op,
                ViewOp | Elemwise
                if "beta" not in model.model_config
                else model.model_config["beta"].pymc_distribution,
            )
            assert model.model.coords == {
                "customer_id": tuple(range(1, self.N + 1)),
                "cohort": ("highend", "regular"),
                "dropout_covariate": (),
            }

        assert default_model.model.eval_rv_shapes() == {
            "kappa": (np.int64(2),),
            "kappa_interval__": (np.int64(2),),
            "phi": (np.int64(2),),
            "phi_interval__": (np.int64(2),),
        }
        assert custom_model.model.eval_rv_shapes() == {
            "alpha": (np.int64(2),),
            "alpha_log__": (np.int64(2),),
            "beta": (np.int64(2),),
            "beta_log__": (np.int64(2),),
        }

    @pytest.fixture(scope="class")
    def prediction_targets(self):
        """Validation data for testing predictive methods."""
        # Data found in Table 1 on pg(3) of paper: https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_jim_07.pdf
        df = pd.DataFrame(
            {
                "regular": [
                    100.0,
                    63.1,
                    46.8,
                    38.2,
                    32.6,
                    28.9,
                    26.2,
                    24.1,
                    22.3,
                    20.7,
                    19.4,
                    18.3,
                    17.3,
                ],
                "highend": [
                    100.0,
                    86.9,
                    74.3,
                    65.3,
                    59.3,
                    55.1,
                    51.7,
                    49.1,
                    46.8,
                    44.5,
                    42.7,
                    40.9,
                    39.4,
                ],
            }
        )
        alive_prob_highend_obs = df["highend"].values / 100
        alive_prob_regular_obs = df["regular"].values / 100
        retention_rate_highend_obs = (
            alive_prob_highend_obs[1:] / alive_prob_highend_obs[:-1]
        )
        retention_rate_regular_obs = (
            alive_prob_regular_obs[1:] / alive_prob_regular_obs[:-1]
        )

        # DERL values from Table 3, pg(5) of paper: https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_contractual_mksc_10.pdf
        # Case 1: alpha=3.80, beta=15.20 (high retention case)
        # Case 2: alpha=0.067, beta=0.267 (low retention case)
        # DERL for T=1,2,3,4,5 renewals with d=0.1 discount rate
        residual_lifetime_case1_obs = np.array([3.31, 3.45, 3.59, 3.72, 3.84])
        residual_lifetime_case2_obs = np.array([7.68, 9.46, 9.86, 10.06, 10.19])

        return (
            alive_prob_highend_obs,
            alive_prob_regular_obs,
            retention_rate_highend_obs,
            retention_rate_regular_obs,
            residual_lifetime_case1_obs,
            residual_lifetime_case2_obs,
        )

    @pytest.fixture(scope="class")
    def predicted_time_series_data(self):
        """Format prediction data to compare against observed prediction_targets."""
        # model was trained on 8 time periods, but will be evaluated against full time series
        max_T = 13
        cohort_names = np.array(["regular", "highend"])
        cohorts_covar = np.array([0, 1])
        T_rng = np.arange(1, max_T + 1, 1)  # T=1 to T=13

        return pd.DataFrame(
            {
                "customer_id": np.arange(1, 1 + max_T * 2, 1),
                "T": np.repeat(T_rng, len(cohort_names)),
                "cohort": np.tile(cohort_names, max_T),
                "covar_cohort": np.tile(cohorts_covar, max_T),
            }
        )

    @pytest.fixture(scope="class")
    def erl_test_data(self):
        """Test data for expected_residual_lifetime with case1/case2 cohorts."""
        # Create test data for (T=1,2,3,4,5) renewals for both case1 and case2 cohorts
        n_periods = 5
        cohort_names = np.array(["case1", "case2"])
        T_rng = np.arange(1, n_periods + 1, 1)

        return pd.DataFrame(
            {
                "customer_id": np.arange(1, 1 + n_periods * 2, 1),
                "T": np.repeat(T_rng, len(cohort_names)),
                "cohort": np.tile(cohort_names, n_periods),
            }
        )

    @pytest.mark.parametrize(
        "missing_column",
        ["customer_id", "recency", "T", "cohort"],
    )
    def test_missing_cols(self, missing_column):
        data_invalid = self.data.drop(columns=missing_column)

        with pytest.raises(
            ValueError,
            match=rf"The following required columns are missing from the input data: \['{missing_column}'\]",
        ):
            model = ShiftedBetaGeoModel()
            model.build_model(data=data_invalid)

    def test_customer_id_duplicate(self):
        with pytest.raises(
            ValueError, match=r"Column customer_id has duplicate entries"
        ):
            data = pd.DataFrame(
                {
                    "customer_id": np.asarray([1, 1]),
                    "recency": np.asarray([1, 1]),
                    "T": np.asarray([1, 1]),
                    "cohort": np.asarray(["A", "A"]),
                }
            )
            model = ShiftedBetaGeoModel()
            model.build_model(data=data)

    @pytest.mark.parametrize(
        "recency, match",
        [
            (np.asarray([1, 3]), r"recency cannot be greater than T"),
            (np.asarray([0, 1]), r"Column recency must be at least 1"),
        ],
    )
    def test_invalid_recency(self, recency, match):
        data = pd.DataFrame(
            {
                "customer_id": np.asarray([1, 2]),
                "recency": recency,
                "T": np.asarray([2, 2]),
                "cohort": np.asarray(["A", "A"]),
            }
        )
        with pytest.raises(ValueError, match=match):
            model = ShiftedBetaGeoModel()
            model.build_model(data=data)

    def test_invalid_T(self):
        data = pd.DataFrame(
            {
                "customer_id": np.asarray([1, 2]),
                "recency": np.asarray([1, 1]),
                "T": np.asarray([1, 1]),
                "cohort": np.asarray(["A", "A"]),
            }
        )
        with pytest.raises(ValueError, match=r"Column T must be at least 2"):
            model = ShiftedBetaGeoModel()
            model.build_model(data=data)

    def test_cohort_T_homogeneity(self):
        data = pd.DataFrame(
            {
                "customer_id": np.asarray([1, 2]),
                "recency": np.asarray([1, 1]),
                "T": np.asarray([2, 3]),
                "cohort": np.asarray(["A", "A"]),
            }
        )
        with pytest.raises(
            ValueError, match=r"T must be homogeneous within each cohort."
        ):
            model = ShiftedBetaGeoModel()
            model.build_model(data=data)

    def test_cohorts_raises_attribute_error_when_data_unspecified(self):
        """Test that accessing cohorts raises AttributeError when data is unspecified."""
        model = ShiftedBetaGeoModel()
        with pytest.raises(
            AttributeError, match=r"cohorts not available. Call build_model"
        ):
            _ = model.cohorts

    def test_cohort_idx_raises_attribute_error_when_data_unspecified(self):
        """Test that accessing cohort_idx raises AttributeError when data is unspecified."""
        model = ShiftedBetaGeoModel()
        with pytest.raises(
            AttributeError, match=r"cohort_idx not available. Call build_model"
        ):
            _ = model.cohort_idx

    def test_model_repr(self, custom_model_config):
        default_repr = (
            "ShiftedBeta-Geometric"
            "\nphi~Uniform(0,1)"
            "\nkappa~Pareto(1,1)"
            "\nalpha=Deterministic(f(kappa,phi))"
            "\nbeta=Deterministic(f(kappa,phi))"
            "\ndropout~CensoredShiftedBetaGeometric(f(alpha),f(beta),<constant>,1)"
        )

        custom_repr = (
            "ShiftedBeta-Geometric"
            "\nalpha~HalfNormal(0,10)"
            "\nbeta~HalfStudentT(4,10)"
            "\ndropout~CensoredShiftedBetaGeometric(f(alpha),f(beta),<constant>,1)"
        )

        for repr in zip(
            [custom_model_config, None], [custom_repr, default_repr], strict=False
        ):
            model = ShiftedBetaGeoModel(
                model_config=repr[0],
            )
            model.build_model(data=self.data)
            assert model.__repr__().replace(" ", "") == repr[1]

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "method, rtol",
        [
            ("mcmc", 0.1),
            ("map", 0.1),
            ("advi", 0.6),
        ],
    )
    def test_model_convergence(self, method, rtol):
        model = ShiftedBetaGeoModel()
        model.build_model(data=self.data)

        sample_kwargs = dict(random_seed=self.seed) if method == "mcmc" else {}
        model.fit(data=self.data, method=method, progressbar=False, **sample_kwargs)

        fit = model.idata.posterior
        # Compare mean of each cohort parameter (averaging over chains and draws)
        alpha_means = fit["alpha"].mean(dim=["chain", "draw"]).values
        beta_means = fit["beta"].mean(dim=["chain", "draw"]).values

        np.testing.assert_allclose(
            alpha_means,
            self.alpha_hi_reg,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            beta_means,
            self.beta_hi_reg,
            rtol=rtol,
        )

    def test_fit_result_without_fit(self, mocker, custom_model_config):
        model = ShiftedBetaGeoModel(model_config=custom_model_config)
        model.build_model(data=self.data)
        with pytest.raises(RuntimeError, match=r"The model hasn't been fit yet"):
            model.fit_result

        mocker.patch("pymc.sample", mock_sample)

        idata = model.fit(
            data=self.data,
            tune=5,
            chains=2,
            draws=10,
            compute_convergence_checks=False,
        )
        assert isinstance(idata, xr.DataTree)
        assert len(idata.posterior.chain) == 2
        assert len(idata.posterior.draw) == 10
        assert model.idata is idata

    def test_save_load(self, tmp_path):
        model = ShiftedBetaGeoModel()
        save_path = tmp_path / "test_model"
        model.build_model(data=self.data)
        model.fit(data=self.data, method="map")
        model.save(save_path)
        model2 = ShiftedBetaGeoModel.load(save_path)
        assert model.model_config == model2.model_config
        assert model.sampler_config == model2.sampler_config
        assert model.idata == model2.idata

    def test_requires_cohort_dims_on_alpha_beta_missing_raises(self):
        config_missing_dims = {
            "alpha": Prior("HalfNormal", sigma=10),  # missing dims
            "beta": Prior("HalfStudentT", nu=4, sigma=10, dims="cohort"),
        }
        with pytest.raises(ValueError, match=r'dims="cohort"'):
            model = ShiftedBetaGeoModel(
                model_config=config_missing_dims,
            )
            model.build_model(data=self.data)

    def test_requires_cohort_dims_on_alpha_beta_incorrect_raises(self):
        config_incorrect_dims = {
            "alpha": Prior("HalfNormal", sigma=10, dims="customer_id"),
            "beta": Prior("HalfStudentT", nu=4, sigma=10, dims="cohort"),
        }
        with pytest.raises(ValueError, match=r'dims="cohort"'):
            model = ShiftedBetaGeoModel(
                model_config=config_incorrect_dims,
            )
            model.build_model(data=self.data)

    def test_accepts_alpha_beta_with_cohort_dims(self):
        config_ok = {
            "alpha": Prior("HalfNormal", sigma=10, dims="cohort"),
            "beta": Prior("HalfStudentT", nu=4, sigma=10, dims="dim"),
        }
        with pytest.raises(ValueError, match=r'dims="cohort"'):
            model = ShiftedBetaGeoModel(
                model_config=config_ok,
            )
            model.build_model(data=self.data)

    def test_extract_predictive_variables_invalid(self):
        invalid_cohort_data = pd.DataFrame(
            {
                "customer_id": np.asarray([1, 2]),
                "recency": np.asarray([1, 1]),
                "T": np.asarray([2, 3]),
                "cohort": np.asarray(["A", "A"]),
            }
        )
        with pytest.raises(
            ValueError,
            match=r"Cohorts in prediction data do not match cohorts used to fit the model.",
        ):
            self.model._extract_predictive_variables(
                invalid_cohort_data,
                customer_varnames=["customer_id", "recency", "T", "cohort"],
            )

        T_lt_2_data = pd.DataFrame(
            {
                "customer_id": np.asarray([1, 2]),
                "recency": np.asarray([1, 1]),
                "T": np.asarray([0, 0]),
                "cohort": np.asarray(["highend", "highend"]),
            }
        )
        with pytest.raises(
            ValueError, match=r"T must be a non-zero, positive whole number."
        ):
            self.model._extract_predictive_variables(
                T_lt_2_data, customer_varnames=["customer_id", "recency", "T", "cohort"]
            )

    def test_expected_probability_alive(
        self, prediction_targets, predicted_time_series_data
    ):
        # extract observed active probability arrays by cohort
        alive_prob_highend_obs, alive_prob_regular_obs, _, _, _, _ = prediction_targets

        # run full predictions and extract by cohort
        expected_retention_rate_cohorts = self.model.expected_probability_alive(
            predicted_time_series_data,
            future_t=-1,  # prob_alive is always 1 when T==1
        ).mean(("chain", "draw"))
        expected_alive_prob_highend = expected_retention_rate_cohorts.sel(
            cohort="highend"
        ).values
        expected_alive_prob_regular = expected_retention_rate_cohorts.sel(
            cohort="regular"
        ).values

        np.testing.assert_allclose(
            expected_alive_prob_highend, alive_prob_highend_obs, rtol=0.05
        )
        np.testing.assert_allclose(
            expected_alive_prob_regular, alive_prob_regular_obs, rtol=0.05
        )

    def test_expected_retention_rate(
        self, prediction_targets, predicted_time_series_data
    ):
        # extract observed retention rate data arrays by cohort
        _, _, retention_rate_highend_obs, retention_rate_regular_obs, _, _ = (
            prediction_targets
        )

        # retention rate is estimated from T-1 time periods, reducing array length by 1
        retention_data = predicted_time_series_data[
            predicted_time_series_data["T"] <= 12
        ]
        # run full predictions and extract by cohort
        expected_retention_rate_cohorts = self.model.expected_retention_rate(
            retention_data,
            future_t=0,
        ).mean(("chain", "draw"))
        expected_retention_rate_highend = expected_retention_rate_cohorts.sel(
            cohort="highend"
        ).values
        expected_retention_rate_regular = expected_retention_rate_cohorts.sel(
            cohort="regular"
        ).values

        np.testing.assert_allclose(
            expected_retention_rate_highend, retention_rate_highend_obs, rtol=0.05
        )
        np.testing.assert_allclose(
            expected_retention_rate_regular, retention_rate_regular_obs, rtol=0.05
        )

    def test_expected_residual_lifetime(self, prediction_targets, erl_test_data):
        """Test expected_residual_lifetime against Table 3 values from the paper."""
        # Extract observed residual lifetime data arrays by cohort
        (
            _,
            _,
            _,
            _,
            residual_lifetime_case1_obs,
            residual_lifetime_case2_obs,
        ) = prediction_targets

        # Run full predictions with discount_rate=0.1 (as used in paper Table 3)
        expected_residual_lifetime_cohorts = self.erl_model.expected_residual_lifetime(
            erl_test_data,
            discount_rate=0.1,
        ).mean(("chain", "draw"))

        # Extract predictions by cohort
        expected_residual_lifetime_case1 = expected_residual_lifetime_cohorts.sel(
            cohort="case1"
        ).values
        expected_residual_lifetime_case2 = expected_residual_lifetime_cohorts.sel(
            cohort="case2"
        ).values

        # Compare against paper values (Table 3, pg 5)
        np.testing.assert_allclose(
            expected_residual_lifetime_case1, residual_lifetime_case1_obs, rtol=1e-2
        )
        np.testing.assert_allclose(
            expected_residual_lifetime_case2, residual_lifetime_case2_obs, rtol=1e-2
        )

    def test_expected_retention_elasticity(self, erl_test_data):
        """Test expected_retention_elasticity against Table 3 values from the paper.
        Note no values for retention elasticity are provided in the paper,
        but can be derived from the DERL and retention rate:
        retention_elasticity(T) = DERL(T-1) / retention_rate(T)
        """
        # Compute expected retention elasticity from the model with discount_rate=0.1
        expected_retention_elasticity_cohorts = (
            self.erl_model.expected_retention_elasticity(
                erl_test_data,
                discount_rate=0.1,
            ).mean(("chain", "draw"))
        )

        # Extract elasticity predictions by cohort
        expected_retention_elasticity_case1 = expected_retention_elasticity_cohorts.sel(
            cohort="case1"
        ).values
        expected_retention_elasticity_case2 = expected_retention_elasticity_cohorts.sel(
            cohort="case2"
        ).values

        ### DERIVE VALUES FROM OTHER EXPRESSIONS FOR VALIDATION ###

        # compute expected retention rates for T (current period)
        expected_retention_rate_cohorts = self.erl_model.expected_retention_rate(
            erl_test_data,
            future_t=0,
        ).mean(("chain", "draw"))

        expected_retention_rate_case1 = expected_retention_rate_cohorts.sel(
            cohort="case1"
        ).values
        expected_retention_rate_case2 = expected_retention_rate_cohorts.sel(
            cohort="case2"
        ).values

        # Create test data for DERL T-1 (one period earlier) to compute DERL(T-1)
        erl_test_data_T_minus_1 = erl_test_data.copy()
        erl_test_data_T_minus_1["T"] = erl_test_data_T_minus_1["T"] - 1
        # Filter out T=0 (can't have T < 1 in the model)
        erl_test_data_T_minus_1 = erl_test_data_T_minus_1[
            erl_test_data_T_minus_1["T"] >= 1
        ]

        # Compute DERL for T-1
        expected_residual_lifetime_cohorts_T_minus_1 = (
            self.erl_model.expected_residual_lifetime(
                erl_test_data_T_minus_1,
                discount_rate=0.1,
            ).mean(("chain", "draw"))
        )

        expected_residual_lifetime_case1_T_minus_1 = (
            expected_residual_lifetime_cohorts_T_minus_1.sel(cohort="case1").values
        )
        expected_residual_lifetime_case2_T_minus_1 = (
            expected_residual_lifetime_cohorts_T_minus_1.sel(cohort="case2").values
        )

        # Compute expected elasticity using the formula: DERL(T-1) / retention_rate(T)
        # Skip T=1 since we don't have T-1=0
        expected_elasticity_case1_formula = (
            expected_residual_lifetime_case1_T_minus_1
            / expected_retention_rate_case1[1:]
        )
        expected_elasticity_case2_formula = (
            expected_residual_lifetime_case2_T_minus_1
            / expected_retention_rate_case2[1:]
        )

        ### ASSERTIONS ###
        # Compare model's elasticity against the formula-based elasticity
        # Skip first value (T=1) since formula needs T-1
        np.testing.assert_allclose(
            expected_retention_elasticity_case1[1:],
            expected_elasticity_case1_formula,
            rtol=1e-1,
            err_msg="Case1: Model elasticity doesn't match DERL(T-1)/retention_rate(T)",
        )
        np.testing.assert_allclose(
            expected_retention_elasticity_case2[1:],
            expected_elasticity_case2_formula,
            rtol=2.0,  # T=1 variance is very high, but var for T=5 is 1e-2
            err_msg="Case2: Model elasticity doesn't match DERL(T-1)/retention_rate(T)",
        )

        # Additional validation: elasticity properties
        # The elasticity should be positive
        assert np.all(expected_retention_elasticity_case1 > 0)
        assert np.all(expected_retention_elasticity_case2 > 0)

        # Elasticity should generally be higher for the low retention case
        # (customers with lower retention are more sensitive to retention changes)
        assert np.mean(expected_retention_elasticity_case2) > np.mean(
            expected_retention_elasticity_case1
        )

        # Verify retention rates are valid (between 0 and 1) and increasing with T
        assert np.all(
            (expected_retention_rate_case1 > 0) & (expected_retention_rate_case1 < 1)
        )
        assert np.all(
            (expected_retention_rate_case2 > 0) & (expected_retention_rate_case2 < 1)
        )
        # Retention rates should increase over time (with more renewals)
        assert np.all(np.diff(expected_retention_rate_case1) >= 0)
        assert np.all(np.diff(expected_retention_rate_case2) >= 0)

    def test_model_with_covariates_hierarchical(self, covariate_test_data):
        """Test model building and fitting with covariates using hierarchical parameterization."""
        model_config = {
            "phi": Prior("Uniform", lower=0, upper=1, dims="cohort"),
            "kappa": Prior("Pareto", alpha=1, m=1, dims="cohort"),
            "dropout_coefficient": Prior("Normal", mu=0, sigma=2),
            "dropout_covariate_cols": ["channel", "tier"],
        }

        model = ShiftedBetaGeoModel(model_config=model_config)
        model.build_model(data=covariate_test_data)

        # Check that coords include dropout_covariate
        assert list(model.model.coords["dropout_covariate"]) == ["channel", "tier"]

        # Check that covariate columns are stored
        assert model.dropout_covariate_cols == ["channel", "tier"]

        # Check that alpha_scale and beta_scale exist (not alpha and beta directly)
        assert "alpha_scale" in model.model.named_vars
        assert "beta_scale" in model.model.named_vars

        # Check that customer-level alpha and beta exist
        assert "alpha" in model.model.named_vars
        assert "beta" in model.model.named_vars

        # Check that dropout coefficients exist
        assert "dropout_coefficient_alpha" in model.model.named_vars
        assert "dropout_coefficient_beta" in model.model.named_vars

        # Check shapes
        assert model.model["alpha_scale"].eval().shape == (2,)  # 2 cohorts
        assert model.model["beta_scale"].eval().shape == (2,)  # 2 cohorts
        assert model.model["alpha"].eval().shape == (20,)  # 20 customers
        assert model.model["beta"].eval().shape == (20,)  # 20 customers
        assert model.model["dropout_coefficient_alpha"].eval().shape == (
            2,
        )  # 2 covariates
        assert model.model["dropout_coefficient_beta"].eval().shape == (
            2,
        )  # 2 covariates

        # Fit with MAP to verify it runs without errors
        model.fit(data=covariate_test_data, method="map", maxeval=10)
        assert model.idata is not None

    def test_model_with_covariates_direct(self, covariate_test_data):
        """Test model building with covariates using direct alpha/beta parameterization."""
        model_config = {
            "alpha": Prior("HalfNormal", sigma=10, dims="cohort"),
            "beta": Prior("HalfNormal", sigma=10, dims="cohort"),
            "dropout_coefficient": Prior("Normal", mu=0, sigma=2),
            "dropout_covariate_cols": ["channel"],
        }

        model = ShiftedBetaGeoModel(model_config=model_config)
        model.build_model(data=covariate_test_data)

        # Check that coords include dropout_covariate
        assert "dropout_covariate" in model.model.coords
        assert list(model.model.coords["dropout_covariate"]) == ["channel"]

        # Check that alpha_scale and beta_scale exist (direct parameterization with covariates)
        assert "alpha_scale" in model.model.named_vars
        assert "beta_scale" in model.model.named_vars

        # Check that customer-level alpha and beta exist
        assert "alpha" in model.model.named_vars
        assert "beta" in model.model.named_vars

        # Check shapes
        assert model.model["alpha_scale"].eval().shape == (2,)  # 2 cohorts
        assert model.model["beta_scale"].eval().shape == (2,)  # 2 cohorts
        assert model.model["alpha"].eval().shape == (20,)  # 20 customers
        assert model.model["beta"].eval().shape == (20,)  # 20 customers

        # Fit with MAP to verify it runs without errors
        model.fit(data=covariate_test_data, method="map", maxeval=10)
        assert model.idata is not None

    def test_predictions_with_covariates(
        self, covariate_test_data, covariate_pred_data
    ):
        """Test that prediction methods work with covariates."""
        model_config = {
            "phi": Prior("Uniform", lower=0, upper=1, dims="cohort"),
            "kappa": Prior("Pareto", alpha=1, m=1, dims="cohort"),
            "dropout_coefficient": Prior("Normal", mu=0, sigma=2),
            "dropout_covariate_cols": ["channel"],
        }

        model = ShiftedBetaGeoModel(model_config=model_config)
        model.build_model(data=covariate_test_data)
        model.fit(data=covariate_test_data, method="map", maxeval=10)

        # Use only the channel covariate column from pred_data
        pred_data = covariate_pred_data[
            ["customer_id", "T", "cohort", "channel"]
        ].copy()

        # Test each prediction method
        retention_rate = model.expected_retention_rate(data=pred_data, future_t=1)
        assert retention_rate.shape == (1, 1, 5)
        assert np.all(retention_rate > 0) and np.all(retention_rate < 1)

        prob_alive = model.expected_probability_alive(data=pred_data, future_t=0)
        assert prob_alive.shape == (1, 1, 5)
        assert np.all(prob_alive > 0) and np.all(prob_alive <= 1)

        residual_lifetime = model.expected_residual_lifetime(
            data=pred_data, discount_rate=0.05
        )
        assert residual_lifetime.shape == (1, 1, 5)
        assert np.all(residual_lifetime > 0)

        retention_elasticity = model.expected_retention_elasticity(
            data=pred_data, discount_rate=0.05
        )
        assert retention_elasticity.shape == (1, 1, 5)
        assert np.all(retention_elasticity > 0)

    def test_missing_covariate_columns_raises_error(self, covariate_test_data):
        """Test that missing covariate columns in data raises an error."""
        # Create test data without the required covariate columns by dropping them
        data_missing_covariates = covariate_test_data.drop(
            columns=["channel", "tier"]
        ).iloc[:10]

        model_config = {
            "phi": Prior("Uniform", lower=0, upper=1, dims="cohort"),
            "kappa": Prior("Pareto", alpha=1, m=1, dims="cohort"),
            "dropout_coefficient": Prior("Normal", mu=0, sigma=2),
            "dropout_covariate_cols": ["channel"],
        }

        with pytest.raises(ValueError, match="missing from the input data"):
            model = ShiftedBetaGeoModel(model_config=model_config)
            model.build_model(data=data_missing_covariates)

    def test_covariate_cols_only_in_config(self, covariate_test_data):
        """Test that passing only dropout_covariate_cols into model_config (without priors) works."""

        model_config = {
            "dropout_covariate_cols": ["channel"],
        }

        # Should not raise ModelConfigError
        model = ShiftedBetaGeoModel(model_config=model_config)
        assert model.dropout_covariate_cols == ["channel"]
        model.build_model(data=covariate_test_data)
        assert "dropout_covariate" in model.model.coords

    def test_predictions_with_covariates_subset_cohorts(self):
        # Training data with 3 cohorts
        train_data = pd.DataFrame(
            {
                "customer_id": range(30),
                "recency": [3, 4, 5] * 10,
                "T": [5] * 30,
                "cohort": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
                "channel": [0, 1, 0, 1, 0] * 6,
            }
        )

        model = ShiftedBetaGeoModel(
            model_config={"dropout_covariate_cols": ["channel"]}
        )
        model.fit(data=train_data, method="map", maxeval=10)

        # Prediction data with subset of cohorts (NOT starting at index 0)
        pred_data = pd.DataFrame(
            {
                "customer_id": [100, 101, 102],
                "T": [3, 3, 3],
                "cohort": ["B", "C", "C"],  # Missing cohort "A"
                "channel": [1, 0, 1],
            }
        )

        # Should not raise IndexError
        prob_alive = model.expected_probability_alive(data=pred_data, future_t=1)
        assert prob_alive.shape[-1] == 3  # 3 customers

        retention = model.expected_retention_rate(data=pred_data, future_t=1)
        assert retention.shape[-1] == 3

    def test_covariate_predictions_single_vs_all_cohorts_consistency(self):
        """Predictions for a cohort must be identical whether queried via
        all-cohorts or single-cohort data."""
        train_data = pd.DataFrame(
            {
                "customer_id": range(60),
                "recency": [3, 4, 5, 5, 5, 5] * 10,  # Mix of churned and active
                "T": [5] * 60,
                "cohort": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
                "channel": [0, 1] * 30,
            }
        )

        model = ShiftedBetaGeoModel(
            model_config={"dropout_covariate_cols": ["channel"]}
        )
        model.fit(data=train_data, method="map", maxeval=50)

        # Get active customers for predictions
        active_all = train_data.query("recency == T").copy()
        active_B = active_all[active_all["cohort"] == "B"].copy()

        # Scenario A: Predict all cohorts, select cohort B
        pred_all = model.expected_probability_alive(data=active_all, future_t=2)
        pred_all_B = pred_all.sel(cohort="B")

        # Scenario B: Predict single cohort B directly
        pred_single_B = model.expected_probability_alive(data=active_B, future_t=2)

        # Values must be identical (within floating point tolerance)
        xr.testing.assert_allclose(
            pred_all_B.mean("cohort"),  # Average over customers in cohort B
            pred_single_B.mean("cohort"),
            rtol=1e-10,
        )

    def test_covariate_effect_preserved_single_cohort(self):
        """Covariate effects must be identical regardless of query method."""
        train_data = pd.DataFrame(
            {
                "customer_id": range(40),
                "recency": [4, 4, 4, 4] * 10,  # All active
                "T": [4] * 40,
                "cohort": ["X"] * 20 + ["Y"] * 20,
                "feature": [0.0, 1.0] * 20,  # Binary covariate
            }
        )

        model = ShiftedBetaGeoModel(
            model_config={"dropout_covariate_cols": ["feature"]}
        )
        model.fit(data=train_data, method="map", maxeval=50)

        active = train_data.query("recency == T").copy()
        active_X = active[active["cohort"] == "X"].copy()

        # All cohorts prediction
        dataset_all = model._extract_predictive_variables(
            active.assign(future_t=1), customer_varnames=["T", "future_t", "cohort"]
        )

        # Single cohort prediction
        dataset_single = model._extract_predictive_variables(
            active_X.assign(future_t=1), customer_varnames=["T", "future_t", "cohort"]
        )

        # Alpha values for cohort X customers must match
        alpha_all_X = dataset_all["alpha"].sel(cohort="X")
        alpha_single_X = dataset_single["alpha"]

        xr.testing.assert_allclose(alpha_all_X, alpha_single_X, rtol=1e-10)

    def test_covariate_path_coordinate_integrity(self):
        """Verify customer_id coordinates are preserved through covariate path."""
        train_data = pd.DataFrame(
            {
                "customer_id": [100, 101, 102, 103],
                "recency": [3, 3, 3, 3],
                "T": [3] * 4,
                "cohort": ["P", "P", "Q", "Q"],
                "cov": [0.5, 1.5, 0.5, 1.5],
            }
        )

        model = ShiftedBetaGeoModel(model_config={"dropout_covariate_cols": ["cov"]})
        model.fit(data=train_data, method="map", maxeval=10)

        pred_data = train_data.query("recency == T").assign(future_t=1)
        dataset = model._extract_predictive_variables(
            pred_data, customer_varnames=["T", "future_t", "cohort"]
        )

        # Verify customer_id coordinate exists and matches input
        # (After swap_dims, customer_id becomes a coord on cohort dim)
        assert "customer_id" in dataset.coords or "cohort" in dataset.dims

        # Verify alpha has proper dimensions
        assert "alpha" in dataset
        assert dataset["alpha"].dims[-1] == "cohort"  # After swap_dims


FH_ALPHA = [0.668, 0.704]  # highend, regular (Fader & Hardie 2007, p. 7)
FH_BETA = [3.806, 1.182]
FH_TABLE = pd.DataFrame(
    {
        "cohort": ["highend"] * 8 + ["regular"] * 8,
        "period": list(range(8)) * 2,
        "n_alive": [
            *[1000, 869, 743, 653, 593, 551, 517, 491],
            *[1000, 631, 468, 382, 326, 289, 262, 241],
        ],
    }
)


def synthetic_covariate_data(n: int = 400, seed: int = 11) -> pd.DataFrame:
    """Three cohorts with cohort-specific T and two categorical covariates."""
    rng = np.random.default_rng(seed)
    cohort = rng.choice(["c1", "c2", "c3"], size=n)
    T = np.select([cohort == "c1", cohort == "c2"], [6, 8], default=5)
    p = rng.beta(0.8, 2.0, size=n)
    recency = np.minimum(rng.geometric(p), T)
    return pd.DataFrame(
        {
            "customer_id": rng.permutation(np.arange(1, n + 1)),
            "recency": recency,
            "T": T,
            "cohort": cohort,
            "channel": rng.integers(0, 2, size=n),
            "tier": rng.integers(1, 4, size=n),
        }
    )


def mock_posterior(model, draws=20, chains=2, seed=0, spread=0.3):
    """Attach a mocked posterior with every variable the predictive methods read.

    ``spread`` is the log-scale standard deviation of the alpha/beta draws; keep it small
    for PSIS-LOO tests so the Pareto k diagnostics stay low.
    """
    rng = np.random.default_rng(seed)
    cohorts = list(model.cohorts)
    coords = {"chain": np.arange(chains), "draw": np.arange(draws), "cohort": cohorts}
    size = (chains, draws, len(cohorts))
    data = {}

    def lognormal(center):
        return center * np.exp(spread * rng.normal(size=size))

    alpha, beta = lognormal(0.7), lognormal(2.0)
    # free RVs of the default hierarchical priors, consistent with alpha/beta
    data["phi"] = (("chain", "draw", "cohort"), alpha / (alpha + beta))
    data["kappa"] = (("chain", "draw", "cohort"), alpha + beta)
    if model.dropout_covariate_cols:
        data["alpha_scale"] = (("chain", "draw", "cohort"), alpha)
        data["beta_scale"] = (("chain", "draw", "cohort"), beta)
        coords["dropout_covariate"] = model.dropout_covariate_cols
        csize = (chains, draws, len(model.dropout_covariate_cols))
        data["dropout_coefficient_alpha"] = (
            ("chain", "draw", "dropout_covariate"),
            rng.normal(0, 0.3, csize),
        )
        data["dropout_coefficient_beta"] = (
            ("chain", "draw", "dropout_covariate"),
            rng.normal(0, 0.3, csize),
        )
    else:
        data["alpha"] = (("chain", "draw", "cohort"), alpha)
        data["beta"] = (("chain", "draw", "cohort"), beta)
    posterior = xr.Dataset(data, coords=coords)
    model.idata = xr.DataTree.from_dict({"/posterior": posterior})
    model.set_idata_attrs(model.idata)
    model.idata["/fit_data"] = model.create_fit_data_group()
    return model


class TestShiftedBetaGeoAggregate:
    """Opt-in weighted-count likelihood (``aggregate=True``) of ``ShiftedBetaGeoModel``."""

    @pytest.fixture(scope="class")
    def cohort_data(self) -> pd.DataFrame:
        return pd.read_csv("data/sbg_cohorts.csv")

    @pytest.fixture(scope="class")
    def covariate_data(self) -> pd.DataFrame:
        return synthetic_covariate_data()

    @staticmethod
    def _pair(data, model_config=None):
        """Build the same model with and without aggregation."""
        model_config = dict(model_config or {})
        plain = ShiftedBetaGeoModel(model_config={**model_config, "aggregate": False})
        agg = ShiftedBetaGeoModel(model_config={**model_config, "aggregate": True})
        plain.build_model(data=data)
        agg.build_model(data=data)
        return plain, agg

    def test_aggregate_flag_defaults_to_false(self):
        assert ShiftedBetaGeoModel().model_config["aggregate"] is False
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = ShiftedBetaGeoModel(model_config={"aggregate": True})
        assert model.model_config["aggregate"] is True

    def test_default_graph_keeps_customer_dims(self, cohort_data):
        model = ShiftedBetaGeoModel()
        model.build_model(data=cohort_data)

        assert UNIQUE_DIM not in model.model.coords
        assert model.model.named_vars_to_dims["dropout"] == ("customer_id",)
        assert "weights" not in model.model.named_vars
        assert model.aggregation.compression_ratio == 1.0
        np.testing.assert_array_equal(
            model.aggregation.customer_id, cohort_data["customer_id"]
        )

    def test_aggregated_graph_uses_unique_obs(self, cohort_data):
        model = ShiftedBetaGeoModel(model_config={"aggregate": True})
        model.build_model(data=cohort_data)

        agg = model.aggregation
        assert agg.n_unique == 16  # 2 cohorts x 8 recency values
        assert list(model.model.coords[UNIQUE_DIM]) == list(range(16))
        assert list(model.model.coords["customer_id"]) == list(
            cohort_data["customer_id"]
        )
        assert model.model.named_vars_to_dims["dropout"] == (UNIQUE_DIM,)
        assert model.model.named_vars_to_dims["weights"] == (UNIQUE_DIM,)
        np.testing.assert_array_equal(model.model["weights"].eval(), agg.weights)
        assert agg.weights.sum() == len(cohort_data)

    @pytest.mark.parametrize(
        "dataset, model_config",
        [
            ("cohort_data", None),
            (
                "cohort_data",
                {
                    "alpha": Prior("HalfNormal", sigma=10, dims="cohort"),
                    "beta": Prior("HalfStudentT", nu=4, sigma=10, dims="cohort"),
                },
            ),
            ("covariate_data", {"dropout_covariate_cols": ["channel", "tier"]}),
            (
                "covariate_data",
                {
                    "alpha": Prior("HalfNormal", sigma=10, dims="cohort"),
                    "beta": Prior("HalfStudentT", nu=4, sigma=10, dims="cohort"),
                    "dropout_covariate_cols": ["channel", "tier"],
                },
            ),
        ],
        ids=["hier", "direct", "hier-covariates", "direct-covariates"],
    )
    def test_logp_and_gradient_match_unaggregated(self, request, dataset, model_config):
        """A2: identical model log-density and gradient at shared points."""
        data = request.getfixturevalue(dataset)
        plain, agg = self._pair(data, model_config)

        point = plain.model.initial_point()
        assert set(point) == set(agg.model.initial_point())
        rng = np.random.default_rng(5)
        perturbed = {
            k: v + 0.3 * rng.normal(size=np.shape(v)) for k, v in point.items()
        }

        logp_plain, logp_agg = plain.model.compile_logp(), agg.model.compile_logp()
        dlogp_plain, dlogp_agg = plain.model.compile_dlogp(), agg.model.compile_dlogp()
        for pt_ in (point, perturbed):
            np.testing.assert_allclose(logp_agg(pt_), logp_plain(pt_), rtol=1e-10)
            np.testing.assert_allclose(dlogp_agg(pt_), dlogp_plain(pt_), rtol=1e-10)

    def test_covariate_branch_registers_customer_alpha_only_when_unaggregated(
        self, covariate_data
    ):
        plain, agg = self._pair(
            covariate_data, {"dropout_covariate_cols": ["channel", "tier"]}
        )

        assert plain.model.named_vars_to_dims["alpha"] == ("customer_id",)
        assert "alpha" not in agg.model.named_vars
        assert "beta" not in agg.model.named_vars
        for name in ("alpha_scale", "beta_scale", "dropout_coefficient_alpha"):
            assert name in agg.model.named_vars
        assert agg.model.named_vars_to_dims["dropout_data"] == (
            UNIQUE_DIM,
            "dropout_covariate",
        )

    @pytest.mark.parametrize(
        "dataset, model_config",
        [
            ("cohort_data", None),
            ("covariate_data", {"dropout_covariate_cols": ["channel", "tier"]}),
        ],
        ids=["cohorts", "covariates"],
    )
    def test_predictive_methods_invariant(self, request, dataset, model_config):
        """A6: every expected_* method returns identical customer-level arrays."""
        data = request.getfixturevalue(dataset)
        plain, agg = self._pair(data, model_config)
        mock_posterior(plain)
        agg.idata = plain.idata

        for method, kwargs in [
            ("expected_probability_alive", {"future_t": 2}),
            ("expected_retention_rate", {"future_t": 1}),
            ("expected_residual_lifetime", {"discount_rate": 0.05}),
            ("expected_retention_elasticity", {"discount_rate": 0.05}),
        ]:
            expected = getattr(plain, method)(**kwargs)
            got = getattr(agg, method)(**kwargs)
            xr.testing.assert_identical(got, expected)

    def test_save_load_roundtrip(self, cohort_data, tmp_path):
        """A7: the aggregate flag, graph and id survive save/load without base-class changes."""
        model = ShiftedBetaGeoModel(model_config={"aggregate": True})
        model.build_model(data=cohort_data)
        model.fit(data=cohort_data, method="map", progressbar=False)
        model.save(tmp_path / "sbg_agg")

        loaded = ShiftedBetaGeoModel.load(tmp_path / "sbg_agg")

        assert loaded.model_config["aggregate"] is True
        assert loaded.model_config == model.model_config
        assert loaded.id == model.id
        assert loaded.model.named_vars_to_dims["dropout"] == (UNIQUE_DIM,)
        np.testing.assert_array_equal(
            loaded.aggregation.weights, model.aggregation.weights
        )
        np.testing.assert_array_equal(
            loaded.aggregation.inverse, model.aggregation.inverse
        )
        assert loaded.idata == model.idata

        plain = ShiftedBetaGeoModel()
        assert plain.id != model.id

    def test_weight_column_enables_aggregation(self):
        from pymc_marketing.clv._aggregation import survival_counts_to_sbg_data

        weighted = survival_counts_to_sbg_data(FH_TABLE)
        model = ShiftedBetaGeoModel()  # aggregate=False, but weights present
        model.build_model(data=weighted)

        assert model.model.named_vars_to_dims["dropout"] == (UNIQUE_DIM,)
        assert model.aggregation.total_weight == 2000
        assert (
            model.aggregation.compression_ratio == 16 / 2000
        )  # no low-compression warning

    def test_non_positive_weight_column_rejected(self):
        from pymc_marketing.clv._aggregation import survival_counts_to_sbg_data

        weighted = survival_counts_to_sbg_data(FH_TABLE)
        weighted.loc[0, "weight"] = 0.0
        with pytest.raises(ValueError, match="weight"):
            ShiftedBetaGeoModel().build_model(data=weighted)

    def test_map_on_survival_table_matches_customer_level(self, cohort_data):
        """A9: the converter's weighted rows fit like the 2000-customer frame, near F&H."""
        from pymc_marketing.clv._aggregation import survival_counts_to_sbg_data

        weighted = survival_counts_to_sbg_data(FH_TABLE)
        table_model = ShiftedBetaGeoModel()
        table_model.fit(data=weighted, method="map", progressbar=False)
        customer_model = ShiftedBetaGeoModel()
        customer_model.fit(data=cohort_data, method="map", progressbar=False)

        for name, published in [("alpha", FH_ALPHA), ("beta", FH_BETA)]:
            table = table_model.fit_result[name].sel(cohort=["highend", "regular"])
            customer = customer_model.fit_result[name].sel(
                cohort=["highend", "regular"]
            )
            np.testing.assert_allclose(table, customer, rtol=1e-3)
            np.testing.assert_allclose(table.squeeze(), published, rtol=0.1)

    def test_low_compression_warns_on_build(self, covariate_data):
        noisy = covariate_data.assign(noise=np.linspace(0, 1, len(covariate_data)))
        model = ShiftedBetaGeoModel(
            model_config={"aggregate": True, "dropout_covariate_cols": ["noise"]}
        )
        with pytest.warns(UserWarning, match="compression"):
            model.build_model(data=noisy)

    @pytest.mark.parametrize(
        "dataset, model_config",
        [
            ("cohort_data", None),
            ("covariate_data", {"dropout_covariate_cols": ["channel", "tier"]}),
        ],
        ids=["cohorts", "covariates"],
    )
    def test_compute_log_likelihood_is_unweighted(self, request, dataset, model_config):
        """A3: weights x per-row log-likelihood sums to the customer-level total per draw."""
        data = request.getfixturevalue(dataset)
        plain, agg = self._pair(data, model_config)
        mock_posterior(plain, draws=30, seed=2)
        mock_posterior(agg, draws=30, seed=2)

        ll_plain = plain.compute_log_likelihood()["dropout"]
        ll_agg = agg.compute_log_likelihood()["dropout"]

        assert ll_plain.dims == ("chain", "draw", "customer_id")
        assert ll_agg.dims == ("chain", "draw", UNIQUE_DIM)
        assert "log_likelihood" in plain.idata and "log_likelihood" in agg.idata
        xr.testing.assert_identical(plain.idata.log_likelihood["dropout"], ll_plain)

        weights = xr.DataArray(agg.aggregation.weights, dims=UNIQUE_DIM)
        np.testing.assert_allclose(
            (ll_agg * weights).sum(UNIQUE_DIM), ll_plain.sum("customer_id"), rtol=1e-10
        )
        expanded = expand_to_customers(ll_agg, agg.aggregation)
        np.testing.assert_allclose(expanded.values, ll_plain.values, rtol=1e-10)

        raw = pm.compute_log_likelihood(
            agg.idata,
            var_names=["dropout"],
            model=agg.model,
            extend_inferencedata=False,
        )["dropout"]
        np.testing.assert_allclose(raw.values, (ll_agg * weights).values, rtol=1e-12)

    def test_customer_loo_matches_unaggregated(self, cohort_data):
        """A4: customer-level LOO from the aggregated fit equals az.loo on the customer graph."""
        plain, agg = self._pair(cohort_data)
        mock_posterior(plain, draws=200, chains=2, seed=3, spread=0.05)
        mock_posterior(agg, draws=200, chains=2, seed=3, spread=0.05)

        plain.compute_log_likelihood()
        reference = az.loo(plain.idata, var_name="dropout", pointwise=True)
        got = agg.loo()  # computes the log-likelihood group on demand

        assert got.kind == "loo"
        assert got.scale == "log"
        assert got.n_data_points == reference.n_data_points == len(cohort_data)
        assert got.n_samples == reference.n_samples
        assert got.good_k == reference.good_k
        assert got.warning == reference.warning
        for attr in ("elpd", "se", "p"):
            np.testing.assert_allclose(
                getattr(got, attr), getattr(reference, attr), rtol=1e-8
            )
        assert got.elpd_i.dims == ("customer_id",)
        np.testing.assert_array_equal(
            got.elpd_i["customer_id"], cohort_data["customer_id"]
        )
        np.testing.assert_allclose(
            got.elpd_i.values, reference.elpd_i.values, rtol=1e-8
        )
        np.testing.assert_allclose(
            got.pareto_k.values, reference.pareto_k.values, rtol=1e-8
        )

        # the result plugs straight into az.compare
        comparison = az.compare({"customer": reference, "aggregated": got})
        assert set(comparison.index) == {"customer", "aggregated"}
        assert comparison["elpd_diff"].abs().max() < 1e-6

        # plain models use az.loo directly; az.loo on the aggregated idata is per configuration
        np.testing.assert_allclose(plain.loo().elpd, reference.elpd)
        assert (
            az.loo(agg.idata, var_name="dropout").n_data_points
            == agg.aggregation.n_unique
        )

    @pytest.mark.slow
    def test_posterior_agrees_within_mcse(self, cohort_data):
        """A5: independent MCMC fits agree within Monte Carlo error."""
        import arviz_stats as azs

        plain, agg = self._pair(cohort_data)
        kwargs = dict(chains=2, draws=400, tune=400, progressbar=False, random_seed=123)
        plain.fit(data=cohort_data, **kwargs)
        agg.fit(data=cohort_data, **{**kwargs, "random_seed": 321})

        for name in ("alpha", "beta", "phi", "kappa"):
            s_plain = azs.summary(plain.idata, var_names=[name])
            s_agg = azs.summary(agg.idata, var_names=[name])
            diff = np.abs(s_plain["mean"].to_numpy() - s_agg["mean"].to_numpy())
            tol = 4 * np.hypot(
                s_plain["mcse_mean"].to_numpy(), s_agg["mcse_mean"].to_numpy()
            )
            assert np.all(diff <= tol), f"{name}: {diff} > {tol}"
