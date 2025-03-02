```mermaid
---
title: pymc_marketing
---
classDiagram
    class MMMPlotSuite {
        - __init__(self, idata) None
        - _init_subplots(self, n_subplots, ncols, width_per_col, height_per_row) tuple[Figure, NDArray[Axes]]
        - _build_subplot_title(self, dims, combo, fallback_title) str
        - _get_additional_dim_combinations(self, data, variable, ignored_dims) tuple[list[str], list[tuple]]
        - _reduce_and_stack(self, data, dims_to_ignore) xr.DataArray
        - _compute_ci(self, data, ci, sample_dim) tuple[xr.DataArray, xr.DataArray, xr.DataArray]
        - _get_posterior_predictive_data(self, idata) xr.Dataset
        + posterior_predictive(self, var, idata) tuple[Figure, NDArray[Axes]]
        + contributions_over_time(self, var, ci) tuple[Figure, NDArray[Axes]]
        + saturation_curves_scatter(self) tuple[Figure, NDArray[Axes]]
    }

    class ConvMode {
        + str After
        + str Before
        + str Overlap
    }

    class WeibullType {
        + str PDF
        + str CDF
    }

    class TanhSaturationParameters {
        + pt.TensorLike b
        + pt.TensorLike c
        + baseline(self, x0) "TanhSaturationBaselinedParameters"
    }

    class TanhSaturationBaselinedParameters {
        + pt.TensorLike x0
        + pt.TensorLike gain
        + pt.TensorLike r
        + debaseline(self) TanhSaturationParameters
        + rebaseline(self, x1) "TanhSaturationBaselinedParameters"
    }

    class Basis {
        + str prefix
        + str lookup_name
        + sample_curve(self, parameters, days) xr.DataArray
    }

    class EventEffect {
        + InstanceOf[Basis] basis
        + InstanceOf[Prior] effect_size
        + str | tuple[str, ...] dims
        - _dims_to_tuple(self)
        - _validate_dims(self)
        + apply(self, X, name) TensorVariable
        + to_dict(self) dict
        + @classmethod from_dict(cls, data) "EventEffect"
    }

    class GaussianBasis {
        + str lookup_name
        + function(self, x, sigma) TensorVariable
        + dict default_priors
    }

    class ValidateTargetColumn {
        + validate_target(self, data) None
    }

    class ValidateDateColumn {
        + str date_column
        + validate_date_col(self, data) None
    }

    class ValidateChannelColumns {
        + list[str] | tuple[str] channel_columns
        + validate_channel_columns(self, data) None
    }

    class ValidateControlColumns {
        + list[str] | None control_columns
        + validate_control_columns(self, data) None
    }

    class BaseMMM {
        - str _model_name
        - str _model_type
        + str version
        - __init__(self, date_column, channel_columns, adstock, saturation, time_varying_intercept, time_varying_media, model_config, sampler_config, validate_data, control_columns, yearly_seasonality, adstock_first, dag, treatment_nodes, outcome_node) None
        + default_sampler_config(self) dict
        + output_var(self) Literal["y"]
        - _generate_and_preprocess_model_data(self, X, y) None
        + create_idata_attrs(self) dict[str, str]
        + forward_pass(self, x) pt.TensorVariable
        + build_model(self, X, y, **kwargs) None
        + default_model_config(self) dict
        + channel_contributions_forward_pass(self, channel_data, disable_logger_stdout) npt.NDArray
        - _serializable_model_config(self) dict[str, Any]
        + @classmethod attrs_to_init_kwargs(cls, attrs) dict[str, Any]
        - _data_setter(self, X, y) None
        - @classmethod _model_config_formatting(cls, model_config) dict
    }

    class MMM {
        - str _model_type
        + str version
        + channel_contributions_forward_pass(self, channel_data, disable_logger_stdout) npt.NDArray
        + get_channel_contributions_forward_pass_grid(self, start, stop, num) DataArray
        + plot_channel_parameter(self, param_name, **plt_kwargs) plt.Figure
        + get_ts_contribution_posterior(self, var_contribution, original_scale) DataArray
        + plot_components_contributions(self, original_scale, **plt_kwargs) plt.Figure
        + plot_channel_contributions_grid(self, start, stop, num, absolute_xrange, **plt_kwargs) plt.Figure
        + new_spend_contributions(self, spend, one_time, spend_leading_up, prior, original_scale, **sample_posterior_predictive_kwargs) DataArray
        + plot_new_spend_contributions(self, spend_amount, one_time, lower, upper, ylabel, idx, channels, prior, original_scale, ax, **sample_posterior_predictive_kwargs) plt.Axes
        - _validate_data(self, X, y)
        - _channel_scales(self) np.ndarray
        - _channel_map_scales(self) dict
        + format_recovered_transformation_parameters(self, quantile) dict[str, dict[str, dict[str, float]]]
        - _plot_response_curve_fit(self, ax, channel, color_index, xlim_max, label, quantile_lower, quantile_upper) None
        + plot_direct_contribution_curves(self, show_fit, same_axes, xlim_max, channels, quantile_lower, quantile_upper, method) plt.Figure
        + sample_posterior_predictive(self, X, extend_idata, combined, include_last_observations, original_scale, **sample_posterior_predictive_kwargs) DataArray
        + add_lift_test_measurements(self, df_lift_test, dist, name) None
        - _create_synth_dataset(self, df, date_column, allocation_strategy, channels, controls, target_col, time_granularity, time_length, lag, noise_level) pd.DataFrame
        + sample_response_distribution(self, allocation_strategy, time_granularity, num_periods, noise_level) az.InferenceData
        - _set_predictors_for_optimization(self, num_periods) pm.Model
        + optimize_budget(self, budget, num_periods, budget_bounds, response_variable, utility_function, constraints, default_constraints, **minimize_kwargs) tuple[DataArray, OptimizeResult]
        + allocate_budget_to_maximize_response(self, budget, time_granularity, num_periods, budget_bounds, custom_constraints, noise_level, utility_function, **minimize_kwargs) az.InferenceData
        + plot_budget_allocation(self, samples, figsize, ax, original_scale) tuple[plt.Figure, plt.Axes]
        + plot_allocated_contribution_by_channel(self, samples, lower_quantile, upper_quantile, original_scale) plt.Figure
    }

    class MinimizeException {
        - __init__(self, message) None
    }

    class OptimizerCompatibleModelWrapper {
        + Any adstock
        - Any _channel_scales
        + InferenceData idata
        - _set_predictors_for_optimization(self, num_periods) Model
    }

    class BudgetOptimizer {
        + int num_periods
        + InstanceOf[OptimizerCompatibleModelWrapper] mmm_model
        + str response_variable
        + UtilityFunctionType utility_function
        + DataArray | None budgets_to_optimize
        + Sequence[Constraint] custom_constraints
        + bool default_constraints
        + model_config
        + ClassVar[dict] DEFAULT_MINIMIZE_KWARGS
        - __init__(self, **data) None
        + set_constraints(self, constraints, default) None
        - _replace_channel_data_by_optimization_variable(self, model) Model
        + extract_response_distribution(self, response_variable) pt.TensorVariable
        - _compile_objective_and_grad(self)
        + allocate_budget(self, total_budget, budget_bounds, minimize_kwargs, return_if_fail) tuple[DataArray, OptimizeResult]
    }

    class SaturationTransformation {
        + str prefix
        + sample_curve(self, parameters, max_value) xr.DataArray
    }

    class LogisticSaturation {
        + str lookup_name
        + function(self, x, lam, beta)
        + dict default_priors
    }

    class InverseScaledLogisticSaturation {
        + str lookup_name
        + function(self, x, lam, beta)
        + dict default_priors
    }

    class TanhSaturation {
        + str lookup_name
        + function(self, x, b, c)
        + dict default_priors
    }

    class TanhSaturationBaselined {
        + str lookup_name
        + function(self, x, x0, gain, r, beta)
        + dict default_priors
    }

    class MichaelisMentenSaturation {
        + str lookup_name
        + function(self, x, alpha, lam)
        + dict default_priors
    }

    class HillSaturation {
        + str lookup_name
        + function(self, x, slope, kappa, beta)
        + dict default_priors
    }

    class HillSaturationSigmoid {
        + str lookup_name
        + function
        + dict default_priors
    }

    class RootSaturation {
        + str lookup_name
        + function(self, x, alpha, beta)
        + dict default_priors
    }

    class ParameterPriorException {
        - __init__(self, priors, parameters) None
    }

    class MissingDataParameter {
        - __init__(self) None
    }

    class Transformation {
        + str prefix
        + dict[str, Prior] default_priors
        + Any function
        + str lookup_name
        - __init__(self, priors, prefix) None
        - __repr__(self) str
        + set_dims_for_all_priors(self, dims)
        + to_dict(self) dict[str, Any]
        - __eq__(self, other) bool
        + function_priors(self) dict[str, Prior]
        + function_priors(self, priors) None
        + update_priors(self, priors) None
        + model_config(self) dict[str, Any]
        - _checks(self) None
        - _has_all_attributes(self) None
        - _has_defaults_for_all_arguments(self) None
        - _function_works_on_instances(self) None
        + variable_mapping(self) dict[str, str]
        + combined_dims(self) tuple[str, ...]
        - _infer_output_core_dims(self) tuple[str, ...]
        - _create_distributions(self, dims) dict[str, TensorVariable]
        + sample_prior(self, coords, **sample_prior_predictive_kwargs) xr.Dataset
        + plot_curve(self, curve, subplot_kwargs, sample_kwargs, hdi_kwargs, axes, same_axes, colors, legend, sel_to_string) tuple[Figure, npt.NDArray[Axes]]
        - _sample_curve(self, var_name, parameters, x, coords) xr.DataArray
        + plot_curve_samples(self, curve, n, rng, plot_kwargs, subplot_kwargs, axes) tuple[Figure, npt.NDArray[Axes]]
        + plot_curve_hdi(self, curve, hdi_kwargs, plot_kwargs, subplot_kwargs, axes) tuple[Figure, npt.NDArray[Axes]]
        + apply(self, x, dims) TensorVariable
    }

    class DuplicatedTransformationError {
        - __init__(self, name, lookup_name) None
    }

    class AdstockTransformation {
        + str prefix
        + str lookup_name
        - __init__(self, l_max, normalize, mode, priors, prefix) None
        - __repr__(self) str
        + to_dict(self) dict
        + sample_curve(self, parameters, amount) xr.DataArray
    }

    class GeometricAdstock {
        + str lookup_name
        + function(self, x, alpha)
        + dict default_priors
    }

    class DelayedAdstock {
        + str lookup_name
        + function(self, x, alpha, theta)
        + dict default_priors
    }

    class WeibullPDFAdstock {
        + str lookup_name
        + function(self, x, lam, k)
        + dict default_priors
    }

    class WeibullCDFAdstock {
        + str lookup_name
        + function(self, x, lam, k)
        + dict default_priors
    }

    class MaxAbsScaleTarget {
        + Pipeline target_transformer
        + max_abs_scale_target_data(self, data) np.ndarray | pd.Series
    }

    class MaxAbsScaleChannels {
        + list[str] | tuple[str] channel_columns
        + max_abs_scale_channel_data(self, data) pd.DataFrame
    }

    class StandardizeControls {
        + list[str] control_columns
        + standardize_control_data(self, data) pd.DataFrame
    }

    class CausalGraphModel {
        - __init__(self, causal_model, treatment, outcome) None
        + @classmethod build_graphical_model(cls, graph, treatment, outcome) "CausalGraphModel"
        + get_backdoor_paths(self) list[list[str]]
        + get_unique_adjustment_nodes(self) list[str]
        + compute_adjustment_sets(self, channel_columns, control_columns) list[str] | None
    }

    class LinearTrend {
        + InstanceOf[dict[str, Prior]] priors
        + tuple[str] | InstanceOf[Dims] | str | None dims
        + int n_changepoints
        + bool include_intercept
        - _dims_is_tuple(self) Self
        - _priors_are_set(self) Self
        - _check_parameters(self) Self
        - _check_dims_are_subsets(self) Self
        + default_priors(self) dict[str, Prior]
        + apply(self, t) TensorVariable
        + sample_prior(self, coords, **sample_prior_predictive_kwargs) xr.Dataset
        + sample_curve(self, parameters, max_value) xr.DataArray
        + plot_curve(self, curve, subplot_kwargs, sample_kwargs, hdi_kwargs, include_changepoints, axes, same_axes, colors, legend, sel_to_string) tuple[Figure, npt.NDArray[Axes]]
    }

    class MuEffect {
        + create_data(self, mmm) None
        + create_effect(self, mmm) pt.TensorVariable
        + set_data(self, mmm, model, X) None
    }

    class MMM {
        - str _model_type
        + str version
        - __init__(self, date_column, channel_columns, target_column, adstock, saturation, time_varying_intercept, time_varying_media, dims, model_config, sampler_config, control_columns, yearly_seasonality, adstock_first) None
        - _check_compatible_media_dims(self) None
        + default_sampler_config(self) dict
        - _data_setter(self, X, y)
        + add_events(self, df_events, prefix, effect) None
        - _serializable_model_config(self) dict[str, Any]
        + create_idata_attrs(self) dict[str, str]
        - @classmethod _model_config_formatting(cls, model_config) dict
        + @classmethod attrs_to_init_kwargs(cls, attrs) dict[str, Any]
        + plot(self) MMMPlotSuite
        + default_model_config(self) dict
        + output_var(self) Literal["y"]
        - _validate_idata_exists(self) None
        - _validate_dims_in_multiindex(self, index, dims, date_column) list[str]
        - _validate_dims_in_dataframe(self, df, dims, date_column) list[str]
        - _validate_metrics(self, data, metric_list) list[str]
        - _process_multiindex_series(self, series, date_column, valid_dims, metric_coordinate_name) xr.Dataset
        - _process_dataframe(self, df, date_column, valid_dims, valid_metrics, metric_coordinate_name) xr.Dataset
        - _create_xarray_from_pandas(self, data, date_column, dims, metric_list, metric_coordinate_name) xr.Dataset
        - _generate_and_preprocess_model_data(self, X, y)
        + forward_pass(self, x, dims) pt.TensorVariable
        - _compute_scales(self) None
        + get_scales_as_xarray(self) dict[str, xr.DataArray]
        - _validate_model_was_built(self) None
        - _validate_contribution_variable(self, var) None
        + add_original_scale_contribution_variable(self, var) None
        + build_model(self, X, y, **kwargs) None
        - _posterior_predictive_data_transformation(self, X, y, include_last_observations) xr.Dataset
        - _set_xarray_data(self, dataset_xarray, clone_model) pm.Model
        + fit(self, X, y, progressbar, random_seed, **kwargs) az.InferenceData
        + sample_posterior_predictive(self, X, extend_idata, combined, include_last_observations, clone_model, **sample_posterior_predictive_kwargs) xr.DataArray
    }

    class Constraint {
        - __init__(self, key, constraint_type, constraint_fun) None
    }

    class MediaTransformation {
        + AdstockTransformation adstock
        + SaturationTransformation saturation
        + bool adstock_first
        + Dims | None dims
        - __post_init__(self)
        - _check_compatible_dims(self)
        - __call__(self, x)
        + to_dict(self) dict
        + @classmethod from_dict(cls, data) MediaTransformation
    }

    class MediaConfig {
        + str name
        + list[str] columns
        + MediaTransformation media_transformation
        + to_dict(self) dict
        + @classmethod from_dict(cls, data) MediaConfig
    }

    class MediaConfigList {
        - __init__(self, media_configs) None
        - __eq__(self, other) bool
        - __getitem__(self, key) MediaConfig
        + media_values(self) list[str]
        + to_dict(self) list[dict]
        + @classmethod from_dict(cls, data) MediaConfigList
        - __call__(self, x) pt.TensorVariable
    }

    class FourierBase {
        + int n_order
        + float days_in_period
        + str prefix
        + InstanceOf[Prior] | InstanceOf[VariableFactory] prior
        + str | None variable_name
        + model_post_init(self, __context) None
        - _check_variable_name(self) Self
        - _check_prior_has_right_dimensions(self) Self
        + serialize_prior(prior) dict[str, Any]
        + nodes(self) list[str]
        + get_default_start_date(self, start_date) str | datetime.datetime
        - _get_default_start_date(self) datetime.datetime
        - _get_days_in_period(self, dates) pd.Index
        + apply(self, dayofperiod, result_callback) pt.TensorVariable
        + sample_prior(self, coords, **kwargs) xr.Dataset
        + sample_curve(self, parameters, use_dates, start_date) xr.DataArray
        + plot_curve(self, curve, subplot_kwargs, sample_kwargs, hdi_kwargs, axes, same_axes, colors, legend, sel_to_string) tuple[plt.Figure, npt.NDArray[plt.Axes]]
        + plot_curve_hdi(self, curve, hdi_kwargs, subplot_kwargs, plot_kwargs, axes) tuple[plt.Figure, npt.NDArray[plt.Axes]]
        + plot_curve_samples(self, curve, n, rng, plot_kwargs, subplot_kwargs, axes) tuple[plt.Figure, npt.NDArray[plt.Axes]]
        + to_dict(self) dict[str, Any]
        + @classmethod from_dict(cls, data) Self
    }

    class YearlyFourier {
        + float days_in_period
        - _get_default_start_date(self) datetime.datetime
        - _get_days_in_period(self, dates) pd.Index
    }

    class MonthlyFourier {
        + float days_in_period
        - _get_default_start_date(self) datetime.datetime
        - _get_days_in_period(self, dates) pd.Index
    }

    class WeeklyFourier {
        + float days_in_period
        - _get_default_start_date(self) datetime.datetime
        - _get_days_in_period(self, dates) pd.Index
    }

    class MMMModelBuilder {
        + pm.Model model
        + str _model_type
        + str version
        - __init__(self, date_column, channel_columns, model_config, sampler_config) None
        + methods(self) list[Any]
        + validation_methods(self) tuple[list[Callable[["MMMModelBuilder", pd.DataFrame | pd.Series | np.ndarray], None]], list[Callable[["MMMModelBuilder", pd.DataFrame | pd.Series | np.ndarray], None]]]
        + validate(self, target, data) None
        + preprocessing_methods(self) tuple[list[Callable[["MMMModelBuilder", pd.DataFrame | pd.Series | np.ndarray], pd.DataFrame | pd.Series | np.ndarray]], list[Callable[["MMMModelBuilder", pd.DataFrame | pd.Series | np.ndarray], pd.DataFrame | pd.Series | np.ndarray]]]
        + preprocess(self, target, data) pd.DataFrame | pd.Series | np.ndarray
        + get_target_transformer(self) Pipeline
        - _get_group_predictive_data(self, group, original_scale) Dataset
        - _get_prior_predictive_data(self, original_scale) Dataset
        - _get_posterior_predictive_data(self, original_scale) Dataset
        - _add_mean_to_plot(self, ax, group, original_scale, color, linestyle, **kwargs) plt.Axes
        - _add_hdi_to_plot(self, ax, group, original_scale, hdi_prob, color, alpha, **kwargs) plt.Axes
        - _add_gradient_to_plot(self, ax, group, original_scale, n_percentiles, palette, **kwargs) plt.Axes
        - _plot_group_predictive(self, group, original_scale, hdi_list, add_mean, add_gradient, ax, **plt_kwargs) plt.Figure
        + plot_prior_predictive(self, original_scale, hdi_list, add_mean, add_gradient, ax, **plt_kwargs) plt.Figure
        + plot_posterior_predictive(self, original_scale, hdi_list, add_mean, add_gradient, ax, **plt_kwargs) plt.Figure
        + get_errors(self, original_scale) DataArray
        + plot_errors(self, original_scale, ax, **plt_kwargs) plt.Figure
        - _format_model_contributions(self, var_contribution) DataArray
        + plot_components_contributions(self, **plt_kwargs) plt.Figure
        + compute_channel_contribution_original_scale(self, prior) DataArray
        + compute_mean_contributions_over_time(self, original_scale) pd.DataFrame
        + plot_grouped_contribution_breakdown_over_time(self, stack_groups, original_scale, area_kwargs, **plt_kwargs) plt.Figure
        + get_channel_contributions_share_samples(self, prior) DataArray
        + plot_channel_contribution_share_hdi(self, hdi_prob, prior, **plot_kwargs) plt.Figure
        - _process_decomposition_components(self, data) pd.DataFrame
        + plot_prior_vs_posterior(self, var_name, alphabetical_sort, figsize) plt.Figure
        + plot_waterfall_components_decomposition(self, original_scale, figsize, **kwargs) plt.Figure
    }

    class BaseValidateMMM

    class UnalignedValuesError {
        - __init__(self, unaligned_values) None
    }

    class MissingValueError {
        - __init__(self, missing_values, required_values) None
    }

    class NonMonotonicError

    class CovFunc {
        + str ExpQuad
        + str Matern52
        + str Matern32
    }

    class HSGPBase {
        + int m
        + InstanceOf[TensorVariable] | InstanceOf[np.ndarray] | None X
        + float | None X_mid
        + Dims dims
        + register_data(self, X) Self
        - _register_user_input_X(self) Self
        - _dim_is_at_least_one(self) Self
        + create_variable(self, name) TensorVariable
        + to_dict(self) dict
        + sample_prior(self, coords, **sample_prior_predictive_kwargs) xr.Dataset
        + plot_curve(self, curve, subplot_kwargs, sample_kwargs, hdi_kwargs, axes, same_axes, colors, legend, sel_to_string) tuple[Figure, npt.NDArray[Axes]]
    }

    class HSGP {
        + InstanceOf[Prior] | float ls
        + InstanceOf[Prior] | float eta
        + float L
        + bool centered
        + bool drop_first
        + CovFunc cov_func
        - _ls_is_scalar_prior(self) Self
        - _eta_is_scalar_prior(self) Self
        + @classmethod parameterize_from_data(cls, X, dims, X_mid, eta_mass, eta_upper, ls_lower, ls_upper, ls_mass, cov_func, centered, drop_first) HSGP
        + create_variable(self, name) TensorVariable
        + @classmethod from_dict(cls, data) HSGP
    }

    class PeriodicCovFunc {
        + str Periodic
    }

    class HSGPPeriodic {
        + InstanceOf[Prior] | float ls
        + InstanceOf[Prior] | float scale
        + PeriodicCovFunc cov_func
        + float period
        - _ls_is_scalar_prior(self) Self
        - _scale_is_scalar_prior(self) Self
        + create_variable(self, name) TensorVariable
        + @classmethod from_dict(cls, data) HSGPPeriodic
    }

    class SoftPlusHSGP {
        + create_variable(self, name) TensorVariable
    }

    class DifferentModelError

    class ModelBuilder {
        + str _model_type
        + str version
        + pd.DataFrame | None X
        + pd.Series | np.ndarray | None y
        - __init__(self, model_config, sampler_config) None
        - _validate_data(self, X, y)
        - _data_setter(self, X, y) None
        + output_var(self) str
        + default_model_config(self) dict
        + default_sampler_config(self) dict
        + build_model(self, X, y, **kwargs) None
        + create_idata_attrs(self) dict[str, str]
        + set_idata_attrs(self, idata) az.InferenceData
        + save(self, fname) None
        - @classmethod _model_config_formatting(cls, model_config) dict
        + @classmethod attrs_to_init_kwargs(cls, attrs) dict[str, Any]
        + build_from_idata(self, idata) None
        + @classmethod load_from_idata(cls, idata) "ModelBuilder"
        + @classmethod load(cls, fname)
        + create_fit_data(self, X, y) xr.Dataset
        + fit(self, X, y, progressbar, random_seed, **kwargs) az.InferenceData
        + fit_result(self) xr.Dataset
        + fit_result(self, res) None
        + predict(self, X, extend_idata, **kwargs) np.ndarray
        + sample_prior_predictive(self, X, y, samples, extend_idata, combined, **kwargs)
        + sample_posterior_predictive(self, X, extend_idata, combined, **sample_posterior_predictive_kwargs)
        - _serializable_model_config(self) dict[str, int | float | dict]
        + predict_proba(self, X, extend_idata, combined, **kwargs) xr.DataArray
        + predict_posterior(self, X, extend_idata, combined, **kwargs) xr.DataArray
        + id(self) str
        + graphviz(self, **kwargs)
        + prior
        + prior_predictive
        + posterior
        + posterior_predictive
        + predictions
    }

    class MVITS {
        + str _model_type
        + str version
        - __init__(self, existing_sales, saturated_market, model_config, sampler_config) None
        - _distribution_checks(self)
        + create_idata_attrs(self) dict[str, str]
        + @classmethod attrs_to_init_kwargs(cls, attrs) dict[str, Any]
        + default_model_config(self) dict
        + inform_default_prior(self, data) Self
        + default_sampler_config(self) dict
        + output_var(self) str
        - _serializable_model_config(self) dict[str, int | float | dict]
        - _generate_and_preprocess_model_data(self, X, y) None
        + build_model(self, X, y, **kwargs) None
        - _data_setter(self, X, y) None
        + calculate_counterfactual(self, random_seed) None
        + sample(self, X, y, random_seed, sample_prior_predictive_kwargs, fit_kwargs, sample_posterior_predictive_kwargs) Self
        + causal_impact(self, variable) DataArray
        + plot_fit(self, variable, plot_total_sales, ax)
        + plot_counterfactual(self, variable, plot_total_sales, ax)
        + plot_causal_impact_sales(self, variable, ax)
        + plot_causal_impact_market_share(self, variable, ax)
        + plot_data(self, plot_total_sales, ax)
        + predictions
    }

    class ModelConfigError

    class MMMWrapper {
        - __init__(self, model, predict_method, extend_idata, combined, include_last_observations, original_scale, var_names, **sample_kwargs) None
        + predict(self, context, model_input, params) Any
    }

    class HSGPKwargs {
        + int m
        + Annotated[float, Field(gt=0, description="\n                Extent of basis functions. Set this to reflect the expected range of in+out-of-sample data\n                (considering that time-indices are zero-centered).Default is `X_mid * 2` (identical to `c=2` in HSGP)\n                ")] | None L
        + float eta_lam
        + float ls_mu
        + float ls_sigma
        + InstanceOf[pm.gp.cov.Covariance] | str | None cov_func
    }

    class UnsupportedShapeError

    class UnsupportedDistributionError

    class UnsupportedParameterizationError

    class MuAlreadyExistsError {
        - __init__(self, distribution) None
    }

    class UnknownTransformError

    class VariableFactory {
        + tuple[str, ...] dims
        + create_variable(self, name) pt.TensorVariable
    }

    class Prior {
        + dict[str, dict[str, float]] non_centered_distributions
        + type[pm.Distribution] pymc_distribution
        + Callable[[pt.TensorLike], pt.TensorLike] | None pytensor_transform
        - __init__(self, distribution, *, dims, centered, transform, **parameters) None
        + distribution(self) str
        + distribution(self, distribution) None
        + transform(self) str | None
        + transform(self, transform) None
        + dims(self) Dims
        + dims(self, dims) None
        - __getitem__(self, key) Prior | Any
        - _checks(self) None
        - _parameters_are_at_least_subset_of_pymc(self) None
        - _convert_lists_to_numpy(self) None
        - _parameters_are_correct_type(self) None
        - _correct_non_centered_distribution(self) None
        - _unique_dims(self) None
        - _param_dims_work(self) None
        - __str__(self) str
        - __repr__(self) str
        - _create_parameter(self, param, value, name)
        - _create_centered_variable(self, name)
        - _create_non_centered_variable(self, name) pt.TensorVariable
        + create_variable(self, name) pt.TensorVariable
        + preliz(self)
        + to_dict(self) dict[str, Any]
        + to_json(self) dict[str, Any]
        + @classmethod from_dict(cls, data) Prior
        + @classmethod from_json(cls, json) Prior
        + constrain(self, lower, upper, mass, kwargs) Prior
        - __eq__(self, other) bool
        + sample_prior(self, coords, name, **sample_prior_predictive_kwargs) xr.Dataset
        - __deepcopy__(self, memo) Prior
        + deepcopy(self) Prior
        + to_graph(self)
        + create_likelihood_variable(self, name, mu, observed) pt.TensorVariable
    }

    class VariableNotFound

    class Censored {
        + InstanceOf[Prior] distribution
        + float | InstanceOf[pt.TensorVariable] lower
        + float | InstanceOf[pt.TensorVariable] upper
        - __post_init__(self) None
        + dims(self) tuple[str, ...]
        + dims(self, dims) None
        + create_variable(self, name) pt.TensorVariable
        + to_dict(self) dict[str, Any]
        + @classmethod from_dict(cls, data) Censored
        + sample_prior(self, coords, name, **sample_prior_predictive_kwargs) xr.Dataset
        + to_graph(self)
        + create_likelihood_variable(self, name, mu, observed) pt.TensorVariable
    }

    class Deserializer {
        + IsType is_type
        + Deserialize deserialize
    }

    class DeserializableError {
        - __init__(self, data) None
    }

    class BetaGeoBetaBinomModel {
        + str _model_type
        - __init__(self, data, *, model_config, sampler_config) None
        + default_model_config(self) ModelConfig
        + build_model(self) None
        - @staticmethod _logp(alpha, beta, gamma, delta, x, t_x, T) xarray.DataArray
        - _extract_predictive_variables(self, data, customer_varnames) xarray.Dataset
        + expected_purchases(self, data, *, future_t) xarray.DataArray
        + expected_probability_alive(self, data, *, future_t) xarray.DataArray
        + expected_purchases_new_customer(self, data, *, t) xarray.DataArray
        - _distribution_new_customers(self, data, *, T, random_seed, var_names, n_samples) xarray.Dataset
        + distribution_new_customer_dropout(self, data, *, random_seed) xarray.Dataset
        + distribution_new_customer_purchase_rate(self, data, *, random_seed) xarray.Dataset
        + distribution_new_customer_recency_frequency(self, data, *, T, random_seed, n_samples) xarray.Dataset
    }

    class ParetoNBDModel {
        + str _model_type
        - __init__(self, data, *, model_config, sampler_config) None
        + default_model_config(self) ModelConfig
        + build_model(self) None
        + fit(self, fit_method, **kwargs)
        - @staticmethod _logp(r, alpha, s, beta, x, t_x, T) xarray.DataArray
        - _extract_predictive_variables(self, data, customer_varnames) xarray.Dataset
        + expected_purchases(self, data, *, future_t) xarray.DataArray
        + expected_probability_alive(self, data, *, future_t) xarray.DataArray
        + expected_purchase_probability(self, data, *, n_purchases, future_t) xarray.DataArray
        + expected_purchases_new_customer(self, data, *, t) xarray.DataArray
        + distribution_new_customer(self, data, *, T, random_seed, var_names, n_samples) xarray.Dataset
        + distribution_new_customer_dropout(self, data, *, random_seed) xarray.Dataset
        + distribution_new_customer_purchase_rate(self, data, *, random_seed) xarray.Dataset
        + distribution_new_customer_recency_frequency(self, data, *, T, random_seed, n_samples) xarray.Dataset
    }

    class ModifiedBetaGeoModel {
        + str _model_type
        + build_model(self) None
        + expected_num_purchases(self, customer_id, t, frequency, recency, T) xarray.DataArray
        + expected_purchases(self, data, *, future_t) xarray.DataArray
        + expected_purchases_new_customer(self, data, *, t) xarray.DataArray
        + expected_probability_alive(self, data) xarray.DataArray
        + expected_probability_no_purchase(self, t, data) xarray.DataArray
        + distribution_new_customer(self, data, *, T, random_seed, var_names, n_samples) xarray.Dataset
    }

    class BetaGeoModel {
        + str _model_type
        - __init__(self, data, model_config, sampler_config) None
        + default_model_config(self) ModelConfig
        + build_model(self) None
        - _unload_params(self)
        - _extract_predictive_variables(self, data, customer_varnames) xarray.Dataset
        + expected_num_purchases(self, customer_id, t, frequency, recency, T) xarray.DataArray
        + expected_purchases(self, data, *, future_t) xarray.DataArray
        + expected_probability_alive(self, data) xarray.DataArray
        + expected_probability_no_purchase(self, t, data) xarray.DataArray
        + expected_num_purchases_new_customer(self, *args, **kwargs) xarray.DataArray
        + expected_purchases_new_customer(self, data, *, t) xarray.DataArray
        + distribution_new_customer(self, data, *, T, random_seed, var_names, n_samples) xarray.Dataset
        + distribution_new_customer_dropout(self, data, *, random_seed) xarray.Dataset
        + distribution_new_customer_purchase_rate(self, data, *, random_seed) xarray.Dataset
        + distribution_new_customer_recency_frequency(self, data, *, T, random_seed, n_samples) xarray.Dataset
    }

    class CLVModel {
        + str _model_type
        - __init__(self, data, *, model_config, sampler_config, non_distributions) None
        - @staticmethod _validate_cols(data, required_cols, must_be_unique, must_be_homogenous)
        - __repr__(self) str
        - _add_fit_data_group(self, data) None
        + fit(self, fit_method, **kwargs) az.InferenceData
        - _fit_mcmc(self, **kwargs) az.InferenceData
        - _fit_MAP(self, **kwargs) az.InferenceData
        - _fit_DEMZ(self, **kwargs) az.InferenceData
        - _fit_approx(self, method, **kwargs) az.InferenceData
        + @classmethod load(cls, fname)
        - @classmethod _build_with_idata(cls, idata)
        - _rename_posterior_variables(self)
        + thin_fit_result(self, keep_every)
        + default_sampler_config(self) dict
        - _serializable_model_config(self) dict
        + fit_summary(self, **kwargs)
        + output_var(self)
        - _data_setter(self)
    }

    class BaseGammaGammaModel {
        + distribution_customer_spend(self, data, random_seed) xarray.DataArray
        + expected_customer_spend(self, data) xarray.DataArray
        + distribution_new_customer_spend(self, n, random_seed) xarray.DataArray
        + expected_new_customer_spend(self) xarray.DataArray
        + expected_customer_lifetime_value(self, transaction_model, data, future_t, discount_rate, time_unit) xarray.DataArray
    }

    class GammaGammaModel {
        + str _model_type
        - __init__(self, data, model_config, sampler_config) None
        + default_model_config(self) ModelConfig
        + build_model(self) None
    }

    class GammaGammaModelIndividual {
        + str _model_type
        - __init__(self, data, model_config, sampler_config) None
        + default_model_config(self) dict
        + build_model(self) None
    }

    class ShiftedBetaGeoModelIndividual {
        + str _model_type
        - __init__(self, data, model_config, sampler_config) None
        + default_model_config(self) dict
        + build_model(self) None
        + distribution_customer_churn_time(self, customer_id, random_seed) DataArray
        - _distribution_new_customer(self, n, random_seed, var_names) Dataset
        + distribution_new_customer_churn_time(self, n, random_seed) DataArray
        + distribution_new_customer_theta(self, n, random_seed) DataArray
    }

    class ContNonContractRV {
        + str name
        + str signature
        + str dtype
        + tuple _print_name
        - __call__(self, lam, p, T, size, **kwargs)
        + @classmethod rng_fn(cls, rng, lam, p, T, size)
    }

    class ContNonContract {
        + rv_op
        + @classmethod dist(cls, lam, p, T, **kwargs)
        + logp(value, lam, p, T)
    }

    class ContContractRV {
        + str name
        + str signature
        + str dtype
        + tuple _print_name
        - __call__(self, lam, p, T, size, **kwargs)
        + @classmethod rng_fn(cls, rng, lam, p, T, size)
        - _supp_shape_from_params(*args, **kwargs)
    }

    class ContContract {
        + rv_op
        + @classmethod dist(cls, lam, p, T, **kwargs)
        + logp(value, lam, p, T)
    }

    class ParetoNBDRV {
        + str name
        + str signature
        + str dtype
        + tuple _print_name
        - __call__(self, r, alpha, s, beta, T, size, **kwargs)
        + @classmethod rng_fn(cls, rng, r, alpha, s, beta, T, size)
    }

    class ParetoNBD {
        + rv_op
        + @classmethod dist(cls, r, alpha, s, beta, T, **kwargs)
        + logp(value, r, alpha, s, beta, T)
    }

    class BetaGeoBetaBinomRV {
        + str name
        + str signature
        + str dtype
        + tuple _print_name
        - __call__(self, alpha, beta, gamma, delta, T, size, **kwargs)
        + @classmethod rng_fn(cls, rng, alpha, beta, gamma, delta, T, size) np.ndarray
    }

    class BetaGeoBetaBinom {
        + rv_op
        + @classmethod dist(cls, alpha, beta, gamma, delta, T, **kwargs)
        + logp(value, alpha, beta, gamma, delta, T)
    }

    class BetaGeoNBDRV {
        + str name
        + str signature
        + str dtype
        + tuple _print_name
        - __call__(self, a, b, r, alpha, T, size, **kwargs)
        + @classmethod rng_fn(cls, rng, a, b, r, alpha, T, size)
    }

    class BetaGeoNBD {
        + rv_op
        + @classmethod dist(cls, a, b, r, alpha, T, **kwargs)
        + logp(value, a, b, r, alpha, T)
    }

    class ModifiedBetaGeoNBDRV {
        + str name
        + str signature
        + str dtype
        + tuple _print_name
        - __call__(self, a, b, r, alpha, T, size, **kwargs)
        + @classmethod rng_fn(cls, rng, a, b, r, alpha, T, size)
    }

    class ModifiedBetaGeoNBD {
        + rv_op
        + @classmethod dist(cls, a, b, r, alpha, T, **kwargs)
        + logp(value, a, b, r, alpha, T)
    }

    ConvMode --|> str

    ConvMode --|> `enum.Enum`

    WeibullType --|> str

    WeibullType --|> `enum.Enum`

    TanhSaturationParameters --|> `typing.NamedTuple`

    TanhSaturationBaselinedParameters --|> `typing.NamedTuple`

    Basis --|> `pymc_marketing.mmm.components.base.Transformation`

    EventEffect --|> `pydantic.BaseModel`

    GaussianBasis --|> Basis

    BaseMMM --|> `pymc_marketing.mmm.base.BaseValidateMMM`

    MMM --|> `pymc_marketing.mmm.preprocessing.MaxAbsScaleTarget`

    MMM --|> `pymc_marketing.mmm.preprocessing.MaxAbsScaleChannels`

    MMM --|> `pymc_marketing.mmm.validating.ValidateControlColumns`

    MMM --|> BaseMMM

    MinimizeException --|> Exception

    OptimizerCompatibleModelWrapper --|> `typing.Protocol`

    BudgetOptimizer --|> `pydantic.BaseModel`

    SaturationTransformation --|> `pymc_marketing.mmm.components.base.Transformation`

    LogisticSaturation --|> SaturationTransformation

    InverseScaledLogisticSaturation --|> SaturationTransformation

    TanhSaturation --|> SaturationTransformation

    TanhSaturationBaselined --|> SaturationTransformation

    MichaelisMentenSaturation --|> SaturationTransformation

    HillSaturation --|> SaturationTransformation

    HillSaturationSigmoid --|> SaturationTransformation

    RootSaturation --|> SaturationTransformation

    ParameterPriorException --|> Exception

    MissingDataParameter --|> Exception

    DuplicatedTransformationError --|> Exception

    AdstockTransformation --|> `pymc_marketing.mmm.components.base.Transformation`

    GeometricAdstock --|> AdstockTransformation

    DelayedAdstock --|> AdstockTransformation

    WeibullPDFAdstock --|> AdstockTransformation

    WeibullCDFAdstock --|> AdstockTransformation

    LinearTrend --|> `pydantic.BaseModel`

    MuEffect --|> `typing.Protocol`

    MMM --|> `pymc_marketing.model_builder.ModelBuilder`

    FourierBase --|> `pydantic.BaseModel`

    YearlyFourier --|> FourierBase

    MonthlyFourier --|> FourierBase

    WeeklyFourier --|> FourierBase

    MMMModelBuilder --|> `pymc_marketing.model_builder.ModelBuilder`

    BaseValidateMMM --|> MMMModelBuilder

    BaseValidateMMM --|> `pymc_marketing.mmm.validating.ValidateTargetColumn`

    BaseValidateMMM --|> `pymc_marketing.mmm.validating.ValidateDateColumn`

    BaseValidateMMM --|> `pymc_marketing.mmm.validating.ValidateChannelColumns`

    UnalignedValuesError --|> Exception

    MissingValueError --|> KeyError

    NonMonotonicError --|> ValueError

    CovFunc --|> str

    CovFunc --|> `enum.Enum`

    HSGPBase --|> `pydantic.BaseModel`

    HSGP --|> HSGPBase

    PeriodicCovFunc --|> str

    PeriodicCovFunc --|> `enum.Enum`

    HSGPPeriodic --|> HSGPBase

    SoftPlusHSGP --|> HSGP

    DifferentModelError --|> Exception

    ModelBuilder --|> `abc.ABC`

    MVITS --|> `pymc_marketing.model_builder.ModelBuilder`

    ModelConfigError --|> Exception

    MMMWrapper --|> `mlflow.pyfunc.PythonModel`

    HSGPKwargs --|> `pydantic.BaseModel`

    UnsupportedShapeError --|> Exception

    UnsupportedDistributionError --|> Exception

    UnsupportedParameterizationError --|> Exception

    MuAlreadyExistsError --|> Exception

    UnknownTransformError --|> Exception

    VariableFactory --|> `typing.Protocol`

    VariableNotFound --|> Exception

    DeserializableError --|> Exception

    BetaGeoBetaBinomModel --|> `pymc_marketing.clv.models.basic.CLVModel`

    ParetoNBDModel --|> `pymc_marketing.clv.models.basic.CLVModel`

    ModifiedBetaGeoModel --|> `pymc_marketing.clv.models.BetaGeoModel`

    BetaGeoModel --|> `pymc_marketing.clv.models.basic.CLVModel`

    CLVModel --|> `pymc_marketing.model_builder.ModelBuilder`

    BaseGammaGammaModel --|> `pymc_marketing.clv.models.CLVModel`

    GammaGammaModel --|> BaseGammaGammaModel

    GammaGammaModelIndividual --|> BaseGammaGammaModel

    ShiftedBetaGeoModelIndividual --|> `pymc_marketing.clv.models.CLVModel`

    ContNonContractRV --|> `pytensor.tensor.random.op.RandomVariable`

    ContNonContract --|> `pymc.distributions.continuous.PositiveContinuous`

    ContContractRV --|> `pytensor.tensor.random.op.RandomVariable`

    ContContract --|> `pymc.distributions.continuous.PositiveContinuous`

    ParetoNBDRV --|> `pytensor.tensor.random.op.RandomVariable`

    ParetoNBD --|> `pymc.distributions.continuous.PositiveContinuous`

    BetaGeoBetaBinomRV --|> `pytensor.tensor.random.op.RandomVariable`

    BetaGeoBetaBinom --|> `pymc.distributions.distribution.Discrete`

    BetaGeoNBDRV --|> `pytensor.tensor.random.op.RandomVariable`

    BetaGeoNBD --|> `pymc.distributions.continuous.PositiveContinuous`

    ModifiedBetaGeoNBDRV --|> `pytensor.tensor.random.op.RandomVariable`

    ModifiedBetaGeoNBD --|> `pymc.distributions.continuous.PositiveContinuous`
```
