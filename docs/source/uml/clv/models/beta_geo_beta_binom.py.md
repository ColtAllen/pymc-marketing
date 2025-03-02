```mermaid
---
title: clv/models/beta_geo_beta_binom.py
---
classDiagram
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

    BetaGeoBetaBinomModel --|> `pymc_marketing.clv.models.basic.CLVModel`
```
