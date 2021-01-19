module BayesianFactorModels

using LinearAlgebra, MAT, LinearAlgebra, Distributions, Polynomials, SpecialFunctions, StatsBase, Random, Test

greet() = print("BayesianFactorModels Julia package")

export sampling_β, sampling_σ², sampling_factor, sampling_factor_loading, sampling_df, sampling_mixture_scale,
       sampling_carter_kohn, sampling_factor_dynamic, sampling_stochastic_volatility, 
       mcmc_sampling,
       kalman_filter,
       LinearModel, LinearModelT, LinearFactorModel, DynamicLinearFactorModel, StochasticVolatilityModel, 
       DynamicLinearFactorSVModel, 
       NormalFactorModelDGP, DynamicFactorModelDGP, 
       dgp_generate

# Add sampling functions:
include("sampling/sampling_beta.jl")
include("sampling/sampling_sigma_squared.jl")
include("sampling/sampling_factor.jl")
include("sampling/sampling_factor_loading.jl")
include("sampling/sampling_df.jl")
include("sampling/sampling_mixture_scale.jl")
include("sampling/sampling_carter_kohn.jl")
include("sampling/sampling_factor_dynamic.jl")
include("sampling/sampling_stochastic_volatility.jl")

# Add filters:
include("filtering/kalman_filter.jl")

# Add data generating processes:
include("dgp/dgp_models.jl")
include("dgp/dgp_generate.jl")

# Add models:
include("models/models.jl")

# Add MCMC sampling:
include("mcmc/mcmc.jl")
include("mcmc/utils.jl")

end # module
