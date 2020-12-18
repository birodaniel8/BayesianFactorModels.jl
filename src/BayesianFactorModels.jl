module BayesianFactorModels

using LinearAlgebra, MAT, LinearAlgebra, Distributions, Polynomials, SpecialFunctions, StatsBase, Random, Test

greet() = print("BayesianFactorModels Julia package")

export sampling_β, sampling_σ², sampling_factor, sampling_factor_loading,
       mcmc_sampling,
       LinearModel, LinearFactorModel,
       dgp_normal

# Add sampling functions:
include("sampling/sampling_beta.jl")
include("sampling/sampling_sigma_squared.jl")
include("sampling/sampling_factor.jl")
include("sampling/sampling_factor_loading.jl")

# Add data generating processes:
include("dgp/dgp_factor_model.jl")

# Add models:
include("models/models.jl")

# Add MCMC sampling:
include("mcmc/mcmc.jl")
include("mcmc/utils.jl")

end # module
