module BayesianFactorModels

using LinearAlgebra, MAT, LinearAlgebra, Distributions, Polynomials, SpecialFunctions, StatsBase, Random, Test

greet() = print("BayesianFactorModels Julia package")

export sampling_β, sampling_σ², 
       mcmc_sampling,
       LinearModel

# Add sampling functions:
include("sampling/sampling_beta.jl")
include("sampling/sampling_sigma_squared.jl")

# Add models:
include("models/models.jl")

# Add MCMC sampling:
include("mcmc/mcmc.jl")
end # module
