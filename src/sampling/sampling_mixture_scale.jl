"""
    sampling_mixtrue_scale(ϵ, σ²=[0.01], ν=[30])

Sampling the mixture scale parameter of the normal linear model with independent normal-gamma priors and known
heteroscedasticity (ie t-errors)

It iterates trough all elements of the error matrix `ϵ` and takes a sample from \$\\Gamma(\\alpha_\\nu,\\beta_\\nu)\$, where

\$\\alpha_\\nu = \\frac{\\nu + 1}{2}\$

\$\\beta_\\nu = \\frac{2}{\\epsilon_{ij}^2/\\sigma^2} + \\nu\$

## Arguments
- `ϵ::Array`: (T x m) error matrix of the regressions (conditioned on the coefficients)
- `σ²::Vector`: (m) static variance of the error terms (\$\\sigma^2\$)
- `ν::Vector`: (m) vector of degree of freedom parameters (\$\\nu\$)

## Returns
- `sampled_λ::Vector`: (T x m) sampled mixture scale parameter
"""
function sampling_mixture_scale(ϵ::Array, 
                                _σ²::Vector=[0.01], 
                                _ν::Vector=[30];
                                σ²::Vector=_σ², ν::Vector=_ν
                                )::Array
    N = size(ϵ, 1)
    m = size(ϵ, 2)

    # Sampling:
    sampled_λ = zeros(N, m)
    for i = 1:m
        for j = 1:N
            α_λ = (ν[i] + 1) / 2
            β_λ = 2 / ((ϵ[j, i] ^ 2) / σ²[i] + ν[i])
            sampled_λ[j, i] = rand(Gamma(α_λ, β_λ))
        end
    end
    return sampled_λ
end
