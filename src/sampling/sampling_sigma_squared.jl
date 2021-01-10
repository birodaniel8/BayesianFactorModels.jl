"""
    sampling_σ²(x, γ_prior = 1.5, δ_prior = 0.5)

Sampling error variances of the normal linear model with independent normal-gamma priors.

It iterates trough the \$\\epsilon_j\$ columns of the error matrix \$\\epsilon\$ and takes a sample from \$\\Gamma(\\gamma_{posterior},\\delta_{posterior})\$, where

\$\\gamma_{posterior} = \\frac{T}{2} + \\gamma_{prior}\$

\$\\delta_{posterior} = \\frac{\\epsilon_j'\\epsilon_j}{2} + \\delta_{prior}\$

## Arguments
- `ϵ::Array`: (T x m) error matrix of the regressions (conditioned on the coefficients)
- `γ_prior::Number`: Shape of the prior distribution (\$\\gamma_{prior}\$)
- `δ_prior::Number`: Scale of the prior distribution (\$\\delta_{prior}\$)

## Returns
- `sampled_σ²::Vector`: (m x 1) sampled error variances

## Note
The prior mean of the distribution is \$\\frac{\\gamma_{prior} - 1}{\\delta_{prior}}\$ (default = 1).
"""
function sampling_σ²(ϵ::Array, 
                     _γ_prior::Number=1.5, 
                     _δ_prior::Number=0.5;
                     γ_prior::Number=_γ_prior, δ_prior::Number=_δ_prior
                     )::Vector

    t = size(ϵ, 1)
    k = size(ϵ, 2)

    # Calculate posterior parameters:
    γ_posterior = ones(k) * t / 2 .+ γ_prior
    if k == 1
        δ_posterior = (ϵ' * ϵ) / 2 .+ δ_prior
    else
        δ_posterior = diag(ϵ' * ϵ) / 2 .+ δ_prior
    end

    # Sampling:
    sampled_σ² = zeros(k)
    for i = 1:k
        sampled_σ²[i] = rand(InverseGamma(γ_posterior[i], δ_posterior[i]))
    end
    return sampled_σ²
end
