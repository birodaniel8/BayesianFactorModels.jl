"""
    sampling_factor(y, β, Σ)

Sampling the factors of the normal factor model with independent normal-gamma priors.

It samples the factors by taking independent samples at each time \$t\$ from the distribution \$N(\\beta_f,V_f)\$, where

\$V_f = (I + \\beta'\\Sigma^{-1}\\beta)^{-1}\$

\$\\beta_f = V_f\\beta'\\Sigma^{-1}y)\$

## Arguments
- `y::Array`: (T x m) matrix of observations
- `β::Array`: (m x k) estimated factor loadings (\$\\beta\$)
- `Σ::Array`: (m x m) error covariance matrix (\$\\Sigma\$)

## Returns
- `sampled_factor::Array`: (T x k) sampled factors

## Note
- This approach is equivalent with having independent \$N(0,1)\$ priors on each factor elements.
- If `Σ` is given as a vector, an (m x m) diagonal matrix is created having the input values in the diagonal
"""
function sampling_factor(y::Array, 
                         β::Array, 
                         Σ::AbstractArray
                         )::Array
                         
    T = size(y, 1)
    k = size(β, 2)
    # Transform prior inputs to the right format:
    Σ = isa(Σ, Vector) ? Diagonal(Σ) : Σ  # Vector to array

    # Sampling factor values:
    sampled_factor = zeros(T, k)
    for t = 1:T
        factor_var = inv(I(k) + β' * inv(Σ) * β)
        factor_var = Matrix(Hermitian(factor_var))
        factor_mean = factor_var * β' * inv(Σ) * y[t, :]
        factor_sampled_t = rand(MultivariateNormal(factor_mean, factor_var))
        sampled_factor[t, 1:k] = factor_sampled_t
    end
    return sampled_factor
end
