"""
    sampling_factor_loading(y, f, β_prior, V_prior, Σ)

Sampling the \$\\beta\$ factor loadings of the normal factor model with independent normal-gamma priors.

It creates an (m x k) lower triangular matrix of factor loadings by taking sample row-by-row from a posterior distribution,
which is \$N(\\beta_{posterior,1:i},V_{posterior,1:i,1:i})1(\\beta_i>0)\$ if \$i\\le k\$, where

\$V_{posterior,1:i,1:i} = (V_{prior,1:i,1:i}^{-1} + f_{1:i}'\\Sigma_{1:i,1:i}^{-1}f_{1:i})^{-1}\$

\$\\beta_{posterior,1:i} = V_{posterior,1:i,1:i}(V_{prior,1:i,1:i}^{-1}\\beta_{prior,1:i} + f_{1:i}'\\Sigma_{1:i,1:i}^{-1}y)\$

and \$N(\\beta_{posterior},V_{posterior})\$ else, where

\$V_{posterior} = (V_{prior}^{-1} + f'\\Sigma^{-1}f)^{-1}\$

\$\\beta_{posterior} = V_{posterior}(V_{prior}^{-1}\\beta_{prior} + f'\\Sigma^{-1}y)\$

## Arguments
- `y::Array`: (T x m) matrix of observations
- `f::Array`: (T x k) estimated factor values
- `β_prior::Union{Number, Array}`: (m x k) mean of the prior distribution of factor loadings (\$\\beta_{prior}\$)
- `V_prior::Union{Number, Array}`: (m x k) variance of the prior distributions (\$V_{prior}\$)
- `Σ::Union{Vector, AbstractArray}`: (m x m x (T)) error covariance matrix (\$\\Sigma\$)

## Returns
- `sampled_β::Array`: (m x k) sampled \$\\beta\$ factor loadings

## Note
- all factor loadings are treated as independent random variables and we have independent \$N(\\beta_{i,j},V_{i,j})\$ priors for each of them.
- If `β_prior` is given as a number, an (m x k) array is created and filled by the given value
- If `V_prior` is given as a number, an (m x k) array is created and filled by the given value
- If `Σ` is given as a vector, an (m x m) diagonal matrix is created having the input values in the diagonal
"""
function sampling_factor_loading(y::Array, 
                                 f::Array, 
                                 β_prior::Union{Number, Array}, 
                                 V_prior::Union{Number, Array}, 
                                 Σ::Union{Vector, AbstractArray}
                                 )::Array
    m = size(y, 2)
    k = size(f, 2)
    # Transform prior inputs to the right format:
    β_prior = isa(β_prior, Number) ? ones(m, k) * β_prior : β_prior  # Number to array
    V_prior = isa(V_prior, Number) ? ones(m, k) * V_prior : V_prior  # Number to array
    Σ = isa(Σ, Vector) ? Diagonal(Σ) : Σ  # Vector to array

    # Sampling factor loading row by row (creating a lower triangular matrix with positive diagonal elements):
    sampled_β = zeros(m, k)
    for i = 1:m
        error_variance = size(Σ, 3) > 1 ? Σ[i, i, :] : Σ[i, i]
        if i <= k
            sampled_β[i, 1:i] = sampling_β(y[:, i], f[:, 1:i], β_prior[i, 1:i], V_prior[i, 1:i], error_variance, 
                                           last_truncated=true)
        else
            sampled_β[i, :] = sampling_β(y[:, i], f, β_prior[i, :], V_prior[i, :], error_variance)
        end
    end
    return sampled_β
end
