"""
    sampling_β(y, x, β_prior = 0, V_prior = 1, σ² = 0.01; last_truncated = false, stationarity_check = false, constant_included = true, max_iterations = 10000)

Sampling the \$\\beta\$ coefficients of the normal linear model with independent normal-gamma priors.

It takes a sample from \$N(\\beta_{posterior},V_{posterior})\$, where

\$V_{posterior} = (V_{prior}^{-1} + X'\\sigma^{-2}X)^{-1}\$

\$\\beta_{posterior} = V_{posterior}(V_{prior}^{-1}\\beta_{prior} + X'\\sigma^{-2}y)\$

## Arguments
- `y::Vector`: (T x 1) dependent variable
- `x::Array`: (T x k) explanatory variables
- `β_prior::Union{Number, Vector}`: (k) mean of the prior distribution (\$\\beta_{prior}\$)
- `V_prior::Union{Number, AbstractArray}`: (k x k) covariance of the prior distribution (\$V_{prior}\$)
- `σ²::Union{Number, AbstractArray}`: error (co)variance (\$\\sigma^2\$)
- `last_truncated::Bool`: if true, the sample is taken from a multivariate normal distribution but the last element is truncated at 0
- `stationarity_check::Bool`: if true, the sampling is repeated until the sampled coefficients stand for a stationary AR(k) model (without intercept)
- `constant_included::Bool`: if true, the stationarity check is based on an AR(k-1) model (with intercept as a 1st variable)
- `max_iterations::Int`: maximum number of samples taken to sample coefficients for a stationary AR(k) model (without intercept)

## Returns
- `sampled_β::Vector`: (k) sampled \$\\beta\$ coefficients

## Note
- If `β_prior` is given as a number, a (k) length vector is created and filled by the given value
- If `V_prior` is given as a number or a vector, a (k x k) diagonal matrix is created having the input values in the diagonal
- The implementation allows the error variance to be dependent on the observation and then \$\\sigma^{2}\$ is replaced by \$\\Sigma\$, which is an (N x N) diagonal matrix with \$\\sigma_{i}^{2}\$ in the diagonal elements.
"""
function sampling_β(y::Vector, 
                    x::Array, 
                    _β_prior::Union{Number, Vector}=0, 
                    _V_prior::Union{Number, AbstractArray}=1, 
                    _σ²::Union{Number, AbstractArray}=0.01;
                    last_truncated::Bool=false, 
                    stationarity_check::Bool=false, 
                    constant_included::Bool=false, 
                    max_iterations::Int=10000,
                    β_prior::Union{Number, Vector}=_β_prior, V_prior::Union{Number, AbstractArray}=_V_prior, 
                    σ²::Union{Number, AbstractArray}=_σ²
                    )::Vector
    
    k = size(x, 2)
    i0 = constant_included ? 2 : 1
    # Transform prior inputs to the right format:
    β_prior = isa(β_prior, Number) ? ones(k) * β_prior : β_prior  # Number to vector
    V_prior = isa(V_prior, Number) ? I(k) * V_prior : V_prior  # Number to diag array
    V_prior = isa(V_prior, Vector) ? Diagonal(V_prior) : V_prior  # Vector to diag array
    σ² = isa(σ², Vector) ? Diagonal(σ²) : σ²  # Vector to array

    # Calculate posteriror parameters:
    V_posterior = inv(inv(V_prior) .+ (x' * inv(σ²) * x))
    V_posterior = Matrix(Hermitian(V_posterior))
    β_posterior = V_posterior * (inv(V_prior) * β_prior .+ x' * inv(σ²) * y)

    # Sampling:
    sampled_β = zeros(k)
    i = 1
    while true && i <= max_iterations
        if ~last_truncated
        # Sampling from multivariate normal:
            sampled_β = rand(MultivariateNormal(β_posterior, V_posterior))
        else
        # sampling from multivariate normal with the last component truncated at 0:
            sampled_β = [rand(MultivariateNormal(β_posterior[1:k-1], V_posterior[1:k-1, 1:k-1]));
                         rand(truncated(Normal(β_posterior[k], V_posterior[k, k]), 0, Inf))]
        end
        # check the stationarity of the estimated AR model if required:
        if (stationarity_check && all(abs.(roots(Polynomial([1; -sampled_β[i0:k]]))) .> 1)) || ~stationarity_check
            break
        end
            i += 1
        if i > max_iterations
            error("The sampling procedure has reached the maximum number of iterations (no stationary solution sampled)")
        end
    end
    return sampled_β
end
