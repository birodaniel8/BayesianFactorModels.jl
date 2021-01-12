"""
    sampling_factor_dynamic(y, β, θ, Σ, Σ_factor=1)

Sampling the factors of the bayesian dynamic normal factor model. It first gets the Kalman Filter state estimates, then a
sample is taken using the Carter & Kohn (1994) sampling algorithm.

\$y_t = f_t \\beta_t' + \\epsilon_t \\quad \\quad \\epsilon_{t,i} \\sim N(0,\\sigma^2_i)\$

\$f_{t,j} = \\theta_j f_{t,j-1} +\\eta_{t,j} \\quad \\quad \\eta_{t,j} \\sim N(0,1)\$

where \$i = 1...m\$, \$j = 1...k\$.

## Arguments
- `y::Array`: (T x m) matrix of observations
- `β::Array`: (m x k x (T)) factor loadings
- `θ::Union{Number, Vector}`: (k) autoregressive coefficient of the factors
- `Σ::Union{Number, AbstractArray}`: (m x m x (T)) covariance matrix of the observation equation
- `Σ_factor::Union{Number, AbstractArray}`: (k x k x (T)) covariance matrix of the factor equation

## Returns
- `sampled_factor::Array`: (T x k) sampled factor values

## Note
- If `θ` is given as a number, a (k) length vector is created and filled by the given value
- If `Σ` is given as a number or a vector, a (m x m) diagonal matrix is created having the input values in the diagonal
- If `Σ_factor` is given as a number or a vector, a (k x k) diagonal matrix is created having the input values in the diagonal
- The function can also sample with time-varying `Σ`, `Σ_factor` parameters
"""
function sampling_factor_dynamic(y::Array, 
                                 β::Array, 
                                 θ::Union{Number, Vector}, 
                                 Σ::Union{Number, AbstractArray}, 
                                 Σ_factor::Union{Number, AbstractArray}=1)::Array
    k = size(β, 2)
    m = size(β, 1)

    # Transform prior inputs to the right format:
    θ = isa(θ, Number) ? ones(k) * θ : θ  # Number to vector
    Σ = isa(Σ, Number) ? I(k) * Σ : Σ  # Number to diag array
    Σ = isa(Σ, Vector) ? Diagonal(Σ) : Σ  # Vector to diag array
    Σ_factor = isa(Σ_factor, Number) ? I(k) * Σ_factor : Σ_factor  # Number to diag array
    Σ_factor = isa(Σ_factor, Vector) ? Diagonal(Σ_factor) : Σ_factor  # Vector to diag array

    H = β
    R = Σ
    G = Diagonal(θ)
    Q = Σ_factor
    x0 = zeros(k)
    P0 = I(k)

    # Sampling factor values:
    F, P = kalman_filter(y, H, R, G, Q, 0, x0, P0)
    sampled_factor = sampling_carter_kohn(F, P, G, Q)
    return sampled_factor
end
