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
