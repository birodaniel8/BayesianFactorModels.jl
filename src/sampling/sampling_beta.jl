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
