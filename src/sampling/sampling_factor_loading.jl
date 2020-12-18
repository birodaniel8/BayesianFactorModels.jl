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
        if i <= k
            sampled_β[i, 1:i] = sampling_β(y[:, i], f[:, 1:i], β_prior[i, 1:i], V_prior[i, 1:i], Σ[i, i], 
                                           last_truncated=true)
        else
            sampled_β[i, :] = sampling_β(y[:, i], f, β_prior[i, :], V_prior[i, :], Σ[i, i])
        end
    end
    return sampled_β
end
