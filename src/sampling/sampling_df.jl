function sampling_df(λ::Array, 
                     _ν_previous::Union{Number, Vector}=30, 
                     _ν_prior::Union{Number, Vector}=30, 
                     _hm_variance::Number=0.25;
                     ν_previous::Union{Number, Vector}=_ν_previous, ν_prior::Union{Number, Vector}=_ν_prior, 
                     hm_variance::Number=_hm_variance
                     )::Vector
                     
    N = size(λ, 1)
    m = size(λ, 2)
    # Transform inputs to the right format:
    ν_prior = isa(ν_prior, Number) ? ones(m) * ν_prior : ν_prior  # Number to vector
    ν_previous = isa(ν_previous, Number) ? ones(m) * ν_previous : ν_previous  # Number to vector
    hm_variance = isa(hm_variance, Number) ? ones(m) * hm_variance : hm_variance  # Number to vector

    # Sampling:
    ν_sampled = zeros(m)
    for i = 1:m
        # Metropolis-Hastings sampling:
        ν_proposed = ν_previous[i] + sqrt(hm_variance[i]) * randn()
        η = 1 / ν_prior[i] + 0.5 * sum(-log.(λ[:, i]) + λ[:, i])
        # Calculate acceptance probability:
        if ν_proposed > 0
            l_post_proposed = 0.5 * N * ν_proposed * log(0.5 * ν_proposed) - N * loggamma( 0.5 * ν_proposed) - η * ν_proposed
            l_post_sampled = 0.5 * N * ν_previous[i] * log.(0.5 * ν_previous[i]) - N * loggamma.(0.5 * ν_previous[i]) - η * ν_previous[i]
            α = exp.(l_post_proposed - l_post_sampled)
        else
            α = 0
        end

        if rand() < α
            ν_sampled[i] = ν_proposed
        else
            ν_sampled[i] = ν_previous[i]
        end
    end
    return ν_sampled
end
