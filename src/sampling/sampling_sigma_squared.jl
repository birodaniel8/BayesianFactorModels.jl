function sampling_σ²(ϵ::Array, 
                     _γ_prior::Number=1.5, 
                     _δ_prior::Number=0.5;
                     γ_prior=_γ_prior, δ_prior=_δ_prior
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
