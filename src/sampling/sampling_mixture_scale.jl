function sampling_mixtrue_scale(ϵ::Array, 
                                _σ²::Vector, 
                                _ν::Vector;
                                σ²::Vector=_σ², ν::Vector=_ν)
    N = size(ϵ, 1)
    m = size(ϵ, 2)

    # Sampling:
    sampled_λ = zeros(N, m)
    for i = 1:m
        for j = 1:N
            α_λ = (ν[i] + 1) / 2
            β_λ = 2 / ((ϵ[j, i] ^ 2) / σ²[i] + ν[i])
            sampled_λ[j, i] = rand(Gamma(α_λ, β_λ))
        end
    end
    return sampled_λ
end
