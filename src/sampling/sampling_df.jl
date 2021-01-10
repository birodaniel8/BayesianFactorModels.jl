"""
    sampling_df(λ, ν_previous, ν_prior = 30, hm_variance = 0.25)

Sampling the degree of freedom parameter of the normal linear model with independent normal-gamma priors and known
    heteroscedasticity (ie t-errors).

It iterates trough the columns of the matrix `λ` and takes a sample from the distribution given as:

\$p(v|...) \\propto \\Big(\\frac{v}{2}\\Big)^{0.5Tv}\\Gamma\\Big(\\frac{v}{2}\\Big)^{-T}e^{-\\eta v}\$
where \$\\eta = \\frac{1}{v_{prior}} + 0.5\\sum_{i=1}^T[ln(\\lambda^{-1}) + \\lambda]\$ via random walk Metropolis-Hastings algorithm.


## Arguments
- `λ::Array`: (T x m) mixture scale parameters (\$\\lambda\$)
- `ν_previous::Union{Number, Vector}`: (m) vector of the previous degree of freedom parameters
- `ν_prior::Union{Number, Vector}`: (m) vector of prior degree of freedom parameters (\$v_{prior}\$)
- `hm_variance::Union{Number, Vector}`: Random walk Metropolis-Hastings algorithm variance parameters

## Returns
- `ν_sampled::Vector`: (m) sampled degree of freedoms

## Note
- If `ν_previous` is given as a number, an (m) length vector is created and filled by its value
- If `ν_prior` is given as a number, an (m) length vector is created and filled by its value
- If `hm_variance` is given as a number, an (m) length vector is created and filled by its value
"""
function sampling_df(λ::Array, 
                     _ν_previous::Union{Number, Vector}=30, 
                     _ν_prior::Union{Number, Vector}=30, 
                     _hm_variance::Union{Number, Vector}=0.25;
                     ν_previous::Union{Number, Vector}=_ν_previous, ν_prior::Union{Number, Vector}=_ν_prior, 
                     hm_variance::Union{Number, Vector}=_hm_variance
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
