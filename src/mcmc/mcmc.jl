function mcmc_sampling(model::LinearModel, y::Vector, x::Array;
                       ndraw::Int=1500, 
                       burnin::Int=500, 
                       init_vals::Dict=Dict(), 
                       display::Bool=true,
                       display_step::Int=250
                       )::Array{Array{Float64,2},1}
    
    N = size(x, 1)
    k = size(x, 2)

    # Add constant:
    if model.add_constant
        x = [ones(N) x]
        k = k + 1
    end

    # Create containers:
    sampled_β = zeros(k, ndraw - burnin)
    sampled_σ² = zeros(1, ndraw - burnin)

    # Initial values:
    σ² = haskey(init_vals, "σ²") ? [init_vals["σ²"]] : [(model.γ_prior - 1) / model.δ_prior]
    
    # Sampling:
    display ? println("Estimate normal linear model (via Gibbs sampling)") : -1
    for i = 1:ndraw
        (mod(i, display_step) == 0 && display) ? println(i) : -1

        # Sampling:
        β = sampling_β(y, x, model.β_prior, model.V_prior, σ²)  # sampling β coefficients
        σ² = sampling_σ²(y - x * β[:,:], model.γ_prior, model.δ_prior)  # sampling error variance

        # Save samples:
        if i > burnin
        sampled_β[:, i-burnin] = β
        sampled_σ²[:, i-burnin] = σ²
        end
    end
    display ? println("Done") : -1
    return [sampled_β, sampled_σ²]
end


function mcmc_sampling(model::LinearFactorModel, y::Array;
                       ndraw::Int=1500, 
                       burnin::Int=500, 
                       init_vals::Dict=Dict(), 
                       display::Bool=true,
                       display_step::Int=250
                       )
    
    T = size(y, 1)
    m = size(y, 2)

    # Create containers:
    sampled_β = zeros(m, model.k, ndraw - burnin)
    sampled_σ² = zeros(m, ndraw - burnin)
    sampled_factor = zeros(T, model.k, ndraw - burnin)

    # Initial values:
    σ² = haskey(init_vals, "σ²") ? [init_vals["σ²"]] : ones(m) * (model.γ_prior - 1) / model.δ_prior
    σ² = isa(σ², Number) ? ones(m) * σ² : σ²
    factor = haskey(init_vals, "factor") ? [init_vals["factor"]] : factor_initialize(y, model.k)
    
    # Sampling:
    display ? println("Estimate normal linear factor model (via Gibbs sampling)") : -1
    for i = 1:ndraw
        (mod(i, display_step) == 0 && display) ? println(i) : -1

        # Sampling:
        β = sampling_factor_loading(y, factor, model.β_prior, model.V_prior, σ²)  # sampling factor loadings
        σ² = sampling_σ²(y - factor * β', model.γ_prior, model.δ_prior)  # sampling error variance
        factor = sampling_factor(y, β, σ²)  # sampling factors from normal distribution

        # Save samples:
        if i > burnin
        sampled_β[:, :, i-burnin] = β
        sampled_σ²[:, i-burnin] = σ²
        sampled_factor[:, :, i-burnin] = factor
        end
    end
    display ? println("Done") : -1
    return [sampled_β, sampled_σ², sampled_factor]
end
