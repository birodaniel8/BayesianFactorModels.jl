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


function mcmc_sampling(model::LinearModelT, y::Vector, x::Array;
                       ndraw::Int=1500, 
                       burnin::Int=500, 
                       init_vals::Dict=Dict(), 
                       hm_variance::Number=0.25,
                       display::Bool=true,
                       display_step::Int=250
                       )
                        
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
    sampled_ν = zeros(1, ndraw - burnin)

    # Initial values
    # Initial values:
    σ² = haskey(init_vals, "σ²") ? [init_vals["σ²"]] : [(model.γ_prior - 1) / model.δ_prior]
    λ = haskey(init_vals, "λ") ? [init_vals["λ"]] : ones(N)
    λ = isa(λ, Number) ? ones(m) * λ : λ
    ν = haskey(init_vals, "ν") ? [init_vals["ν"]] : model.ν_prior

    # Sampling:
    display ? println("Estimate linear model with Student's t errors (via Gibbs sampling)") : -1
    for i = 1:ndraw
        (mod(i, display_step) == 0 && display) ? println(i) : -1

        # Sampling:
        y_star = Diagonal(sqrt.(λ[:,1])) * y
        x_star = Diagonal(sqrt.(λ[:,1])) * x
        β = sampling_β(y_star, x_star, model.β_prior, model.V_prior, σ²)  # sampling beta coefficients
        σ² = sampling_σ²(y_star - x_star * β[:,:], model.γ_prior, model.δ_prior)  # sampling error variance
        ν = sampling_df(λ, ν, model.ν_prior, hm_variance)  # sampling degree of freedom
        λ = sampling_mixture_scale(y - x * β[:,:], σ², ν)  # sampling λ

        # Save samples:
        if i > burnin
            sampled_β[:, i-burnin] = β
            sampled_σ²[:, i-burnin] = σ²
            sampled_ν[:, i-burnin] = ν
        end
    end
    display ? println("Done") : -1
    return [sampled_β, sampled_σ², sampled_ν]
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
    σ² = haskey(init_vals, "σ²") ? init_vals["σ²"] : ones(m) * (model.γ_prior - 1) / model.δ_prior
    σ² = isa(σ², Number) ? ones(m) * σ² : σ²
    factor = haskey(init_vals, "factor") ? init_vals["factor"] : factor_initialize(y, model.k)
    factor = isa(factor, Number) ? ones(T, model.k) * factor : factor
    
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


function mcmc_sampling(model::DynamicLinearFactorModel, y::Array;
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
    sampled_θ = zeros(model.k, ndraw - burnin)
    sampled_factor = zeros(T, model.k, ndraw - burnin)

    # Initial values:
    σ² = haskey(init_vals, "σ²") ? init_vals["σ²"] : ones(m) * (model.γ_prior - 1) / model.δ_prior
    σ² = isa(σ², Number) ? ones(m) * σ² : σ²
    factor = haskey(init_vals, "factor") ? init_vals["factor"] : factor_initialize(y, model.k)
    factor = isa(factor, Number) ? ones(T, model.k) * factor : factor
    θ = zeros(model.k)
    
    # Sampling:
    display ? println("Estimate dynamic linear factor model (via Gibbs sampling)") : -1
    for i = 1:ndraw
        (mod(i, display_step) == 0 && display) ? println(i) : -1

        # Sampling:
        β = sampling_factor_loading(y, factor, model.β_prior, model.V_prior, σ²)  # sampling factor loadings
        σ² = sampling_σ²(y - factor * β', model.γ_prior, model.δ_prior)  # sampling error variances
        for j = 1:model.k
            θ[j] = sampling_β(factor[2:T, j],factor[1:T-1, j], model.θ_prior, model.θ_var_prior, 1, stationarity_check=true)[1]  # sampling factor AR(1) coefficients
        end
        factor = sampling_factor_dynamic(y, β, θ, σ²)  # sampling factors

        # Save samples:
        if i > burnin
        sampled_β[:, :, i-burnin] = β
        sampled_σ²[:, i-burnin] = σ²
        sampled_θ[:, i-burnin] = θ
        sampled_factor[:, :, i-burnin] = factor
        end
    end
    display ? println("Done") : -1
    return [sampled_β, sampled_σ², sampled_θ, sampled_factor]
end


function mcmc_sampling(model::StochasticVolatilityModel, y::Array;
                       ndraw::Int=1500, 
                       burnin::Int=500, 
                       init_vals::Dict=Dict(), 
                       display::Bool=true,
                       display_step::Int=250
                       )
    
    T = size(y, 1)

    # Create containers:
    sampled_ρ = zeros(2, ndraw - burnin)
    sampled_τ² = zeros(ndraw - burnin)
    sampled_h = zeros(T, ndraw - burnin)

    # Initial values:
    τ² = haskey(init_vals, "τ²") ? init_vals["τ²"] : (model.τ_γ_prior - 1) / model.τ_δ_prior
    h = haskey(init_vals, "h") ? init_vals["h"] : log.(y.^2)
    h = isa(h, Number) ? ones(T) * h : h
    h0 = haskey(init_vals, "h0") ? init_vals["h0"] : 0
    P0 = haskey(init_vals, "P0") ? init_vals["P0"] : 10
    
    # Sampling:
    display ? println("Estimate 1st order stochastic volatility model (via Gibbs sampling)") : -1
    for i = 1:ndraw
        (mod(i, display_step) == 0 && display) ? println(i) : -1
        
        # Sampling:
        ρ = sampling_β(h[2:T], [ones(T-1) h[1:T-1]], model.ρ_prior, model.ρ_var_prior, τ², 
                       stationarity_check=true, constant_included=true)  # sampling the volatility process parameters
        τ² = sampling_σ²(h[2:T] .- ρ[1] - ρ[2] .* h[1:T-1], model.τ_γ_prior, model.τ_δ_prior)  # sampling volatility of volatility
        h = sampling_stochastic_volatility(y, h, ρ, τ²[1], h0, P0)  # sampling stochastic volatilty

        # Save samples:
        if i > burnin
            sampled_ρ[:, i-burnin] = ρ
            sampled_τ²[i-burnin] = τ²[1]
            sampled_h[:, i-burnin] = h
        end
    end
    display ? println("Done") : -1
    return [sampled_ρ, sampled_τ², sampled_h]
end


function mcmc_sampling(model::DynamicLinearFactorSVModel, y::Array;
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
    sampled_ρ = zeros(2, m, ndraw - burnin)
    sampled_τ² = zeros(m, ndraw - burnin)
    sampled_h = zeros(T, m, ndraw - burnin)
    sampled_θ = zeros(model.k, ndraw - burnin)
    sampled_factor = zeros(T, model.k, ndraw - burnin)
    sampled_ρ_factor = zeros(2, model.k, ndraw - burnin)
    sampled_τ²_factor = zeros(model.k, ndraw - burnin)
    sampled_g = zeros(T, model.k, ndraw - burnin)

    # Initial values:
    τ² = haskey(init_vals, "τ²") ? init_vals["τ²"] : ones(m) * (model.τ_γ_prior - 1) / model.τ_δ_prior
    τ² = isa(τ², Number) ? ones(m) * τ² : τ²
    τ²_factor = haskey(init_vals, "τ²_factor") ? init_vals["τ²_factor"] : ones(model.k) * (model.τ_factor_γ_prior - 1) / model.τ_factor_δ_prior
    τ²_factor = isa(τ²_factor, Number) ? ones(model.k) * τ²_factor : τ²_factor
    factor = haskey(init_vals, "factor") ? init_vals["factor"] : factor_initialize(y, model.k)
    factor = isa(factor, Number) ? ones(T, model.k) * factor : factor
    h = haskey(init_vals, "h") ? init_vals["h"] : zeros(T, m)
    h = isa(h, Number) ? ones(T, m) * h : h
    g = haskey(init_vals, "g") ? init_vals["g"] : zeros(T, model.k)
    g = isa(g, Number) ? ones(T, model.k) * g : g

    θ = zeros(model.k)
    ρ = zeros(2, m)
    ρ_factor = zeros(2, model.k)
    error_variance = zeros(m, m, T)
    error_variance[repeat(Matrix(I(m))[:], T)] = exp.(h)'[:]
    factor_error_variance = zeros(model.k, model.k, T)
    factor_error_variance[repeat(Matrix(I(model.k))[:], T)] = exp.(g)'[:]
    
    # Sampling:
    display ? println("Estimate dynamic linear factor stochastic volatility model (via Gibbs sampling)") : -1
    for i = 1:ndraw
        (mod(i, display_step) == 0 && display) ? println(i) : -1

        # Sampling factor loadings:
        β = sampling_factor_loading(y, factor, model.β_prior, model.V_prior, error_variance)

        # Sampling stochastic volatility components:
        ϵ = y - factor * β'
        for i = 1:m
            ρ[:, i] = sampling_β(h[2:T, i], [ones(T-1) h[1:T-1, i]], model.ρ_prior, model.ρ_var_prior, τ²[i],
                                 stationarity_check=true, constant_included=true)
            τ²[i] = sampling_σ²(h[2:T, i] .- ρ[1, i] - ρ[2, i] .* h[1:T-1, i], model.τ_γ_prior, model.τ_δ_prior)[1]
            h[:, i] = sampling_stochastic_volatility(ϵ[:, i], h[:, i], ρ[:, i], τ²[i])
        end

        # Sampling factor AR(1) coefficients:
        for j = 1:model.k
            factor_star = 1 ./ (exp.(g[2:T, j]/2)) .* factor[2:T, j]
            factor_lag_star = 1 ./(exp.(g[2:T, j]/2)) .* factor[1:T-1, j]
            θ[j] = sampling_β(factor_star, factor_lag_star, model.θ_prior, model.θ_var_prior, 1, 
                              stationarity_check=true)[1]   # sigma2 = 1?
        end 

        # Sampling factors:
        error_variance[repeat(Matrix(I(m))[:], T)] = exp.(h)'[:]
        factor_error_variance[repeat(Matrix(I(model.k))[:], T)] = exp.(g)'[:]
        factor = sampling_factor_dynamic(y, β, θ, error_variance, factor_error_variance)

        # Sampling factor stochastic volatility components:
        ϵ_factor = factor[2:T, :] - factor[1:T-1, :]
        for j = 1:model.k
            ρ_factor[:, j] = sampling_β(g[3:T, j], [ones(T-2) g[2:T-1, j]], model.ρ_factor_prior, 
                                        model.ρ_factor_var_prior, τ²_factor[j], 
                                        stationarity_check=true, constant_included=true)
            τ²_factor[j] = sampling_σ²(g[3:T, j] .- ρ_factor[1, j] - ρ_factor[2, j] .* g[2:T-1, j],
                                       model.τ_factor_γ_prior, model.τ_factor_δ_prior)[1]
            g[2:T, j] = sampling_stochastic_volatility(ϵ_factor[:, j], g[2:T, j], ρ_factor[:, j], τ²_factor[j])
            g[1, j] = g[2, j]
            g = g.-mean(g, dims=1)  # demeaning
        end

        # Save samples:
        if i > burnin
            sampled_β[:, :, i-burnin] = β
            sampled_ρ[:, :, i - burnin] = ρ
            sampled_τ²[:, i - burnin] = τ²
            sampled_h[:, :, i - burnin] = h
            sampled_θ[:, i-burnin] = θ
            sampled_factor[:, :, i-burnin] = factor
            sampled_ρ_factor[:, :, i - burnin] = ρ_factor
            sampled_τ²_factor[:, i - burnin] = τ²_factor
            sampled_g[:, :, i - burnin] = g
        end
    end
    display ? println("Done") : -1
    return [sampled_β, sampled_ρ, sampled_τ², sampled_h, 
            sampled_θ, sampled_factor, sampled_ρ_factor, sampled_τ²_factor, sampled_g]
end