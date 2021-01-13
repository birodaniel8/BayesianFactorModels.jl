"""
sampling_stochastic_volatility(ϵ, h_previous, ρ, τ², h0=0, P0=1)

Sampling the stochastic volatility component of the 1st order autoregressive stochastic volatility model.

It utilizes the procedure of Kim et al (1998), which describes the stochastic volatility by a state space model written on the log squared errors.
However the error term in the observation equation is \$log(\\chi_1)\$ distributed, which is approximated by a normal mixturemodel. 
First we take the normal scale mixture parameter sample from a given discrete distribution.
This distribution is described in Kim et al (1998) and the parameters (\$q_{\\omega}^*, m_{\\omega}^*, {v_{\\omega}^*}^2)\$
are contained in Table 4 of the original paper.
Then we take a sample of the stochastic volatility via the Carter & Kohn (1994) algorithm from the following model:

\$e^*_t = h_t + u_t \\quad \\quad u_t \\sim N(0,{v_t^*}^2)\$

\$h_t = \\rho_0 + \\rho_1 h_{t-1} + \\eta_t \\quad \\quad \\eta_t \\sim N(0,\\tau^2)\$

where \$e_t^* = log(e_t^2) - m_{\\omega}^* + 1.2704\$.

## Arguments
- `ϵ::Vector`: (T) error error vector
- `h_previous::Vector`: (T) stochastic variance component from the previous MCMC step
- `ρ::Vector`: (2) vector of the constant and autoregressive coefficient of the stochastic volatility (\$\\rho_0\$, \$\\rho_1\$)
- `τ²::Union{Number, Vector}`: Variance of the stochastic volatility (\$\\tau^2\$)
- `h0::Number`: initial state value of the stochastic volatility
- `P0::Number`: initial state variance of the stochastic volatility

## Returns
- `sampled_h::Vector`: (T) sampled stochastic variance
"""
function sampling_stochastic_volatility(ϵ::Vector, 
                                        h_previous::Vector, 
                                        ρ::Vector, 
                                        τ²::Union{Number, Vector}, 
                                        h0::Number=0, 
                                        P0::Number=1)
    
    # Input parameters:
    table4 = [[1 0.00730 -10.12999 5.79596];
              [2 0.10556 -3.97281 2.61369];
              [3 0.00002 -8.56686 5.17950];
              [4 0.04395 2.77786 0.16735];
              [5 0.34001 0.61942 0.64009];
              [6 0.24566 1.79518 0.34023];
              [7 0.25750 -1.08819 1.26261]]
    c = 1.2704
    T = size(ϵ, 1)
    τ² = isa(τ², Vector) ? τ²[1] : τ²  # Vector to number
    any(ϵ .== 0) ? println("The error vector contains a 0 value. Apply an offset to aviod it!") : -1

    # Sampling the mixture normal parameter:
    u_star = log.(ϵ.^2) - h_previous
    ω = zeros(T, 1)
    for t = 1:T
        q = map(x -> table4[x, 2] * pdf(Normal(table4[x, 3]-c, sqrt(table4[x, 4])), u_star[t]), 1:7)
        q = q ./ sum(q)
        q_prob = cumsum(q)
        ω[t] = sum(q_prob .< rand(1)) + 1
    end
    ω = convert.(Int, ω)

    # Sampling stochastic volatility:
    ϵ_star = log.(ϵ.^2) - (table4[ω, 3] .- c)
    H = 1
    R = reshape(table4[ω, 4], (1, 1, T))
    h_hat, P_hat = kalman_filter(ϵ_star, H, R, ρ[2], τ², ρ[1], h0, P0)
    sampled_h = sampling_carter_kohn(h_hat, P_hat, ρ[2], τ², ρ[1])
    return sampled_h[:, 1]
end
