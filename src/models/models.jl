struct LinearModel
    β_prior::Union{Number, Vector}
    V_prior::Union{Number, AbstractArray}
    γ_prior::Number
    δ_prior::Number
    add_constant::Bool

    function LinearModel(_β_prior::Union{Number, Vector}=0, 
                         _V_prior::Union{Number, AbstractArray}=1, 
                         _γ_prior::Number=1.5, 
                         _δ_prior::Number=0.5,
                         _add_constant::Bool=false;
                         β_prior::Union{Number, Vector}=_β_prior, V_prior::Union{Number, AbstractArray}=_V_prior, 
                         γ_prior::Number=_γ_prior, δ_prior::Number=_δ_prior, add_constant::Bool=_add_constant)
        new(β_prior, V_prior, γ_prior, δ_prior, add_constant)
    end
end


struct LinearModelT
    β_prior::Union{Number, Vector}
    V_prior::Union{Number, AbstractArray}
    γ_prior::Number
    δ_prior::Number
    ν_prior::Number
    add_constant::Bool

    function LinearModelT(_β_prior::Union{Number, Vector}=0, 
                          _V_prior::Union{Number, AbstractArray}=1, 
                          _γ_prior::Number=1.5, 
                          _δ_prior::Number=0.5,
                          _ν_prior::Number=30,
                          _add_constant::Bool=false;
                          β_prior::Union{Number, Vector}=_β_prior, V_prior::Union{Number, AbstractArray}=_V_prior, 
                          γ_prior::Number=_γ_prior, δ_prior::Number=_δ_prior, ν_prior::Number=_ν_prior,
                          add_constant::Bool=_add_constant)
        new(β_prior, V_prior, γ_prior, δ_prior, ν_prior, add_constant)
    end
end


struct LinearFactorModel
    k::Int
    β_prior::Union{Number, Array}
    V_prior::Union{Number, AbstractArray}
    γ_prior::Number
    δ_prior::Number

    function LinearFactorModel(_k::Int=1,
                               _β_prior::Union{Number, Array}=0, 
                               _V_prior::Union{Number, AbstractArray}=1, 
                               _γ_prior::Number=1.5, 
                               _δ_prior::Number=0.5;
                               k=_k::Int, β_prior::Union{Number, Array}=_β_prior, 
                               V_prior::Union{Number, AbstractArray}=_V_prior, γ_prior::Number=_γ_prior, 
                               δ_prior::Number=_δ_prior)
        new(k, β_prior, V_prior, γ_prior, δ_prior)
    end
end


struct DynamicLinearFactorModel
    k::Int
    β_prior::Union{Number, Array}
    V_prior::Union{Number, AbstractArray}
    γ_prior::Number
    δ_prior::Number
    θ_prior::Number
    θ_var_prior::Number

    function DynamicLinearFactorModel(_k::Int=1,
                                      _β_prior::Union{Number, Array}=0, 
                                      _V_prior::Union{Number, AbstractArray}=1, 
                                      _γ_prior::Number=1.5, 
                                      _δ_prior::Number=0.5,
                                      _θ_prior::Number=0,
                                      _θ_var_prior::Number=1;
                                      k=_k::Int, β_prior::Union{Number, Array}=_β_prior, 
                                      V_prior::Union{Number, AbstractArray}=_V_prior, γ_prior::Number=_γ_prior, 
                                      δ_prior::Number=_δ_prior, θ_prior::Number=_θ_prior, 
                                      θ_var_prior::Number=_θ_var_prior)
        new(k, β_prior, V_prior, γ_prior, δ_prior, θ_prior, θ_var_prior)
    end
end


struct StochasticVolatilityModel
    ρ_prior::Union{Number, Vector}
    ρ_var_prior::Union{Number, AbstractArray}
    τ_γ_prior::Number
    τ_δ_prior::Number

    function StochasticVolatilityModel(_ρ_prior::Union{Number, Vector}=0,
                                       _ρ_var_prior::Union{Number, AbstractArray}=3,
                                       _τ_γ_prior::Number=1.5,
                                       _τ_δ_prior::Number=0.05;
                                       ρ_prior::Union{Number, Vector}=_ρ_prior, 
                                       ρ_var_prior::Union{Number, AbstractArray}=_ρ_var_prior, 
                                       τ_γ_prior::Number=_τ_γ_prior, τ_δ_prior::Number=_τ_δ_prior)
        new(ρ_prior, ρ_var_prior, τ_γ_prior, τ_δ_prior)
    end
end
