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
    β_prior::Union{Number, Vector}
    V_prior::Union{Number, AbstractArray}
    γ_prior::Number
    δ_prior::Number

    function LinearFactorModel(_k::Int,
                               _β_prior::Union{Number, Vector}=0, 
                               _V_prior::Union{Number, AbstractArray}=1, 
                               _γ_prior::Number=1.5, 
                               _δ_prior::Number=0.5;
                               k=_k::Int, β_prior::Union{Number, Vector}=_β_prior, 
                               V_prior::Union{Number, AbstractArray}=_V_prior, γ_prior::Number=_γ_prior, 
                               δ_prior::Number=_δ_prior)
        new(k, β_prior, V_prior, γ_prior, δ_prior)
    end
end
