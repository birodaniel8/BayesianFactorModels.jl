struct LinearModel
    β_prior::Union{Number, Vector}
    V_prior::Union{Number, Array}
    γ_prior::Number
    δ_prior::Number
    add_constant::Bool

    function LinearModel(_β_prior::Union{Number, Vector}=0, 
                         _V_prior::Union{Number, Array}=1, 
                         _γ_prior::Number=1.5, 
                         _δ_prior::Number=0.5,
                         _add_constant::Bool=false;
                         β_prior::Union{Number, Vector}=_β_prior, V_prior::Union{Number, Array}=_V_prior, 
                         γ_prior::Number=_γ_prior, δ_prior::Number=_δ_prior, add_constant::Bool=_add_constant)
        new(β_prior, V_prior, γ_prior, δ_prior, add_constant)
    end
end
