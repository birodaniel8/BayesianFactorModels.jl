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
                         _add_constant=false;
                         β_prior=_β_prior, V_prior=_V_prior, 
                         γ_prior=_γ_prior, δ_prior=_δ_prior,
                         add_constant=_add_constant)
        new(β_prior, V_prior, γ_prior, δ_prior, add_constant)
    end
end