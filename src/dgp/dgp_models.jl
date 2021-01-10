struct NormalFactorModelDGP
    m::Int
    k::Int
    β::Union{Number, AbstractArray, DataType}
    σ²::Union{Number, AbstractArray, DataType}
    default_params::Bool

    function NormalFactorModelDGP(_m::Int=9,
                                  _k::Int=3,
                                  _β::Union{Number, AbstractArray, DataType}=Nothing,
                                  _σ²::Union{Number, AbstractArray, DataType}=Nothing;
                                  default_params::Bool=false,
                                  m::Int=_m, k::Int=_k,
                                  β::Union{Number, AbstractArray, DataType}=_β, 
                                  σ²::Union{Number, AbstractArray, DataType}=_σ²)
        if default_params
            β = [1 0 0;
                 0.45 1 0;
                 0 0.34 1;
                 0.99 0 0;
                 0.99 0 0;
                 0 0.95 0;
                 0 0.95 0;
                 0.56 0 0.90;
                 0 0 0.90]
            σ² = Diagonal([0.02; 0.19; 0.36; 0.02; 0.02; 0.19; 0.19; 0.36; 0.36])
        end
        β = isa(β, Number) ? ones(m, k) * β : β  # fill an m x k matrix with β
        β = β == Nothing ? rand(m, k) * 1.5 .- 0.5 : β  # generate β coeffs in (-0.5, 1)
        β = β[1, 1] < 0 ? β * -1 : β  # if the [1,1] element is negative, flip the signs
        m = size(β, 1)
        k = size(β, 2)
        
        # set the upper triangular part of β to 0
        if k > 1
            for j=2:k, i=1:j-1
                β[i, j] = 0;
            end
        end
        
        σ² = isa(σ², Number) ? Diagonal(ones(m) * σ²) : σ²  # fill a diagonal matrix with σ²
        σ² = isa(σ², Vector) ? Diagonal(σ²) : σ²  # reshape σ² to diagonal matrix
        σ² = σ² == Nothing ? Diagonal(rand(m) * 0.5) : σ²  # generate σ² error variance in (0, 0.5)
        new(m, k, β, σ², default_params)
    end
end


struct DynamicFactorModelDGP
    m::Int
    k::Int
    β::Union{Number, AbstractArray, DataType}
    σ²::Union{Number, AbstractArray, DataType}
    θ::Union{Number, Vector, DataType}
    default_params::Bool

    function DynamicFactorModelDGP(_m::Int=9,
                                   _k::Int=3,
                                   _β::Union{Number, AbstractArray, DataType}=Nothing,
                                   _σ²::Union{Number, AbstractArray, DataType}=Nothing,
                                   _θ::Union{Number, Vector, DataType}=Nothing;
                                   default_params::Bool=false,
                                   m::Int=_m, k::Int=_k,
                                   β::Union{Number, AbstractArray, DataType}=_β, 
                                   σ²::Union{Number, AbstractArray, DataType}=_σ²,
                                   θ::Union{Number, Vector, DataType}=_θ)
        if default_params
            β = [1 0 0;
                 0.45 1 0;
                 0 0.34 1;
                 0.99 0 0;
                 0.99 0 0;
                 0 0.95 0;
                 0 0.95 0;
                 0.56 0 0.90;
                 0 0 0.90]
            σ² = Diagonal([0.02; 0.19; 0.36; 0.02; 0.02; 0.19; 0.19; 0.36; 0.36])
            θ = [0.5; 0; -0.2]
        end
        β = isa(β, Number) ? ones(m, k) * β : β  # fill an m x k matrix with β
        β = β == Nothing ? rand(m, k) * 1.5 .- 0.5 : β  # generate β coeffs in (-0.5, 1)
        β = β[1, 1] < 0 ? β * -1 : β  # if the [1,1] element is negative, flip the signs
        m = size(β, 1)
        k = size(β, 2)
        
        # set the upper triangular part of β to 0
        if k > 1
            for j=2:k, i=1:j-1
                β[i, j] = 0;
            end
        end
        
        σ² = isa(σ², Number) ? Diagonal(ones(m) * σ²) : σ²  # fill a diagonal matrix with σ²
        σ² = isa(σ², Vector) ? Diagonal(σ²) : σ²  # reshape σ² to diagonal matrix
        σ² = σ² == Nothing ? Diagonal(rand(m) * 0.5) : σ²  # generate σ² error variance in (0, 0.5)

        θ = isa(θ, Number) ? ones(k) * θ : θ
        θ = θ == Nothing ? rand(k) * 1.5 .- 0.5 : θ  # generate β coeffs in (-0.5, 1)
        new(m, k, β, σ², θ, default_params)
    end
end