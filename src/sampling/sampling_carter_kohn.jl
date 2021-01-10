function sampling_carter_kohn(x::Array,
                              P::AbstractArray,
                              G::Union{Number, AbstractArray},
                              Q::Union{Number, AbstractArray},
                              _μ::Union{Number, Vector}=0,
                              _j::Union{Int, DataType}=Nothing;
                              μ::Union{Number, Vector}=_μ, j::Union{Int, DataType}=_j
                              )::Array

    j = j == Nothing ? size(Q, 1) : j

    if size(G, 3) > 1 || size(Q, 3) > 1
        sampled_x = _sampling_carter_kohn_tvp(x, P, G, Q, μ, j)
    else
        T = size(x, 1)
        k = size(x, 2)
        x = x'

        # Convert everything to the right format
        G = isa(G, Number) ? I(k) * G : G
        Q = isa(Q, Number) ? I(k) * Q : Q
        G_star = G[1:j, :]
        Q_star = Q[1:j, 1:j]

        # Sampling:
        sampled_x = zeros(k, T)
        for s = T:-1:1
            if s == T
                sampled_x[:, s] = rand(MultivariateNormal(x[:, s], Matrix(Hermitian(P[:, :, s]))))
            else
                x_star = x[:, s] + P[:, :, s] * G_star' * inv(G_star * P[:, :, s] * G_star' + Q_star) * (sampled_x[1:j, s+1] .- μ - G_star * x[:, s])
                P_star = P[:, :, s] - P[:, :, s] * G_star' * inv(G_star * P[:, :, s] * G_star' + Q_star) * G_star * P[:, :, s]
                sampled_x[:, s] = rand(MultivariateNormal(x_star, Matrix(Hermitian(P_star))))
            end
        end
        sampled_x = Array(sampled_x')
    end
    return sampled_x
end


function _sampling_carter_kohn_tvp(x::Array,
                                   P::AbstractArray,
                                   G::Union{Number, AbstractArray},
                                   Q::Union{Number, AbstractArray},
                                   _μ::Union{Number, Vector}=0,
                                   _j::Union{Int, DataType}=Nothing;
                                   μ::Union{Number, Vector}=_μ, j::Union{Int, DataType}=_j
                                   )::Array
                                  
    j = j == Nothing ? size(Q, 1) : j
    T = size(x,1)
    k = size(x,2)
    x = x'

    # Convert everything to 3d arrays:
    G = isa(G, Number) ? I(k) * G : G
    Q = isa(Q, Number) ? I(k) * Q : Q
    G = size(G, 3) == 1 ? G = repeat(G, 1, 1, T) : G
    Q = size(Q, 3) == 1 ? Q = repeat(Q, 1, 1, T) : Q
    G_star = G[1:j, :, :]
    Q_star = Q[1:j, 1:j, :]

    # Sampling:
    sampled_x = zeros(k, T)
    for s = T:-1:1
        if s == T
            sampled_x[:, s] = rand(MultivariateNormal(x[:, s], Matrix(Hermitian(P[:, :, s]))))
        else
            x_star = x[:, s] + P[:, :, s] * G_star[:, :, s]' * inv(G_star[:, :, s] * P[:, :, s] * G_star[:, :, s]' + Q_star[:, :, s]) * (sampled_x[1:j, s+1] .- μ - G_star[:, :, s] * x[:, s])
            P_star = P[:, :, s] - P[:, :, s] * G_star[:, :, s]' * inv(G_star[:, :, s] * P[:, :, s] * G_star[:, :, s]' + Q_star[:, :, s]) * G_star[:, :, s] * P[:, :, s]
            sampled_x[:, s] = rand(MultivariateNormal(x_star, Matrix(Hermitian(P_star))))
        end
    end
    sampled_x = Array(sampled_x')
    return sampled_x
end
