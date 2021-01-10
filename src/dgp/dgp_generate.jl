function dgp_generate(dgp::NormalFactorModelDGP,
                      _T::Int=100;
                      T::Int=_T
                      )::Tuple{Array, Array}
    # Sampling observations:
    f = randn(T, dgp.k)
    y = f * dgp.β' + randn(T, dgp.m) * Matrix(cholesky(dgp.σ²).U)
    return y, f
end


function dgp_generate(dgp::DynamicFactorModelDGP,
                      _T::Int=100;
                      T::Int=_T
                      )::Tuple{Array, Array}
    # Sampling observations:
    f = zeros(T, dgp.k)
    for t = 2:T
        f[t, :] = dgp.θ .* f[t-1, :] + randn(dgp.k)
    end
    y = f * dgp.β' + randn(T, dgp.m) * Matrix(cholesky(dgp.σ²).U)
    return y, f
end