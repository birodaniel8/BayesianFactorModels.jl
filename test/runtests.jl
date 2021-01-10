using BayesianFactorModels
using Test
using Random
using LinearAlgebra

@testset "sampling β test" begin
    # Random sampled input with seed:
    Random.seed!(0)
    n = 10
    y = rand(n)
    x = rand(n)
    x_2 = rand(n, 2)

    # Test cases:
    sampled_β_1 = sampling_β(y, x)
    sampled_β_2 = sampling_β(y, x, 1)
    sampled_β_3 = sampling_β(y, x_2)
    sampled_β_4 = sampling_β(y, x_2, β_prior=[5, 3], σ²=0.05)
    sampled_β_5 = sampling_β(y, x_2, stationarity_check=true)
    sampled_β_6 = sampling_β(y, x_2, V_prior=[[0.5, 0] [0, 0.3]])
    
    # Tests:
    @test sampled_β_1 ≈ [0.61586512]
    @test sampled_β_2 ≈ [0.57539026]
    @test sampled_β_3 ≈ [-0.22455082, 0.77090976]
    @test sampled_β_4 ≈ [0.04423264, 0.52245741]
    @test sampled_β_5 ≈ [-0.20563735, 0.74841896]
    @test sampled_β_6 ≈ [-0.10634513, 0.74995998]
end

@testset "sampling σ² test" begin
    # Random sampled input with seed:
    Random.seed!(0)
    t = 10
    error = rand(t) .- 0.5
    error_2 = rand(t, 3) .- 0.5

    # Test cases:
    sampled_σ²_1 = sampling_σ²(error)
    sampled_σ²_2 = sampling_σ²(error, 5, 0.01)
    sampled_σ²_3 = sampling_σ²(error_2)
    sampled_σ²_4 = sampling_σ²(error_2, γ_prior=4, δ_prior=0.1)

    @test sampled_σ²_1 ≈ [0.114303228]
    @test sampled_σ²_2 ≈ [0.058208240]
    @test sampled_σ²_3 ≈ [0.171990785, 0.179306053, 0.18531160]
    @test sampled_σ²_4 ≈ [0.062340451, 0.061543355, 0.06129676]
end

@testset "sampling λ (mixtrue scale) test" begin
    # Random sampled input with seed:
    Random.seed!(0)
    ϵ = randn(5)
    ϵ2 = randn(5, 2)

    sampled_ν_1 = sampling_mixture_scale(ϵ, [0.1], [5])
    sampled_ν_2 = sampling_mixture_scale(ϵ, σ²=[0.1], ν=[3])
    @test sampled_ν_1 ≈ [0.46030952, 0.01957489, 0.91558343, 0.89474495, 1.09754507]
    @test sampled_ν_2 ≈ [0.36612055, 0.33247866, 0.51515191, 1.18814004, 0.22087024]
    
    sampled_ν_3 = sampling_mixture_scale(ϵ2, [0.1, 0.3], [5, 6])
    sampled_ν_4 = sampling_mixture_scale(ϵ2, σ²=[0.1, 0.4], ν=[3, 4])
    @test sampled_ν_3 ≈ [[1.08340060, 1.39515432, 0.80232465, 0.88058962, 0.16187903] [1.68493236, 1.48817376, 1.19266668, 0.60232980, 0.84324196]]
    @test sampled_ν_4 ≈ [[0.65987117, 0.85963046, 1.30833576, 0.81158283, 0.20084423] [2.22305275, 1.21759091, 1.01091807, 0.55266799, 0.66175961]]
end

@testset "sampling ν (degree of freedom) test" begin
    # Random sampled input with seed:
    Random.seed!(0)
    λ = randn(5).^2
    λ2 = randn(5, 2).^2

    sampled_λ_1 = sampling_df(λ)
    sampled_λ_2 = sampling_df(λ, 5)
    sampled_λ_3 = sampling_df(λ, [5])
    sampled_λ_4 = sampling_df(λ, 5, 20, 0.1)
    sampled_λ_5 = sampling_df(λ, ν_previous=[5], ν_prior=[20], hm_variance=0.1)

    sampled_λ_6 = sampling_df(λ2)
    sampled_λ_7 = sampling_df(λ2, 5)
    sampled_λ_8 = sampling_df(λ2, [5, 3])
    sampled_λ_9 = sampling_df(λ2, 5, 20, 0.1)
    sampled_λ_10 = sampling_df(λ2, ν_previous=5, ν_prior=[20, 4], hm_variance=0.1)

    @test sampled_λ_1 ≈ [29.90621365]
    @test sampled_λ_2 ≈ [3.75960363]
    @test sampled_λ_3 ≈ [5.10984673]
    @test sampled_λ_4 ≈ [4.80986691]
    @test sampled_λ_5 ≈ [4.97197705]
    @test sampled_λ_6 ≈ [30.05571083, 30.23685716]
    @test sampled_λ_7 ≈ [4.61866147, 5.20419340]
    @test sampled_λ_8 ≈ [4.85186115, 3.25343681]
    @test sampled_λ_9 ≈ [4.43995314, 5.0]
    @test sampled_λ_10 ≈ [4.89819263, 4.94346203]
end

@testset "sampling factor loading test" begin
    # Random sampled input with seed:
    Random.seed!(0)
    y = randn(100, 3)
    f = randn(100, 2)
    f2 = randn(100, 1)

    sampled_loading_1 = sampling_factor_loading(y, f, 0, 1, [0.2, 0.3, 0.4])
    sampled_loading_2 = sampling_factor_loading(y, f, [[0.5, 0.6, 0.4] [0.5, 0.6, 0.4]], 
                                                [[0.1, 0.1, 0.2] [0.1, 0.1, 0.2]], [0.2, 0.3, 0.4])
    sampled_loading_3 = sampling_factor_loading(y, f2, 0, 1, [0.2, 0.3, 0.4])
    sampled_loading_4 = sampling_factor_loading(y, f2, [0.5, 0.6, 0.4], [0.1, 0.1, 0.2], [0.2, 0.3, 0.4])
    
    @test sampled_loading_1 ≈ [0.0214861477 0.0; 0.00745356477 0.00019266567; 0.1538865454 0.033081841]
    @test sampled_loading_2 ≈ [0.0293607108 0.0; 0.00777292639 0.00219933746; 0.0361523677 0.044494756]
    @test sampled_loading_3 ≈ [0.06257023, 0.21357754, -0.24474075]
    @test sampled_loading_4 ≈ [0.0721294182; 0.2115473233; -0.277806024]
end

@testset "Kalman filter test" begin
    # Random sampled input with seed:
    Random.seed!(0)
    
    y = randn(100, 1)
    y[20, 1] = NaN  # adding some missing value too
    # Tests with different input data types:
    H = [1. 0. 0.] # m=1, k=3
    R = [0.04]
    G = [[1 1 0.5];
        [0 1 1];
        [0 0 1]]
    Q = [[1 0 0];
        [0 1e-4 0];
        [0 0 1e-6]]
    x, P = kalman_filter(y, H, R, G, Q)
    @test true  # we only check whether the run was successfull

    x0 = zeros(3)
    P0 = Diagonal([10.; 1.; 1.])
    x, P = kalman_filter(y, H, R, G, Q, x0=x0, P0=P0)
    @test true

    R = 0.04
    x, P = kalman_filter(y, H, R, G, Q, x0=x0, P0=P0)
    @test true

    y = randn(100, 2)
    y[2, 1] = NaN
    H = [[1. 0. 0.];
         [0. 1. 0.]]  # m=2, k=3
    x, P = kalman_filter(y, H, R, G, Q, x0=x0, P0=P0)
    @test true

    H = [1, 0.5]  # m=2, k=1
    R = 0.1
    G = 0.2
    Q = 0.1
    x, P = kalman_filter(y, H, R, G, Q)
    @test true

    G = [-0.1]
    x, P = kalman_filter(y, H, R, G, Q)
    @test true

    x, P = kalman_filter(y, H, R, G, Q, μ=0.3, x0=2, P0=0.1)
    @test true

    x, P = kalman_filter(y, H, R, G, Q, μ=0.3, x0=[2], P0=0.1)
    @test true

    G = randn(1, 1, 100)
    x, P = kalman_filter(y, H, R, G, Q, μ=0.3, x0=[2], P0=0.1)
    @test true

    R = [0.2, 0.4]
    x, P = kalman_filter(y, H, R, G, Q, μ=0.3, x0=[2], P0=0.1)
    @test true
end