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