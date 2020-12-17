using BayesianFactorModels
using Test
using Random

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
