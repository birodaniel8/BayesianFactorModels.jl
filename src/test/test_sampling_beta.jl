"""
Unit tests for sampling_beta
"""
model = LinearModel()
display(model)
using Test

@testset "sampling_beta_test" begin
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