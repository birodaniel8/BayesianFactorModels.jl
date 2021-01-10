using Documenter
using BayesianFactorModels

makedocs(
    sitename = "BayesianFactorModels",
    format = Documenter.HTML(),
    modules = [BayesianFactorModels],
    pages=[
        "Home" => "index.md",
        "Models" => [
            "linear_models.md",
            "linear_factor_models.md",
            "dynamic_factor_models.md"
        ],
        "Sampling methods" => [
            "sampling.md",
        ],
        "Filters" => [
            "kalman_filter.md",
        ],
        "Data generating processes" => [
            "dgp.md"
        ]
    ],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
