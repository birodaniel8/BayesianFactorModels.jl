"""
This component sets an initial values for the factors for the normal factor model by the following formula:

\$PCA_{1:k} Q R_{1:k}\$

where \$Q\$ and \$R\$ are the matrixes from QR decomposition and \$PCA\$ is the matrix of the principal components 
ordered by the magnitude of the corresponding eigen values.
"""
function factor_initialize(x::AbstractMatrix, k::Int)
    pca_loading, pca_component = sorted_pca(mapslices(zscore, x, dims=1), 2)
    q, r = qr(pca_loading')
    factor0 = pca_component * q * r[:, 1:k]
    return factor0
end

"""
This component calculates the first \$k\$ principal components and the normalized loading matrix.
"""
function sorted_pca(x::AbstractMatrix, k::Int)
    n = size(x, 2)
    x_eigvecs = eigvecs(x' * x)
    pca_loading = sqrt(n) * x_eigvecs[:, n:-1:n-(k-1)]
    pca_component = x * pca_loading / n
    return [pca_loading, pca_component]
end
