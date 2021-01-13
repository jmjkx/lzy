from sklearn.decomposition import PCA


def myPCA(dim):
    def calMatrix(X):
        pca.fit(X)
        M = pca.components_
        return M.T
    pca = PCA(n_components=dim, svd_solver = 'auto')
    return calMatrix
