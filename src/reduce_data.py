from sklearn.decomposition import PCA

def apply_pca(X, n_components=3):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(X)
    return reduced, pca
