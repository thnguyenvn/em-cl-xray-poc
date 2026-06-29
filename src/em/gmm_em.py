from sklearn.mixture import GaussianMixture


def fit_gmm(features, n_components=5, covariance_type="diag", max_iter=100, random_state=42):
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(features)
    return model


def gmm_bic(features, max_components=10, covariance_type="diag", random_state=42):
    results = []
    for k in range(1, max_components + 1):
        model = fit_gmm(features, k, covariance_type, random_state=random_state)
        results.append({"k": k, "bic": model.bic(features)})
    return results
