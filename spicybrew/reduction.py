__author__ = 'const'
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from numpy import ndarray, ravel, argsort, zeros


############################
# dimensionality reduction #
############################


def reduce_pca(x, n=30):
    red = PCA(n_components=n)
    if type(x) != ndarray:
        return red.fit_transform(x.toarray())
    else:
        return red.fit_transform(x)


#####################
# feature selection #
#####################


def xtc(x, y, nfeats=10):
    clf = ExtraTreesClassifier()
    clf.fit(x, ravel(y)).transform(x)
    average = clf.feature_importances_
    indices = argsort(average)[::-1]

    red = zeros((x.shape[0], nfeats))
    for n in range(nfeats):
        red[:, n] = x[:, indices[n]]
    return red


def skbest(x, y, nfeats=10):
    clf = SelectKBest(chi2, k=nfeats)
    clf.fit(x, ravel(y))
    X = clf.fit_transform(x, ravel(y))
    return X  # , clf._get_support_mask()
