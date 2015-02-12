__author__ = 'const'
from spicybrew import features, reduction
from numpy import hstack

n = [5, 10, 20, 40]

# get data
usr = features.user_data()  # user replies / ground truth
cxtT = features.context(enable_descr=False)  # tags
cxtD = features.context(enable_descr=True)  # tags & descriptions
cnt, truth, feature_names = features.content()  # content, ground truth and fn

# reduce dimensionality
# content via feature selection
cntr_xtc = reduction.xtc(cnt, truth, nfeats=n[0])
cntr_skb = reduction.skbest(cnt, truth, nfeats=n[0])
# context via feature extraction
cxtTr = reduction.reduce_pca(cxtT, n=30)
cxtDr = reduction.reduce_pca(cxtD, n=30)

# TODO: concatenate the spaces
# aggregate feature spaces
aggregate = hstack((cntr_xtc, cxtDr))

# TODO: run evaluation
# evaluate -- 2 ways
# against the reference similarity matrix

# like mirex evaluation