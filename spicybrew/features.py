__author__ = 'const'
from io import *
import gaia2 as g2
from numpy import zeros, mean, asarray, size, diag
from sklearn.feature_extraction.text import TfidfVectorizer


def content(filename='data/idList.json', dataset='data/fs5c.db'):
    idList = jread(filename)
    analysis = 'Freesound5ClassDataset/analysis_files/'

    cdb = g2.DataSet()
    cdb.load(dataset)
    cdb = g2.transform(cdb, 'removevl')
    cdb = g2.transform(cdb, 'fixlength')
    cdb = g2.transform(cdb, 'cleaner')
    cdb = g2.transform(cdb, 'normalize')

    descriptorNames = list(cdb.layout().descriptorNames())
    # we need to remove these as they contain strings
    descriptorNames.pop(descriptorNames.index('.metadata.version.essentia'))
    descriptorNames.pop(descriptorNames.index('.tonal.chords_key'))
    descriptorNames.pop(descriptorNames.index('.tonal.chords_scale'))
    descriptorNames.pop(descriptorNames.index('.tonal.key_key'))
    descriptorNames.pop(descriptorNames.index('.tonal.key_scale'))

    feats = []
    for dn in descriptorNames:
        if type(cdb.point(cdb.pointNames()[0])[dn]) == tuple:
            for idx in range(len(cdb.point(cdb.pointNames()[0])[dn])):
                feats.append(dn + '.' + str(idx))
        else:
            feats.append(dn)
    print len(feats)

    tosklearn = zeros((120, len(feats)))
    feats = []  # this is not a clean way to do this. FIX ME
    for dn in descriptorNames:
        if type(cdb.point(cdb.pointNames()[0])[dn]) == tuple:
            for idx in range(len(cdb.point(cdb.pointNames()[0])[dn])):
                for p in idList:
                    if cdb.point(analysis+str(p)+'.yaml')[dn]:
                        tosklearn[idList.index(p), len(feats)] = cdb.point(analysis+str(p)+'.yaml')[dn][idx]
                    else:
                        tosklearn[idList.index(p), len(feats)] = 0
                feats.append(dn + '.' + str(idx))
        else:
            for p in idList:
                tosklearn[idList.index(p), len(feats)] = cdb.point(analysis+str(p)+'.yaml')[dn]
            feats.append(dn)
    target, cat = zeros((120, 1)), 0
    for point in idList:
        if idList.index(point) % 24 == 0:
            cat += 1
        # ds2sklearn[idList.index(point), len(feats)-1] = cat
        target[idList.index(point)] = cat
    # feats.append('category')
    print size(tosklearn, 0), size(tosklearn, 1)
    return asarray(tosklearn), target, feats


def context(enable_descr=True, mindf=.06, maxdf=.99):
    # we want the dataset
    ids = jread('data/idList.json')
    metadata = 'data/Freesound5ClassDataset/api_sound_metadata/'
    md = []
    description = ''

    for id in ids:
        text = jread(metadata + str(id) + '.json')
        if enable_descr:
            description = text['description']
        tags = ' '.join([str(tag) for tag in text['tags']])
        final_thing = ' '.join([description, tags])
        md.append(final_thing)

    # prepare vectorizer
    if enable_descr:
        cv = TfidfVectorizer(min_df=mindf,
                             max_df=maxdf,
                             stop_words='english')
    else:
        cv = TfidfVectorizer()
    cv.fit(md)
    vec = cv.transform(md)
    return vec


def user_data(idlist='data/idList.json', user_answers='data/answers.txt', norm=False, todiag=0):
    ids = jread(idlist)
    ans = jread(user_answers)
    usm = zeros((120, 120))

    for an in ans.keys():
        id = ans[str(an)]['file_ids']
        usm[ids.index(id[0]), ids.index(id[1])] = mean(ans[str(an)]['scorelist'])
        usm[ids.index(id[1]), ids.index(id[0])] = mean(ans[str(an)]['scorelist'])
    if norm:
        usm = usm/10.
    return usm + diag([todiag] * usm.shape[0])
