__author__ = 'const'
from numpy import where, zeros, argsort, mean


def against_reference_sm(sys_sm, usr_sm, N=10):

    usm_sorted = zeros((120, 120))
    qsm_sorted = zeros((120, 120))
    for row in range(120):
        usm_sorted[row, :] = argsort(usr_sm[row, :])[::-1]
        qsm_sorted[row, :] = argsort(sys_sm[row, :])[::-1]
    s = zeros(N+1)
    u = zeros((120, N+1))
    for i in range(N+1):
        s[i] = .5**((i)/3.)
    for query in range(120):
        for i in range(N+1):
            # u[query, i] = (.5**(2./3.)) ** (usm[query, qsm_sorted[query, i]] - 1)
            u[query, i] = (.5**2.) ** (float(where(usm_sorted[query, :] == qsm_sorted[query, i])[0]))/3.
            # print np.where(usm_sorted[query, :] == qsm_sorted[query, i])[0]
    results = zeros(120)
    for query in range(120):
        for i in range(N+1):
            results[query] += s[i]*u[query, i]
    result = sum(results)/float(max(results)*120)
    return result


def mirex_eval(qsm, usm, length=5):
    # Genre, Artist, Album --> Category
    # Average % of Category matches in the top 5, 10, 20 & 50 results - Precision at 5, 10, 20 & 50
    # Average % of available Genre, Artist and Album matches in the top 5, 10, 20 & 50 results - Recall at 5, 10, 20, 50
    #   (just normalising scores when less than 20 matches for an artist, album or genre are available in the database)
    # Always similar - Maximum # times a file was in the top 5, 10, 20 & 50 results
    # % File never similar (never in a top 5, 10, 20 & 50 result list)
    # ------------- % of 'test-able' song triplets where triangular inequality holds
    # Plot of the "number of times similar curve" -
    #   plot of song number vs. number of times it appeared in a top 20 list with songs sorted
    #   according to number times it appeared in a top 20 list (to produce the curve).
    #   Systems with a sharp rise at the end of this plot have "hubs", while a long 'zero'
    #   tail shows many never similar results.
    # In addition computation times for feature extraction/Index-building and querying will be measured.

    usm_sorted = zeros((120, 120))
    qsm_sorted = zeros((120, 120))
    for row in range(120):
        usm_sorted[row, :] = argsort(usm[row, :])[::-1]
        qsm_sorted[row, :] = argsort(qsm[row, :])[::-1]
    # queryLength = [5, 10, 20, 50]
    category = zeros((120, 5))
    responseCount = zeros(120)
    # for length in queryLength:
    for query in range(120):
        for response in range(length):
            # if query/24 == int(qsm_sorted[query, response])/24:
            category[query, int(qsm_sorted[query, response])/24] += 1/float(length)
            responseCount[int(qsm_sorted[query, response])] += 1
    confusion_matrix = zeros((5, 5))
    for i in range(5):
        confusion_matrix[i, :] = mean(category[24*i:24*(i+1), :], 0)
    prec = zeros(5)
    reca = zeros(5)
    for i in range(5):
        prec[i] = confusion_matrix[i, i] / float(sum(confusion_matrix[:, i]))
        reca[i] = (sum(confusion_matrix[i, :]) - confusion_matrix[i, i]) / float(sum(confusion_matrix[i, :]))
    avgPrec = mean(prec)
    avgReca = mean(reca)

    # index in similarity matrix and number of times returned
    hubb = [int(where(responseCount == responseCount.max())[0][0]),
            responseCount.max()]
    PfilesNeverSimilar = len(where(responseCount == 0)[0])/120.
    # a list of how many times  a song was retrieved
    # to get it sorted in a plot do this
    # indices = np.argsort(responseCount)[::-1]
    # plot(range(120), responseCount[indices])
    PlotData = responseCount
    return avgPrec, avgReca, hubb, PfilesNeverSimilar, PlotData