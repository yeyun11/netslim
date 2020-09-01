import numpy
from scipy.stats import skewnorm

TH_SKEWNORM_PPF = 0.99
PPF_TUNE_X0 = 0.707
TH_SKEWNESS = -5
MIN_SCALE = 0.01
MIN_SCALING_FACTOR = 1e-18
OT_DISCARD_PERCENT = 0.001


def css_thresholding(x, percent=OT_DISCARD_PERCENT):
    x_np = numpy.array(x)
    x_np[x_np < MIN_SCALING_FACTOR] = MIN_SCALING_FACTOR
    x_sorted = numpy.sort(x_np)
    x2 = x_sorted**2
    Z = x2.sum()
    energy_loss = 0
    for i in range(x2.size):
        energy_loss += x2[i]
        if energy_loss / Z > percent:
            break
    th = (x_sorted[i-1] + x_sorted[i]) / 2 if i > 0 else 0
    return th


def find_closest_index(x_sorted, target):
    for i in range(len(x_sorted)):
        if x_sorted[i] <= target <= x_sorted[i + 1]:
            break
    return i if (target - x_sorted[i] > x_sorted[i + 1] - target) else (i + 1)


def sep_two_skewed_normals(x, th_init):
    x0 = x[x < th_init]
    x1 = x[x >= th_init]

    if x0.size == 0:
        return th_init, (x.min() - 1, x1.mean(), 0.01, x1.std(), 0, 0)
    if x1.size == 1:
        a1 = TH_SKEWNESS
        m1 = x0.mean()
        s1 = MIN_SCALE
    else:
        a1, m1, s1 = skewnorm.fit(x1)
        if a1 > TH_SKEWNESS:
            a1, m1, s1 = skewnorm.fit(x1, f0=TH_SKEWNESS)
    if x0.size == 1:
        a0 = TH_SKEWNESS
        m0 = x0.mean()
        s0 = MIN_SCALE
    else:
        a0, m0, s0 = skewnorm.fit(x0)
        if a0 > TH_SKEWNESS:
            a0, m0, s0 = skewnorm.fit(x0, f0=TH_SKEWNESS)

    num_x0_last = x0.size
    num_change = 1
    x_sorted = sorted(x)
    nums_x0 = [num_x0_last, ]
    while num_change:
        # E, binary search for new th
        i0 = int(x0.size/2)
        i1 = x.size - int(x1.size/2)
        while i1 - i0 > 1:
            i = int((i0 + i1) / 2)
            p0 = skewnorm.pdf(x_sorted[i], a0, m0, s0) - skewnorm.pdf(x_sorted[i], a1, m1, s1)
            if p0 > 0:
                i0 = i
            else:
                i1 = i

        th = (x_sorted[i0] + x_sorted[i1]) / 2

        x0 = x[x < th]
        x1 = x[x >= th]

        # M
        if x0.size == 0:
            break
        if x1.size == 1:
            a1 = TH_SKEWNESS
            m1 = x0.mean()
            s1 = MIN_SCALE
        else:
            a1, m1, s1 = skewnorm.fit(x1)
            if a1 > TH_SKEWNESS:
                a1, m1, s1 = skewnorm.fit(x1, f0=TH_SKEWNESS)
        if x0.size == 1:
            a0 = TH_SKEWNESS
            m0 = x0.mean()
            s0 = MIN_SCALE
        else:
            a0, m0, s0 = skewnorm.fit(x0)
            if a0 > TH_SKEWNESS:
                a0, m0, s0 = skewnorm.fit(x0, f0=TH_SKEWNESS)

        # update
        num_change = x0.size - num_x0_last
        num_x0_last = x0.size
        if num_x0_last not in nums_x0:
            nums_x0.append(num_x0_last)
        else:
            break

    th = min(skewnorm.ppf(TH_SKEWNORM_PPF, a0, m0, s0), th)
    # extreme case that under very weak L1 constraint, negligible cluster is fitted with large sigma
    if s1 > 0.1 and s0 / s1 > 10:
        th = min(skewnorm.ppf(1e-4, a1, m1, s1), th)
    return th, (m0, m1, s0, s1, a0, a1)


def em_thresholding(x, alpha=1e-3):
    x_np = numpy.array(x)
    x_np[x_np < MIN_SCALING_FACTOR] = MIN_SCALING_FACTOR
    x_sorted = sorted(x_np)
    prune_th_init = alpha * x_np.max()
    if prune_th_init < x_sorted[2]:
        prune_th_init = (x_sorted[1] + x_sorted[2]) / 2
    x_log10 = numpy.log10(x_np)
    th_log10, (l0, l1, s0, s1, a0, a1) = sep_two_skewed_normals(x_log10, numpy.log10(prune_th_init))
    th = numpy.power(10, th_log10)
    if th > x_np.max():  # Failure case, seldom
        th = prune_th_init
    return th, (l0, l1, s0, s1, a0, a1)


def kmeans_thresholding(x):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(numpy.log10(numpy.array(x)).reshape(-1, 1))
    th = numpy.power(10, kmeans.cluster_centers_.mean())
    log10_info = sorted([_[0] for _ in kmeans.cluster_centers_])
    return th, log10_info


"""
import math
from scipy.optimize import newton
INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
PERCENT_FOR_ESTIMATION = 0.9

def normal_pdf(x, m, s):
    return INV_SQRT_2PI / s * numpy.exp(-0.5 * (x - m) ** 2 / s ** 2)


def calculate_th(m0, s0, m1, s1, w0):
    th_init = m1 + (m0 - m1) * s1 / (s1 + s0)
    try:
        th = newton(
            lambda x: w0 * (x - m0) * normal_pdf(x, m0, s0) / s0 ** 3 + (1 - w0) * (x - m1) * normal_pdf(x, m1,
                                                                                                         s1) / s1 ** 3,
            th_init
        )
    except:
        th = th_init
    return th


def approximate_th(m0, s0, m1, s1, w0):
    #approximate threshold with the equal point for better efficiency

    s0_sq = s0 * s0
    s1_sq = s1 * s1
    m0_sq = m0 * m0
    m1_sq = m1 * m1
    M = w0 * s1 / (1 - w0) / s0
    a = 0.5 * (1 / s0_sq - 1 / s1_sq)
    b = m1 / s1_sq - m0 / s0_sq
    c = 0.5 * (m0_sq / s0_sq - m1_sq / s1_sq) - math.log(M)

    th = 0.5 * (-b - math.sqrt(b * b - 4 * a * c)) / a
    if th < m0 or th > m1:
        th = 0.5 * (-b + math.sqrt(b * b - 4 * a * c)) / a
    return th


def em_two_gaussians(x, th_init, tol=1e-3):
    x0 = x[x < th_init]
    x1 = x[x >= th_init]

    if x0.size == 0:
        return th_init, (x.min() - 1, x1.mean(), 10 * tol, x1.std(), 0)

    m0 = x0.mean()
    m1 = x1.mean()
    s0 = x0.std()
    s1 = x1.std()
    w0 = x0.size / x.size

    if x1.size < 2:
        return th_init, (m0, m1, s0, 10 * tol, w0)
    elif x0.size < 2:
        return th_init, (m0, m1, 10 * tol, s1, w0)

    num_x0_last = x0.size
    num_change = 1
    while num_change:
        # E
        th = calculate_th(m0, s0, m1, s1, w0)
        # th = approximate_th(m0, s0, m1, s1, w0)
        x0 = x[x < th]
        x1 = x[x >= th]

        # avoid singular values
        if x0.size < 2 or x1.size < 2:
            break

        # M
        m0 = x0.mean()
        m1 = x1.mean()
        s0 = x0.std()
        s1 = x1.std()
        w0 = x1.size / x.size

        # update
        num_change = x0.size - num_x0_last
        num_x0_last = x0.size

    # singular break for too large th
    if th > m1:
        x_sorted = sorted(x)
        th = (x_sorted[num_x0_last - 1] + x_sorted[num_x0_last]) / 2

    # non-overlap constraint
    th = min(th, m1 - 3 * s1)
    if th < m0 + 3 * s0:
        th = min(th, th_init)
    return th, (m0, m1, s0, s1, w0)
"""
