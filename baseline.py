import numpy as np
from scipy import signal
from scipy import sparse


def baseline_corr(y, lam, p, niter=10):
    r'''Asymmetric Least Squares Smoothing. by P. Eilers and H. Boelens in 2005
	'''
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

	
def snr(y):
	return (y.max()-y.min())/np.var(np.diff(y))**0.5


def baseline_cwt_br(t, y, widths=np.arange(1, 61), wavelet=signal.ricker, theta_iter=2, deg=5):
	r'''Automatic baseline detect algorithm via CWT.
	by Carlo G. Bertinetto and Tapani Vuorinen, 2014
	DOI: 10.1366/13-07018, Volume 68, Number 2, 2014. APPLIED SPECTROSCOPY
	'''
	cwt_mat = signal.cwt(y, wavelet, widths)

	# for HIGH SNR, HIGH SNR, HIGH SNR!!!
	# H := -\sum_i |c_i|/\sum_j |c_j| \log{|c_i|/\sum_j |c_j|}
	# TODO: Add bootstrap sampling for low SNR systems.
	
	#H_w = np.log(np.abs(cwt_mat))-np.log(np.sum(np.abs(cwt_mat), axis=1))[:,np.newaxis]
	#H_w = np.sum(np.nan_to_num(np.abs(cwt_mat) * H_w), axis=1)
	#H_w = -1/np.sum(np.abs(cwt_mat), axis=1) * H_w
	
	#cwt_mat_p = np.where(cwt_mat < 0, 1e-8, cwt_mat)
	cwt_mat_p = np.abs(cwt_mat) # negative peaks are peaks
	cwt_mat_n = cwt_mat_p / np.sum(np.abs(cwt_mat_p), axis=1)[:, None]
	H_w = -np.sum(np.nan_to_num(cwt_mat_n * np.nan_to_num(np.log(cwt_mat_n))), axis=1)
	
	# Pick out the width that the cwt gives most informations on peaks.
	ind_wh = np.argmin(H_w)
	wh = widths[ind_wh]
	cwt_min = cwt_mat[ind_wh]

	# Evaluation of threashold
	# 10 times or n in 0.05%
	n = 200  # initial 200 bins
	_count = 0                                                                                                            
	while _count < theta_iter:
		n_old = n
		p, e = np.histogram(cwt_min, bins=n)
		e = e[:-1]
		ind = np.logical_and(p>=p.max()/3, p<=p.max())
		sigma = (np.sum((e[ind])**2 * p[ind])/p[ind].sum())**0.5
		n = int(8*(cwt_min.max()-cwt_min.min())/sigma)
		if (n-n_old)/n_old < 0.05:
			_count += 1
	
	n_ind = np.logical_and(cwt_min > -3 * sigma, cwt_min < 3 * sigma)
	theta = sigma * (0.6 + 10 * cwt_min[n_ind].shape[0]/cwt_min.shape[0])
	baseline_ind = np.logical_and(cwt_min>-theta, cwt_min<theta)

	# Iterative Poly-fit until converge
	b_p = y[baseline_ind]
	b_t = t[baseline_ind]
	converge = False
	while not converge:
		n = b_p.shape[0]//100
		# segments length more than N // 100 elements
		b_p_old = np.copy(b_p)
		p = np.polyfit(b_t, b_p, deg=6)
		pi = np.poly1d(p)(b_t)
		std = np.mean((pi-b_p)**2) ** 0.5
		ind1 = b_p - pi < 1.5 * std
		# remove yi - pi > 1.5 * std points from baseline set
		b_p = b_p[ind1]
		b_t = b_t[ind1]
		p = np.polyfit(b_t, b_p, deg=deg)
		pi = np.poly1d(p)(t)
		ind2 = y - pi < 0
		segs = np.cumsum(np.r_[
			0, np.diff(np.flatnonzero(np.diff(ind2))+1, prepend=0, append=ind2.shape[0])
		])
		for i in range(1, segs.shape[0]):
			seg = ind2[segs[i-1]:segs[i]]
			if seg[0] and (seg.shape[0] >= n):
				b_p = np.append(b_p, y[segs[i-1]+seg.shape[0]//2])
				b_t = np.append(b_t, t[segs[i-1]+seg.shape[0]//2])
		# Add t[l+k//2], y[l+k//2] to the baseline set if
		# l, l+1, l+2... l+k is a continous sequence that y < p(t)
		# loop to fit till the baseline point set converge.
		if b_p_old.shape[0] == b_p.shape[0]:
			if np.allclose(b_p_old, b_p):
				converge=True
	return p(t)  # return the baseline
