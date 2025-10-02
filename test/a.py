import argparse
import numpy as np
from scipy.stats import linregress
from scipy.interpolate import interp1d
from argparse import RawTextHelpFormatter
from scipy.integrate import simps

description = """Decomposite GPC data.
Written by Shirui shirui816@gmail.com
### data format (GPC data):
#time response
### Or
#time known_component mixed_signal
"""

arg_parser = argparse.ArgumentParser(
    description=description, formatter_class=RawTextHelpFormatter)
arg_parser.add_argument('-S', '--standard',
                        metavar='strandard_curve', dest='std', type=str,
                        help='Standard curve; required.')
arg_parser.add_argument('-o', '--output',
                        metavar='Output free energy file',
                        default='data', dest='out_put',
                        help="Optional, use 'data' as default",)
arg_parser.add_argument('-X', '--snr',
                        default=0, metavar='Signal Noise Ratio', type=float,
                        dest='snr',
                        help="Optional, using 0 for default.")
arg_parser.add_argument('datafile(s)',
                        nargs='+',
                        help='Data file(s), if multiple files are given, '
			'make sure the sequence is <known> <all>.')

args = arg_parser.parse_args()
alvars = vars(args)

_NOISE_THRESH = 0

datas = alvars['datafile(s)']

if len(datas) == 1:
	_ = np.loadtxt(datas[0]).T
	TIME_KNOWN = TIME_ALL = _[0]
	KNOWN = _[1]
	ALL = _[2]
else:
	_ = np.loadtxt(datas[0]).T
	__ = np.loadtxt(datas[1]).T
	TIME_KNOWN = _[0]
	KNOWN = _[1]
	TIME_ALL = __[0]
	ALL = __[1]

# Clean the data simply. Or some smooth algorithms do better.
KNOWN[KNOWN<0] = 0
ALL[ALL<0] = 0
KNOWN /= simps(KNOWN, TIME_KNOWN)
ALL /= simps(ALL, TIME_ALL)

std = [linregress(_[0], _[1]) for _ in [np.loadtxt(alvars['std']).T]][0]

def std_Func(x):
	r'''GPC standard curve t->M.'''
	return np.exp((std.slope * x + std.intercept)*np.log(10))

def inv_std_Func(x):
	r'''GPC standard curve M->t.'''
	return (np.log(x) - std.intercept) / std.slope

def wd(s, k, l):
	r'''Wiener-deconvolution.'''
	k = np.hstack([k, np.zeros(s.shape[0]-k.shape[0])])
	h = np.fft.fft(k)
	H = np.abs(h) ** 2
	return np.fft.ifft(np.fft.fft(s) / h * (H/(H+l))).real

if __name__ == "__main__":
	M_known = std_Func(TIME_KNOWN)/1000
	M_all = std_Func(TIME_ALL)/1000
	#print(std_Func(simps(KNOWN*TIME_KNOWN, TIME_KNOWN)/simps(KNOWN, TIME_KNOWN))/1000,std_Func(simps(ALL*TIME_ALL, TIME_ALL)/simps(ALL, TIME_ALL))/1000)
	KNOWN_1 =abs(KNOWN/ M_known/std.slope/np.log(10))
	ALL_1  = abs(ALL/M_all/std.slope/np.log(10))
	#KNOWN_1 = KNOWN
	#ALL_1 = ALL
	#Mk = np.linspace(M_known.min(), M_known.max(), int(M_known.max()-M_known.min())+1)
	#Ma = np.linspace(M_all.min(), M_all.max(), int(M_all.max()-M_all.min())+1)
	Mk = np.linspace(M_known.min(), M_known.max(), 3000)
	Ma = np.linspace(M_all.min(), M_all.max(), 3000)
	print(Mk.max(), Mk.min(), Ma.max(), Ma.min())
	intK = interp1d(M_known, KNOWN_1/M_known, kind='cubic', assume_sorted=False)
	intA = interp1d(M_all, ALL_1/M_all, kind='cubic', assume_sorted=False)
	K = intK(Mk)
	A = intA(Ma)
	K /= simps(K, Mk)
	A /= simps(A, Ma)
	print(simps(Mk*K,Mk), simps(Ma*A,Ma))
	#RES = A - (0.62*2-1)*K
	RES = A
	RES[RES<0]=0
	RES /= simps(RES, Ma)
	#RES[RES<0]=0
	U = wd(np.pad(RES, (0,A.shape[0]), constant_values=0, mode='constant'), K, alvars['snr'])
	U = np.fft.fftshift(U) # for range of U is symmetric.
	Mu = np.linspace(Ma.min()-Mk.max(), Ma.max()-Mk.min(), U.shape[0])
	U /= simps(U, Mu)
	U[Mu<0] = 0
	U[U<_NOISE_THRESH] = 0 # Clean the data simply.
	np.savetxt('%s_distribution.txt' % (alvars['out_put']), np.vstack([Mu[Mu>0], U[Mu>0]]).T, fmt='%.6f')
	PDI = simps(U*(Mu)**2, Mu)/simps(U*Mu, Mu)**2
	print(simps(U*Mu, Mu))
	from pylab import *
	fig = figure()
	ax = fig.add_subplot(111)
	cls_ = cm.tab10.colors[1:5]
	ax.plot(M_known, abs(KNOWN_1/simps(KNOWN_1, M_known)), label='PS, normalized', c='k',ls='--', lw=2)
	ax.plot(Ma, RES/simps(RES, Ma), label='PS-b-PMMA, normalized', c='k', ls='-.', lw=2)
	#ax.plot(Mk, K*Mk, label='PS, interpolated',c=cls_[1],lw=2,alpha=.75)
	#ax.plot(Ma, RES*Ma, label='PS-b-PMMA, interpolated', c=cls_[2],lw=2,alpha=.75)
	UU = U*Mu
	ax.plot(Mu, UU/simps(UU, Mu), label='PMMA',c=cls_[3],lw=2)
	# Verify
	K[K<0] = 0
	Kprob = K/K.sum()
	Uprob = U/U.sum()
	#sampleu = np.random.choice(Mu, p=Uprob, size=10000)
	#samplek = np.random.choice(Mk, p=Kprob, size=10000)
	#sample_res = np.random.choice(Mk, p=Kprob, size=int(10000*(62-38)/38))
	#ax.hist(sampleu, bins=50, density=True, label='PMMA', alpha=.75,color=cls_[3])
	#ax.hist(samplek, bins=50, density=True, label='PS', alpha=.75,color=cls_[1])
	#ax.hist(sampleu+samplek, bins=150, density=True, label='PS-b-PMMA', alpha=.75, color=cls_[2])
	#ax.hist(np.append(sampleu+samplek, sample_res), bins=150, density=True, label='PS-b-PMMA', alpha=.75,color=cls_[2])
	ax.legend(fontsize=15, frameon=False)
	a, b = xlim()
	ax.set_title(r'$PDI_\mathrm{PMMA}=%.6f$' % (PDI), fontsize=15)
	ax.set_xlim(-13,250)
	#ax.set_xscale('log')
	ax.set_xlabel(r'$M/kDa$', fontsize=15)
	ax.set_ylabel(r'$P$', fontsize=15)
	for s in ax.spines:
		ax.spines[s].set_linewidth(2)
	ax.tick_params(axis='both', which='both', length=4, width=2, labelsize=13)
	fig.set_tight_layout('tight')
	show()
	fig.savefig('%s.png' % (alvars['out_put']), dpi=330)
