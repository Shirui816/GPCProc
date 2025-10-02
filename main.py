import warnings
import argparse
import numpy as np
import pandas as pd
from scipy.integrate import simps
from scipy.signal import detrend
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from baseline import snr
from baseline import baseline_cwt_br
import re


description = """
A simple GPC data analysis program, for
I dont f**king believe that a simple GPC
data analysis program is
worth 30000 RMB aka $5000 USD.
"""

arg_parser = argparse.ArgumentParser(
    description=description, formatter_class=RawTextHelpFormatter)
arg_parser.add_argument('-S', '--standard',
                        dest='standard', type=float, metavar="standard curve",
                        nargs=2, help="Standard curve, slope `k' and intercept `b0'.")
arg_parser.add_argument('-O', '--output',
                        metavar='Output file',
                        default='mwd', dest='out_put',
                        help="Optional, use `mwd' as default prefix.")
arg_parser.add_argument('gpc_file(s)',
                        nargs=+, type=str
                        help='GPC data file(s).')
arg_parser.add_argument('-M', '--method',
                        nargs=+, type=str, dest='method', metavar='Detector(s).',
                        help='GPC detector(s).')
						
						
args = arg_parser.parse_args()
alvars = vars(args)
k, b0 = alvars['standard']
#k, b0 = -0.5005413052736007, 11.278597269990836  # test data.
# TODO: add combination mode of detectors.
detector = alvars['method']
#mark_houwink = pd.read_csv('mh.dat', sep='_')
# MH equation: K_1 M_1^{a_1} = K_2 M_2^{a_2}, universal calibration curve.
# k2, b0_2 = k * (a_2 + 1)/(a_1 + 1), b0 + k * np.log(K_2/K_1)/np.log(10)  # mk equation to transfer M1->M2
# k, b as function of K, a, if we know the k, b0 and K_1, a_1 for a certain polymer.



def std_func(t, k, b0):
    r'''GPC standard curve t->M.'''
    return np.exp((k*t + b0)*np.log(10))



#TODO: add dedrend and find_peak here, or perhaps a GUI?
# RI signal = K_{RI} * dn/dc * c
# LS signal = K_{LS} * (dn/dc)^2 * Mw * c
# Visc. signal = K_{visc} * IntrisicViscosity * c

ret = []
files = alvars['gpc_file(s)']
for i, f in enumerate(files):
	data = np.loadtxt(f)
	t, y = data.T[0], data.T[1]
	while True:
		plt.plot(t, y)
		plt.show()
		s, e = [float(_) for _ in re.split(',\s*|\s+',input("Trim the GPC, start, end: ")) if _ != '']
		inds = np.logical_and(t>s, t<e)
		t1 = t[inds]
		y1 = y[inds]
		plt.plot(t1, y1)
		plt.show()
		if input("End trim? [y|n] ") == 'y':  # trim data
			break
	b = baseline_cwt_br(t1, y1)  # baseline correction
	plt.plot(t1, y1-b)
	plt.show()
	t = t1
	y = y1
	m, i = std_func(t, k, b0), y
	if detector == 'RI':
		p  = i/m * 1/abs(k*np.log(10))  # P(t) -> P(M); the 1/m factor is used to transform P(Log M) -> P(M)
	np.savetxt('%s_%03d.txt' % (alvars['out_put'], i), np.vstack([m, np.abs(p/simps(p, m))]).T, fmt='%.6f')
