from pylab import *
from scipy.stats import linregress

data = loadtxt('standard.txt').T
l = linregress(data[0], data[1])
fig=figure()
ax = fig.add_subplot(111)
ax.plot(data[0], data[1], lw=2, label='Raw data', marker='o', ls=None)
ax.plot(data[0], l.slope * data[0] + l.intercept, lw=2, ls='--', label='Fitting')
ax.legend(fontsize=15, frameon=False)
ax.set_xlabel(r'$t$', fontsize=15)
ax.set_ylabel(r'$log(M)$', fontsize=15)
for s in ax.spines:
	ax.spines[s].set_linewidth(2)
ax.tick_params(axis='both', which='both', length=4, width=2, labelsize=13)
fig.set_tight_layout('tight')
show()
fig.savefig('%s.png' % ('standard'), dpi=330)
