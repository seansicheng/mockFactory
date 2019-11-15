import Corrfunc
from Corrfunc.theory import wp
from Corrfunc.theory import xi
from Corrfunc.theory.DDsmu import DDsmu
import numpy as np

class Clustering(object):
	def __init__(self, rbins):
		self.rbins = rbins

	def xi_wp_cubic_mock(self, mock, size=1000, xi_name=None, wp_name=None, verbose=False):
		if type(mock) == str:
			mock = np.loadtxt(mock)
		
		results_wp = wp(size, 80, 1, self.rbins, mock[:,0], mock[:,1], np.clip(mock[:,2]+mock[:,5]/100, 10**-5, 0.99998*size), verbose=verbose, output_rpavg=True)
		results_DDsmu = DDsmu(1, 1, self.rbins, 1, 20, mock[:,0], mock[:,1], np.clip(mock[:,2]+mock[:,5]/100, 10**-5, 0.99998*size), boxsize=size, verbose=verbose, output_savg=True)

		density = len(mock)/1000**3

		rmin = np.array([line[0] for line in results_DDsmu])
		rmax = np.array([line[1] for line in results_DDsmu])
		ravg = np.array([line[2] for line in results_DDsmu])
		mu_max = np.array([line[3] for line in results_DDsmu])
		mu_min = mu_max - 0.05
		DD = np.array([line[4] for line in results_DDsmu])

		vol = 2/3*np.pi*(rmax**3 - rmin**3)

		vol *= 2*(mu_max - mu_min)
		xi = DD/(density*len(mock)*vol)-1

		r = ravg.reshape(20,20).mean(axis=1)
		mono = xi.reshape(20,20).mean(axis=1)
		quad = (2.5*(3*(mu_max-0.025)**2-1) * xi).reshape(20,20).mean(axis=1)

		if wp_name:
			np.savetxt(wp_name, results_wp, fmt="%.6f")
		if xi_name:
			np.savetxt(xi_name, np.array([(self.rbins[:-1] + self.rbins[1:])/2, mono, quad]).T, fmt="%.6f")

		return mono, quad, np.array([line[3] for line in results_wp])