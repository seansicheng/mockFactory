import emcee
from mockFactory import MockFactory
import numpy as np 
from clustering import Clustering
from multiprocessing import Pool

class MCMC(object):
	def __init__(self):
		self.nwalkers = 32
		self.ndim = 7
		filename = "tutorial.h5"
		backend = emcee.backends.HDFBackend(filename)
		backend.reset(self.nwalkers, self.ndim)
		
		hod_params = {"M_min": 0, "galaxy_density": 0.00057, "boxsize": 1000, "log_halo_mass_bins": np.arange(10,15,0.1), \
				"halo_histo": np.loadtxt("../data/halo_central_histo.dat")}
		halofile = "../../ELG_HOD_optimization/data/halo_M200b_0.54980_for_mock.dat"
		self.mockfactory = MockFactory(halofile, boxsize=1000, cvir_fac=1, hod_parameters=hod_params)

		# clustering calculator
		rbins = np.logspace(np.log10(0.1), np.log10(70), 21)
		self.cluster = Clustering(rbins)

		# read xi and wp from data, read cov matrix
		self.clustering_data = np.loadtxt("../data/clustering_data.dat")
		self.scaled_cov = np.loadtxt("../data/scaled_cov.dat")

		#with Pool(10) as pool:
		#	self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_prob, backend=backend, pool = pool)
		#	self.run()

		self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_prob, backend=backend)
		self.run()

	def get_prior(self):
		"""
			prior distribution for log(M1), log(M_cut), alpha, MaxCen, sigma_logM, VBIAS, CVIR_FAC
		"""
		prior_mean = np.array([14.3, 12.16, 0.9, 0.092, 0.64, 0.668, 1.022])
		prior_std = np.array([0.3, 0.3, 0.1, 0.02, 0.05, 0.2, 0.2])
		return np.random.randn(self.nwalkers, self.ndim) * prior_std + prior_mean

	def log_prob(self, params):
		"""
			params should be in the format of [log10(M1), log10(M_cut), alpha, MaxCen, sigma_logM, VBIAS, CVIR_FAC]
		"""
		p = {"M_1": 10**params[0], "M_cut": 10**params[1]}
		for key, val in zip(["alpha", "f_max", "sigma_logM", "vbias", "cvir_fac"], params[2:]):
			p[key] = val
		self.mockfactory.update_params(p)
		mock = self.mockfactory.populateSimulation(verbose=True)
		xi0, xi2, wp = self.cluster.xi_wp_cubic_mock(mock[:, 1:], size=1000)
		clustering_mock = np.concatenate([xi0[3:], xi2[3:], wp[3:]])

		diff = self.clustering_data - clustering_mock
		res = -0.5*np.dot(diff, np.linalg.solve(self.scaled_cov,diff))
		print("[MCMC] log prob: {}".format(res))
		return res
		

	def run(self, iteration = 100):
		p0 = self.get_prior()
		self.sampler.run_mcmc(p0, iteration)


if __name__ == '__main__':
	mcmc = MCMC()
