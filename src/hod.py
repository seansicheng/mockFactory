from scipy.special import erf
import numpy as np

class HOD(object):
	"""
		HOD class for ELGs

		For central galaxies:
			1. Gaussian function
			N_cen = f_max * exp(- (log M - log M_min)^2 / 2 sigma_logM^2 )

			2. Gaussian + a softly rising step function
			N_cen = f_max * exp(- (log M - log M_min)^2 / 2 sigma_logM^2 ) + f_b * (1 + erf(log M - log M_b) / sigma_b) )


		For satellite galaxies:
			1. power law with cutoff
			N_sat = 0, if M<M_cut else ((M-M_cut)/M_1)^alpha

	"""
	def __init__(self, cen = 1, f_max = 0.15, M_min = 1.5*10**12, sigma_logM = 0.25,  \
								f_b = 0.1, M_b = 1.5*10**12, sigma_b = 0.2, \
								M_1 = 8*10**13, M_cut = 7*10**11, alpha = 1.0):
		self._params = locals()


	def _print_info(self):
		"""
			unfinished printing function.
		"""
		print("using HOD parameters:")
		return

	def N_cen(self, M):
		ncen = self._params["f_max"] * np.exp(- (np.log10(M) - np.log10(self._params["M_min"]))**2 / 2.0 / self._params["sigma_logM"]**2)
		if self._params["cen"] == 2:
			ncen += self._params["f_b"] * (1 + erf((np.log10(M) - np.log10(self._params["M_b"])) / self._params["sigma_b"]))
		return ncen

	def N_sat(self, M):
		nsat = np.zeros(len(M))
		nsat[M >= self._params["M_cut"]] = ((M[M >= self._params["M_cut"]] - self._params["M_cut"])/ self._params["M_1"])**self._params["alpha"]
		return nsat

