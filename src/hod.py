from scipy.special import erf
import numpy as np
import abc

class BaseHOD(object, metaclass = abc.ABCMeta):
	"""
		Base HOD abstract class
	"""
	def __init__(self):
		self._params = {}

	@abc.abstractmethod
	def N_cen(self, M):
		raise AttributeError("N_cen need to be implemented")

	@abc.abstractmethod
	def N_sat(self, M):
		raise AttributeError("N_sat need to be implemented")

	def _print_info(self):
		"""
			unfinished printing function.
		"""
		print("using HOD parameters:")
		return



class HOD(BaseHOD):
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
								M_1 = 8*10**13, M_cut = 7*10**11, alpha = 1.0, 
								galaxy_density = None, boxsize = None, log_halo_mass_bins = None, halo_histo = None):
		self._params = locals()
		if M_min == 0:
			if galaxy_density is None or boxsize is None or log_halo_mass_bins is None or halo_histo is None:
				raise ValueError("need galaxy_density, boxsize, log_halo_mass_bins and halo_histo to compute M_min")
			self.mass = 10**((self._params["log_halo_mass_bins"][:-1] + self._params["log_halo_mass_bins"][1:])/2)
			self.find_Mmin()


	def N_cen(self, M):
		ncen = self._params["f_max"] * np.exp(- (np.log10(M) - np.log10(self._params["M_min"]))**2 / 2.0 / self._params["sigma_logM"]**2)
		if self._params["cen"] == 2:
			ncen += self._params["f_b"] * (1 + erf((np.log10(M) - np.log10(self._params["M_b"])) / self._params["sigma_b"]))
		return ncen

	def N_sat(self, M):
		nsat = np.zeros(len(M))
		nsat[M >= self._params["M_cut"]] = ((M[M >= self._params["M_cut"]] - self._params["M_cut"])/ self._params["M_1"])**self._params["alpha"]
		return nsat

	def find_Mmin(self, lower_mass = 10.0, upper_mass = 14.0, epsilon = 10**-4):
		# binary search
		while upper_mass - lower_mass > epsilon:
			self._params["M_min"] = 10**((lower_mass + upper_mass) / 2)
			tot_num_gal = sum((self.N_cen(self.mass) + self.N_sat(self.mass)) * self._params["halo_histo"])
			density = tot_num_gal / self._params["boxsize"]**3
			if density > self._params["galaxy_density"]:
				lower_mass = np.log10(self._params["M_min"])
			else:
				upper_mass = np.log10(self._params["M_min"])
		print("[HOD] find M_min = {:.4f}".format(self._params["M_min"]))
		return

	def update_parameters(self, quantities, values):
		for q, v in zip(quantities, values):
			if q not in ["f_max", "M_min", "sigma_logM", "f_b", "M_b", "sigma_b", "M_1", "M_cut" , "alpha"]:
				raise ValueError("Invalid quantity name: {}".format(q))
			self._params[q] = v
		if self._params["galaxy_density"] is not None:
			self.find_Mmin()
		return





