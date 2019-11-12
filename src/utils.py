import time 
import numpy as np
from colossus.cosmology import cosmology
cosmology.setCosmology('planck15')
from colossus.halo import concentration
from parameters import *
from scipy.interpolate import interp1d

def timeWrapper(func):
	"""
		wrapper function to compute time cost, used for class function
	"""

	def newfunc(*arg, **kwarg):
		now = time.time()
		cls = arg[0]
		print("[{}] Begin {} ...".format(cls.__class__.__name__, func.__name__))
		res = func(*arg, **kwarg)
		print("[{}] Done {}, time cost {:.2f}s ...".format(cls.__class__.__name__, func.__name__, time.time() - now))
		return res
	return newfunc

class vectorize(np.vectorize):
	"""
		NOTE: only used as decorator!!
	"""
	def __get__(self, obj, objtype):
		return functools.partial(self.__call__, obj)



class HaloConcentration(object):
	def __init__(self, z):
		self.f = interp1d(np.linspace(9,15,200), \
             concentration.concentration(10**np.linspace(9, 15, 200), "vir", z, model="bullock01"))


	def haloConcentration(self, m):
		return self.f(np.log10(m))


