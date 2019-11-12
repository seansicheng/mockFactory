import time 
import pandas as pd
import numpy as np
from parameters import *
from utils import timeWrapper, vectorize, haloConcentration


def NFWDensity(r, rs, ps):
	return ps * rs / (r*(1+r/rs)*(1+r/rs))


@vectorize
def NFWPosition(m):
	cvir = haloConcentration(m)
	rvir = pow(3*m / (4*DELTA_HALO*np.pi*RHO_CRIT*OMEGA_M), 1.0/3.0)
	rs = rvir/cvir
	max_p = NFWDensity(rs,rs,1.0)*rs*rs*4.0*np.pi

	while True:
		r = np.random.rand() * rvir
		pr = NFWDensity(r,rs,1.0)*r*r*4.0*np.pi / max_p

		if np.random.rand() <= pr:
			costheta = 2.0*np.random.rand() - 1
			sintheta = np.sqrt(1-costheta**2)
			phi = 2*np.pi*np.random.rand()

			return [r*sintheta*np.cos(phi), r*sintheta*np.sin(phi), r*costheta]



def NFWVelocity(m):
	fac = np.sqrt(4.499e-48) * pow(4.0*DELTA_HALO*np.pi*OMEGA_M*RHO_CRIT/3, 1.0/6.0) * 3.09e19
	sigv = fac * pow(m,1.0/3.0) / np.sqrt(2.0)
	return np.random.randn(3)*sigv




class MockFactory(object):
	"""
		docstring for MockFactory
	"""

	def __init__(self, halofile, boxsize, VBIAS_C=1, VBIAS=1):

		"""
			input:
			---------
				halofile has format of "M200b", "x", "y", "z", "vx", "vy", "vz"
				boxsize is the size of halo catalog

		"""

		self.halofile = halofile
		self.boxsize = boxsize
		self.VBIAS_C = VBIAS_C
		self.VBIAS = VBIAS

		now = time.time()
		print("[{}] Reading halo file {} ...".format(self.__class__.__name__, halofile))
		self.halos = pd.read_table(self.halofile, header=None, delimiter=' ').values
		print("[{}] Done reading halo file, time cost {:.2f}s ...".format(self.__class__.__name__, time.time() - now))


		print("[{}] Adjusting position ...".format(self.__class__.__name__))
		self.halos[:,1] = np.array(list(map(self.position_adjust, self.halos[:,1])))
		self.halos[:,2] = np.array(list(map(self.position_adjust, self.halos[:,2])))
		self.halos[:,3] = np.array(list(map(self.position_adjust, self.halos[:,3])))
		print("[{}] Done adjusting position ...".format(self.__class__.__name__))
		
		#self.halos.columns = ["M200b", "x", "y", "z", "vx", "vy", "vz"]
		
		

		self.halolength = len(self.halos)
		self.index = np.arange(self.halolength)
		print("[{}] Read {} halos ...".format(self.__class__.__name__, self.halolength))



	def NFWCenVelocity(self, m):
		return NFWVelocity(m) * self.VBIAS_C

	def NFWSatVelocity(self, m):
		return NFWVelocity(m) * self.VBIAS

	def position_adjust(self, x):
		if x > self.boxsize:
			return x - self.boxsize
		elif x < 0:
			return x + self.boxsize
		else:
			return x



	@timeWrapper
	def populateCentral(self, hod):
		Ncen = np.array([hod.N_cen(m) for m in self.halos[:,0]])
		rand = np.random.rand(self.halolength)
		cen_mock = np.copy(self.halos[Ncen > rand])
		print("number of central is {}".format(len(cen_mock)))
		vg = np.array([self.NFWCenVelocity(m) for m in cen_mock[:,0]])
		cen_mock[:,4] += vg[:,0]
		cen_mock[:,5] += vg[:,1]
		cen_mock[:,6] += vg[:,2]

		return cen_mock


	@timeWrapper
	def populateSatellite(self, hod):
		Nsat = np.array([np.random.poisson(hod.N_sat(m)) for m in self.halos[:,0]])
		print("assign position and velocity...")
		xg, yg, zg, vxg, vyg, vzg, mass = [], [], [], [], [], [], []
		print("number of satellite is {}".format(sum(Nsat)))
		for i in self.index[Nsat != 0]:
			for j in range(Nsat[i]):
				r = NFWPosition(self.halos[i, 0])
				vg = self.NFWSatVelocity(self.halos[i, 0])
				xg.append(self.position_adjust(self.halos[i, 1]+r[0]))
				yg.append(self.position_adjust(self.halos[i, 2]+r[1]))
				zg.append(self.position_adjust(self.halos[i, 3]+r[2]))

				vxg.append(self.halos[i, 4]+vg[0])
				vyg.append(self.halos[i, 5]+vg[1])
				vzg.append(self.halos[i, 6]+vg[2])

				mass.append(self.halos[i, 0])

		sat_mock = np.array([mass, xg, yg, zg, vxg, vyg, vzg]).T

		return sat_mock




	@timeWrapper
	def populateSimulationHOD(self, hod, mock_flnm = None):
		"""
			input:
			----------
				hod is an instance of HOD class
				mock_flnm is the output mock filename
		"""
		mock = np.concatenate([self.populateCentral(hod), self.populateSatellite(hod)])
		if mock_flnm:
			print("Writing to csv ...")
			np.savetxt(mock_flnm, mock, fmt="%.5f")

		return mock








		
