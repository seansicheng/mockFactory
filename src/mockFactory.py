import time 
import pandas as pd
import numpy as np
from parameters import *
from utils import timeWrapper, HaloConcentration


class MockFactory(object):
	"""
		docstring for MockFactory
	"""

	def __init__(self, halofile, boxsize, vbias_c=0, vbias=0, cvir_fac=1):

		"""
			input:
			---------
				halofile has format of "M200b", "x", "y", "z", "vx", "vy", "vz"
				boxsize is the size of halo catalog

		"""

		self.halofile = halofile
		self.boxsize = boxsize
		self.vbias_c = vbias_c
		self.vbias = vbias
		self.cvir_fac = cvir_fac

		now = time.time()
		print("[{}] Reading halo file {} ...".format(self.__class__.__name__, halofile))
		self.halos = pd.read_table(self.halofile, header=None, delimiter=' ').values
		print("[{}] Done reading halo file, time cost {:.2f}s ...".format(self.__class__.__name__, time.time() - now))
		print("MAX HALOMASS {}".format(np.log10(max(self.halos[:,0]))))
		print("MAX HALOMASS {}".format(np.log10(min(self.halos[:,0]))))

		print("[{}] Adjusting position ...".format(self.__class__.__name__))
		self.position_adjust(self.halos[:,1])
		self.position_adjust(self.halos[:,2])
		self.position_adjust(self.halos[:,3])
		print("[{}] Done adjusting position ...".format(self.__class__.__name__))
		
		#self.halos.columns = ["M200b", "x", "y", "z", "vx", "vy", "vz"]
		
		print("[{}] Computing halo concentration ...".format(self.__class__.__name__))
		hc = HaloConcentration(z=0.84)
		self.cvir = hc.haloConcentration(self.halos[:,0]) * self.cvir_fac
		self.rvir = pow(3*self.halos[:,0] / (4*DELTA_HALO*np.pi*RHO_CRIT*OMEGA_M), 1.0/3.0)

		self.halolength = len(self.halos)
		self.index = np.arange(self.halolength)
		print("[{}] Read {} halos ...".format(self.__class__.__name__, self.halolength))



	@staticmethod
	def NFWVelocity(m):
		fac = np.sqrt(4.499e-48) * pow(4.0*DELTA_HALO*np.pi*OMEGA_M*RHO_CRIT/3, 1.0/6.0) * 3.09e19
		sigv = fac * pow(m,1.0/3.0) / np.sqrt(2.0)
		return np.random.randn(3)*sigv


	def NFWCenVelocity(self, m):
		return self.NFWVelocity(m) * self.vbias_c


	def NFWSatVelocity(self, m):
		return self.NFWVelocity(m) * self.vbias

	def position_adjust(self, x):
		x[x > self.boxsize] -= self.boxsize
		x[x < 0] += self.boxsize
		return

	@staticmethod
	def NFWDensity(r, rs, ps):
		return ps * rs / (r*(1+r/rs)*(1+r/rs))

	@classmethod
	def NFWPosition(cls, rvir, cvir):
		rs = rvir/cvir
		max_p = cls.NFWDensity(rs,rs,1.0)*rs*rs*4.0*np.pi
		it = 0
		while True:
			it += 1
			r = np.random.rand() * rvir
			pr = cls.NFWDensity(r,rs,1.0)*r*r*4.0*np.pi / max_p

			if np.random.rand() <= pr:
				#print(it)
				costheta = 2.0*np.random.rand() - 1
				sintheta = np.sqrt(1-costheta**2)
				phi = 2*np.pi*np.random.rand()

				return [r*sintheta*np.cos(phi), r*sintheta*np.sin(phi), r*costheta]


	@timeWrapper
	def populateCentral(self, hod):
		Ncen = hod.N_cen(self.halos[:,0])
		rand = np.random.rand(self.halolength)
		cen_mock = np.zeros((len(self.halos[Ncen > rand]), 8))
		cen_mock[:,:-1] = self.halos[Ncen > rand]
		print("number of central is {}".format(len(cen_mock)))
		vg = np.array([self.NFWCenVelocity(m) for m in cen_mock[:,0]])
		cen_mock[:,4] += vg[:,0]
		cen_mock[:,5] += vg[:,1]
		cen_mock[:,6] += vg[:,2]
		cen_mock[:,7] = 1

		return cen_mock


	@timeWrapper
	def populateSatellite(self, hod):
		Nsat = np.zeros(self.halolength).astype(int)
		nsat = hod.N_sat(self.halos[:,0])
		Nsat[nsat != 0] = np.random.poisson(nsat[nsat!=0])
		#Nsat = Nsat.astype(int)
		print("assign position and velocity...")
		xg, yg, zg, vxg, vyg, vzg, mass = [], [], [], [], [], [], []
		print("number of satellite is {}".format(sum(Nsat)))
		t = time.time()
		for i in self.index[Nsat != 0]:
			for j in range(Nsat[i]):
				r = self.NFWPosition(self.rvir[i], self.cvir[i])
				vg = self.NFWSatVelocity(self.halos[i, 0])
				xg.append(self.halos[i, 1]+r[0])
				yg.append(self.halos[i, 2]+r[1])
				zg.append(self.halos[i, 3]+r[2])

				vxg.append(self.halos[i, 4]+vg[0])
				vyg.append(self.halos[i, 5]+vg[1])
				vzg.append(self.halos[i, 6]+vg[2])

				mass.append(self.halos[i, 0])
		print("for loop takes {:.2f} s".format(time.time() - t))
		sat_mock = np.array([mass, xg, yg, zg, vxg, vyg, vzg, [0]*len(mass)]).T
		self.position_adjust(sat_mock[:,1])
		self.position_adjust(sat_mock[:,2])
		self.position_adjust(sat_mock[:,3])

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








		
