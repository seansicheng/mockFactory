from mockFactory import MockFactory
from hod import HOD
import numpy as np

halofile = "../data/halo_test.dat"
mockfile = "../data/mock_test.dat"

hod_params = {"M_min": 0, "galaxy_density": 0.00057, "boxsize": 1000, "log_halo_mass_bins": np.arange(10,15,0.1), \
				"halo_histo": np.loadtxt("../../HAM/data/HOD/halo_central_histo.dat")}


mockfactory = MockFactory(halofile, boxsize=1000, cvir_fac=0.1, hod_parameters=hod_params)

mockfactory.update_params({"vbias_c": 1, "cvir_fac": 1, "sigma_logM":1.0})

mockfactory.populateSimulation(verbose=False)

