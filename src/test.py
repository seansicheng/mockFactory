from mockFactory import MockFactory
from hod import HOD

halofile = "../data/halo_test.dat"
mockfile = "../data/mock_test.dat"

mockfactory = MockFactory(halofile, 1000)

hod = HOD()

#cen_mock = mockfactory.populateCentral(hod)
mockfactory.populateSimulationHOD(hod, mockfile)