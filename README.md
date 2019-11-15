# mockFactory
Python implementation of mockFactory, right now populateSimulationHOD is implemented. Use it to make mocks and compute clustering.

## Requirements
Python 3.6

numpy

pandas

colossus, for calculating halo concentration

if you want to use clustering module, Corrfunc is required

if you want to use MCMC module, emcee is required

## Usage
### Initialization
Feel free to change any cosmological parameters in `parameters.py`

import mockFactory
```python
from mockFactory import MockFactory
```
You can define input halofile and hod parameters. In this example, "halo_test.dat" in data directory is 1% of MDPL2 simulation halo file at redshift z=0.84. If you don't pass hod params into arguments, default hod will be used.

Halo file should have the first 7 columns as in format: "M200b", "x", "y", "z", "vx", "vy", "vz".
```python
halofile = "../data/halo_test.dat"

hod_params = {'f_max': 0.0919869695747, 'M_min': 6.8*10**12, 'M_cut': 1.45690882049e+12, 
               'sigma_logM':0.640647507034, 'alpha':0.899309153294, 
               'M_1': 1.99576840607e+14}

mockfactory = MockFactory(halofile, boxsize=1000, cvir_fac=1, vbias=0.8, hod_parameters=hod_params)
```
output should be:
```
[MockFactory] Reading halo file ../data/halo_test.dat ...
[MockFactory] Done reading halo file, time cost 1.61s ...
MAX HALOMASS 14.652430079305446
MAX HALOMASS 9.478681907362235
[MockFactory] Adjusting position ...
[MockFactory] Done adjusting position ...
[MockFactory] Computing halo concentration ...
[MockFactory] Read 1213430 halos ...
[MockFactory] memory used 92.5774 Mb ...
```
### Populate mock
If you don't pass mock_flnm into the function, it will not write to file.
```python
mockfile = "../data/mock_test.dat"
mock = mockfactory.populateSimulation(mock_flnm = mockfile, verbose=True)
```
```
[MockFactory] Begin populateSimulation ...
[MockFactory] Begin _populateCentral ...
number of central is 5105
[MockFactory] Done _populateCentral, time cost 0.10s ...
[MockFactory] Begin _populateSatellite ...
assign position and velocity...
number of satellite is 718
[MockFactory] Done _populateSatellite, time cost 0.14s ...
Writing to file ../data/mock_test.dat ...
[MockFactory] Done populateSimulation, time cost 0.27s ...
```
The mock has shape of (n_gal, 8), with columns as M_halo, x, y, z, vx, vy, vz, isCentral.

isCentral == 1 means the galaxy is a central galaxy, 0 means satellite galaxy.
```python
mock.shape
```
```
(5823, 8)
```

## Clustering
You can compute clustering as follows:
```python
from clustering import Clustering
import numpy as np

rbins = np.logspace(np.log10(0.1), np.log10(70), 21)
cluster = Clustering(rbins)
```
The input mock should have first 6 columns as x, y, z, vx, vy, vz.
```python
xi0, xi2, wp = cluster.xi_wp_cubic_mock(mock[:,1:], size=1000, xi_name="../data/xi.dat", wp_name="../data/wp.dat", verbose=True)
```
