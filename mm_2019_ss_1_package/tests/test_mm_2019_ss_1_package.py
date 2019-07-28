"""
Unit and regression test for the mm_2019_ss_1_package package.
"""

# Import package, test suite, and other packages as needed
import mm_2019_ss_1_package
from mm_2019_ss_1_package import Energy,Geom 
import pytest
import sys
import glob
import numpy as np

def test_mm_2019_ss_1_package_imported():
	"""Sample test, will always pass so long as import statement worked"""
	assert "mm_2019_ss_1_package" in sys.modules

def test_energy():
	"""
	Check the total pair energy calculation matches reference LJ calculation in NIST
	"""
	samples = glob.glob('tests/lj_sample_configurations/*.txt')
	samples.sort()

	# Test r_cut = 3.0
	r_cut = 3.0
	reference = np.array([-4.3515E+03,-6.9000E+02,-1.1467E+03,-1.6790E+01])
	calculation = np.zeros(len(samples))
	for i in range(len(samples)):
		sample = samples[i]
		geom = Geom(method='file',file_name=sample)
		energy = Energy(geom,r_cut)
		E_total = energy.calculate_total_pair_energy()
		calculation[i] = E_total
	assert np.allclose(np.around(reference,decimals=1),np.around(calculation,decimals=1))

	# Test r_cut = 4.0
	r_cut = 4.0
	reference = np.array([-4.4675E+03,-7.0460E+02,-1.1754E+03,-1.7060E+01])
	calculation = np.zeros(len(samples))
	for i in range(len(samples)):
		sample = samples[i]
		geom = Geom(method='file',file_name=sample)
		energy = Energy(geom,r_cut)
		E_total = energy.calculate_total_pair_energy()
		calculation[i] = E_total
	assert np.allclose(np.around(reference,decimals=1),np.around(calculation,decimals=1))



