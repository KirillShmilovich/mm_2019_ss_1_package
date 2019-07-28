"""
Unit and regression test for the mm_2019_ss_1_package package.
"""

# Import package, test suite, and other packages as needed
import mm_2019_ss_1_package
from mm_2019_ss_1_package import Energy, Geom 
import pytest
import sys
import numpy as np
import glob

@pytest.fixture()
def trial_sim():
    # Changed the fixture to so we can pass arguments
    def _get_trial_sim(method = 'random', num_particles = 1000, reduced_den = 1.0, reduced_temp = 1.0, max_displacement = 0.1, cutoff = 3.0):
        trial_sim = mm_2019_ss_1_package.MC(method = method, num_particles = num_particles, reduced_den = reduced_den, reduced_temp = reduced_temp, max_displacement = max_displacement, cutoff = cutoff)
        return trial_sim

    return _get_trial_sim

def test_mm_2019_ss_1_package_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "mm_2019_ss_1_package" in sys.modules


def test_minimum_image_distance(trial_sim):
    """Test the method to calculate minimum image distance. Arguments for MC.py are set to be easy ones so that expected value is 2"""

    G = Geom(method = 'random', num_particles = 1000, reduced_den = 1.0)
    trial_point1 = np.array([1,0,0])
    trial_point2 = np.array([9,0,0])
    
    calculated_value = G.minimum_image_distance(trial_point1 , trial_point2)
    expected_value = 2.0 ** 2

    assert np.isclose(calculated_value , expected_value)

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

    
