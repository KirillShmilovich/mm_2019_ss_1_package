import numpy as np

class Geom:
    def __init__(self, method, num_particles, box_length, file_name = None):
        self.method = method
        self.file_name = file_name
        self.num_particles = num_particles
        self.box_length = box_length
        self.volume = np.power(self.box_length,3)
        self.generate_initial_state()

    def generate_initial_state(self):
        """Generate initial coordinates of particles in a box either randomly or based on a file.

        Parameters
        ----------
        method : string, either 'random' or 'file'
            Method of generating initial state.
        file_name : string
            Name of file used to generate initial state.
        num_particles : integer
            Number of particles to generate.
        box_length : integer or float
            Length of box to generate.

        Returns
        -------
        coordinates : array
            Array of particle coordinates generated for an initial state
        """

        if self.method is 'random':
            self.coordinates = (0.5 - np.random.rand(self.num_particles, 3)) * self.box_length
        elif self.method is 'file':
            self.coordinates = np.loadtxt(self.file_name, skiprows=2, usecols=(1,2,3))
        else:
            raise TypeError('Method type not recognized.')

    def minimum_image_distance(self,r_i, r_j):
        """Calculate minimum image distance between two particles, i and j.

        Parameters
        ---------
        box_length : integer or float
            Length of box to generate.
        r_i : array
            Coordinates of particle i.
        r_j : array
            Coordinates of particle j.

        Returns
        -------
        rij2 : square of the distance between particles i and j.

        """
        rij = r_i - r_j
        rij = rij - self.box_length * np.round(rij / self.box_length)
        rij2 = np.dot(rij, rij)
        return rij2
