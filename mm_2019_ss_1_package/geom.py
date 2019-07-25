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
        if self.method is 'random':
            self.coordinates = (0.5 - np.random.rand(self.num_particles, 3)) * self.box_length
        elif self.method is 'file':
            self.coordinates = np.loadtxt(self.file_name, skiprows=2, usecols=(1,2,3))
    
    def minimum_image_distance(self,r_i, r_j):
        rij = r_i - r_j
        rij = rij - self.box_length * np.round(rij / self.box_length)
        rij2 = np.dot(rij, rij)
        return rij2