import numpy as np
from .geom import Geom
from .energy import Energy

class MC:

    def __init__(self, method, reduced_den, reduced_temp, max_displacement, cutoff, num_particles = None, file_name = None, tune_displacement = True):
        self.beta = 1./float(reduced_temp)
        self.n_trials = 0
        self.n_accept = 0
        self.max_displacement = max_displacement
        self.cutoff = cutoff
        self.tune_displacement = tune_displacement

        if method == 'random':
            self.num_particles = num_particles
            self.box_length = np.cbrt(num_particles / reduced_den)
            self.Geom = Geom(method, num_particles = self.num_particles, box_length = self.box_length)
        elif method == 'file':
            self.Geom = Geom(method, file_name = file_name)
            self.num_particles = self.Geom.num_particles
            self.box_length = self.Geom.box_length
        else:
            raise ValueError("Method must be either 'file' or 'random'")

        self.Energy = Energy(self.Geom, self.cutoff, self.num_particles)
        
        self.tail_correction = self.Energy.calculate_tail_correction()
        self.total_pair_energy = self.Energy.calculate_total_pair_energy()

    def _accept_or_reject(self,delta_e):
        if delta_e < 0.0:
            accept = True
        else:
            random_number = np.random.rand(1)
            p_acc = np.exp(-self.beta * delta_e)
            if random_number < p_acc:
                accept = True
            else:
                accept = False
        return accept

    def _adjust_displacement(self):
        acc_rate = float(self.n_accept) / float(self.n_trials)
        if (acc_rate < 0.38):
            self.max_displacement *= 0.8
        elif (acc_rate > 0.42):
            self.max_displacement *= 1.2
        self.n_trials = 0
        self.n_accept = 0
    
    def run(self, n_steps, freq):
        self.n_steps = n_steps
        self.freq = freq
        self.energy_array = np.zeros(n_steps)

        for i_step in range(self.n_steps):
            self.n_trials += 1
            i_particle = np.random.randint(self.Geom.num_particles)
            random_displacement = (2.0 * np.random.rand(3) - 1.0) * self.max_displacement

            current_energy = self.Energy.get_particle_energy(i_particle, self.Geom.coordinates)
            proposed_coordinates = self.Geom.coordinates.copy()
            proposed_coordinates[i_particle] += random_displacement
            proposed_coordinates -= self.box_length * np.round(proposed_coordinates / self.box_length)

            proposed_energy = self.Energy.get_particle_energy(i_particle, proposed_coordinates)
            delta_e = proposed_energy - current_energy
            accept = self._accept_or_reject(delta_e)

            if accept:
                self.total_pair_energy += delta_e
                self.n_accept += 1
                self.Geom.coordinates[i_particle] += random_displacement

            total_energy = (self.total_pair_energy + self.tail_correction) / self.num_particles
            self.energy_array[i_step] = total_energy

            if np.mod(i_step + 1, self.freq) == 0:
                print(i_step + 1, self.energy_array[i_step])
                if self.tune_displacement:
                    self._adjust_displacement()

if __name__ == "__main__":
    sim = MC(method = 'random', num_particles = 100, reduced_den = 0.9, reduced_temp = 0.9, max_displacement = 0.1, cutoff = 3.0)
    sim.run(n_steps = 50000, freq = 1000)