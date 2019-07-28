import numpy as np
#from .geom import Geom
#from .energy import Energy
from geom import Geom
from energy import Energy
class MC:

    def __init__(self, method, reduced_temp, max_displacement, cutoff, num_particles = None, file_name = None, tune_displacement = True, reduced_den = None):
        self.beta = 1./float(reduced_temp)
        self._n_trials = 0
        self._n_accept = 0
        self.max_displacement = max_displacement
        self.tune_displacement = tune_displacement
        self._energy_array = None

        if method == 'random':
            self._Geom = Geom(method, num_particles = num_particles, reduced_den = reduced_den)
        elif method == 'file':
            self._Geom = Geom(method, file_name = file_name)
        else:
            raise ValueError("Method must be either 'file' or 'random'")

        self._Energy = Energy(self._Geom, cutoff)
        
        self.tail_correction = self._Energy.calculate_tail_correction()
        self.total_pair_energy = self._Energy.calculate_total_pair_energy()

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
        acc_rate = float(self._n_accept) / float(self._n_trials)
        if (acc_rate < 0.38):
            self.max_displacement *= 0.8
        elif (acc_rate > 0.42):
            self.max_displacement *= 1.2
        self._n_trials = 0
        self._n_accept = 0

    def get_energy(self):
        if (self._energy_array is None):
            raise ValueError("Simulation has not started running!")
        return self._energy_array

    def get_snapshot(self):
        return self._Geom

    def save_snapshot(self,file_name):
        self._Geom.save_state(file_name)

    def run(self, n_steps, freq, save_dir='.'):
        import os.path
        if (not os.path.exists(save_dir)):
            raise ValueError("Snapshot saving directory does not exist!")
        self._energy_array = np.zeros(n_steps)

        for i_step in range(n_steps):
            self._n_trials += 1
            i_particle = np.random.randint(self._Geom.num_particles)
            random_displacement = (2.0 * np.random.rand(3) - 1.0) * self.max_displacement
            current_energy = self._Energy.get_particle_energy(i_particle, self._Geom.coordinates)
            old_coordinate = self._Geom.coordinates[i_particle,:].copy()
            proposed_coordinate = self._Geom.wrap(old_coordinate + random_displacement)
            self._Geom.coordinates[i_particle,:] = proposed_coordinate
            proposed_energy = self._Energy.get_particle_energy(i_particle, self._Geom.coordinates)
            delta_e = proposed_energy - current_energy
            accept = self._accept_or_reject(delta_e)

            if accept:
                self.total_pair_energy += delta_e
                self._n_accept += 1
            else:
                self._Geom.coordinates[i_particle,:] = old_coordinate

            total_energy = (self.total_pair_energy + self.tail_correction) / self._Geom.num_particles
            self._energy_array[i_step] = total_energy

            if np.mod(i_step + 1, freq) == 0:
                print(i_step + 1, self._energy_array[i_step])
                self.save_snapshot('%s/snap_%d.txt'%(save_dir,i_step+1))
                if self.tune_displacement:
                    self._adjust_displacement()

if __name__ == "__main__":
    sim = MC(method = 'random', num_particles = 100, reduced_den = 0.9, reduced_temp = 0.9, max_displacement = 0.1, cutoff = 3.0)
    sim.run(n_steps = 50000, freq = 1000)