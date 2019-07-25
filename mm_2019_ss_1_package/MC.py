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

class Energy:

    def __init__(self, Geom, cutoff, num_particles):
       self.Geom = Geom 
       self.cutoff  = cutoff
       self.cutoff2 = self.cutoff**2
       self.num_particles = num_particles

    def lennard_jones_potential(self,rij2):
        sig_by_r6 = np.power(1 / rij2, 3)
        sig_by_r12 = np.power(sig_by_r6, 2)
        return 4.0 * (sig_by_r12 - sig_by_r6)

    def calculate_total_pair_energy(self):
        e_total = 0.0
        particle_count = len(self.Geom.coordinates)
        for i_particle in range(particle_count):
            for j_particle in range(i_particle):
                r_i = self.Geom.coordinates[i_particle]
                r_j = self.Geom.coordinates[j_particle]
                rij2 = self.Geom.minimum_image_distance(r_i, r_j)
                if rij2 < self.cutoff2:
                    e_pair = self.lennard_jones_potential(rij2)
                    e_total += e_pair
        return e_total

    def get_particle_energy(self, i_particle, coordinates):
        e_total = 0.0
        i_position = coordinates[i_particle]
        particle_count = len(coordinates)
        for j_particle in range(particle_count):
            if i_particle != j_particle:
                j_position = coordinates[j_particle]
                rij2 = self.Geom.minimum_image_distance(i_position, j_position)
                if rij2 < self.cutoff2:
                    e_pair = self.lennard_jones_potential(rij2)
                    e_total += e_pair
        return e_total
    
    def calculate_tail_correction(self):
        sig_by_cutoff3 = np.power(1.0 / self.cutoff, 3)
        sig_by_cutoff9 = np.power(sig_by_cutoff3, 3)
        e_correction = sig_by_cutoff9 - 3.0 * sig_by_cutoff3
        e_correction *= 8.0 / 9.0 * np.pi * self.num_particles / self.Geom.volume * self.num_particles
        return e_correction
    
class MC:

    def __init__(self, method, num_particles, reduced_den, reduced_temp, max_displacement, cutoff, file_name = None, tune_displacement = True):
        self.beta = 1./float(reduced_temp)
        self.n_trials = 0
        self.n_accept = 0
        self.max_displacement = max_displacement
        self.num_particles = num_particles
        self.box_length = np.cbrt(num_particles / reduced_den)
        self.cutoff = cutoff
        self.tune_displacement = tune_displacement

        self.Geom = Geom(method, self.num_particles, self.box_length, file_name=file_name)
        self.Energy = Energy(self.Geom, self.cutoff, self.num_particles)
        self.tail_correction = self.Energy.calculate_tail_correction()
        self.total_pair_energy = self.Energy.calculate_total_pair_energy()

    def accept_or_reject(self,delta_e):
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

    def adjust_displacement(self):
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
            accept = self.accept_or_reject(delta_e)

            if accept:
                self.total_pair_energy += delta_e
                self.n_accept += 1
                self.Geom.coordinates[i_particle] += random_displacement

            total_energy = (self.total_pair_energy + self.tail_correction) / self.num_particles
            self.energy_array[i_step] = total_energy

            if np.mod(i_step + 1, self.freq) == 0:
                print(i_step + 1, self.energy_array[i_step])
                if self.tune_displacement:
                    self.adjust_displacement()

sim = MC(method = 'random', num_particles = 100, reduced_den = 0.9, reduced_temp = 0.9, max_displacement = 0.1, cutoff = 3.0)
sim.run(n_steps = 50000, freq = 1000)