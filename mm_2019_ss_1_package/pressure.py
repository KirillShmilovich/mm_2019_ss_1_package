import numpy as np

class Pressure:

    def ideal_pressure(num_particles, red_temperature, ):
        kb =  1.380649e-23
        ideal_pressure = reduced_den * reduced_temp * kb

    def force(self, rij2):
        sig_by_r6 = np.power(1 / rij2, 3)
        sig_by_r12 = np.power(sig_by_r6, 2)
        return 48 * (sig_by_r12 - (1 / 2 *sig_by_r6))



    def total_pressure(self)
        v_total = 0 
        particle_count = len(self.Geom.coordinates)
        #virial addition
        for i_particle in range(particle_count):
            for j_particle in range(i_particle):
                r_i = self.Geom.coordinates[i_particle]
                r_j = self.Geom.coordinates[j_particle]
                rij2 = self.Geom.minimum_image_distance(r_i, r_j)
                if rij2 < self.cutoff2:
                    v_pair = self.force(rij2) * rij
                    v_total += v_pair
        
        total_pressure = ideal_pressure + (1 / 3 * v_total)
        return total_pressure

    
    def pressure_tail_correction(self):
        sig_by_cutoff3 = np.power(1.0 / self.cutoff, 3)
        sig_by_cutoff9 = np.power(sig_by_cutoff3, 3)
        e_correction = 2 / 3 * sig_by_cutoff9 - * sig_by_cutoff3
        e_correction *= 16.0 / 3.0 * np.pi * self.num_particles / self.Geom.volume * self.num_particles
        return p_correction

