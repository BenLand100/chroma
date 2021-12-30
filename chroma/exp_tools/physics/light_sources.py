import tqdm
import numpy as np
from chroma.event import Photons


class LaserSource(Photons):

    def __init__(self, center, radius, normal, number, divergence, intensity_profile):
        self.center = center
        self.radius = radius
        self.normal = normal
        self.number = number
        self.div_angle = (divergence / 180) * np.pi
        self.profile = intensity_profile
        
    def generate_photons(self, rate=None):
        radial = self.__radial_vector()
        random_radius = np.random.uniform(0, self.radius**2, int(self.number))
        angles = np.random.uniform(0, 2*np.pi, int(self.number))
        pol_angles = np.random.uniform(0, 2*np.pi, int(self.number))
        div_angles = np.random.uniform(0, self.div_angle, int(self.number))
        
        positions = []
        directions = []
        polzs = []
        wavelengths = []
        
        for i, angle in tqdm.tqdm(enumerate(angles), total=len(angles)):
            rot_matrix = self.__rotation_matrix(self.normal, angle)
            vector = np.matmul(rot_matrix, radial)
            direction = vector * (np.sin(div_angles[i])/np.cos(div_angles[i])) + self.normal
            direction = direction / np.linalg.norm(direction)
            positions.append(vector*np.sqrt(random_radius[i]) + self.center)
            directions.append(direction)
            pol_vector = np.matmul(self.__rotation_matrix(self.normal, pol_angles[i]), radial)
            polzs.append(pol_vector)
        
        self.pos = np.array(positions)
        self.dir = np.array(directions)
        self.pol = np.array(polzs)
        if isinstance(self.profile, float) or isinstance(self.profile, int):
            self.wavelengths = np.array([self.profile]*int(self.number))
        else:
            self.wavelengths = self.profile.rvs(size=int(self.number))
            
        if rate:
            time_spacing = (1 / rate) * 1e9
            self.t = np.arange(0, self.number*time_spacing, time_spacing)
        else:
            self.t = None
            
        super().__init__(self.pos, self.dir, self.pol, self.wavelengths, self.t)
    
    def __radial_vector(self):
        zero_locs = np.where(self.normal == 0)[0]
        bottom_rows = [[0, 0, 0], [0, 0, 0]]
        indices = [0, 1, 2]
        if len(zero_locs) == 1:
            bottom_rows[0][zero_locs[0]] = 1
            indices.remove(zero_locs[0])
            bottom_rows[1][0] = 1
        elif len(zero_locs) == 2:
            bottom_rows[0][zero_locs[0]] = 1
            bottom_rows[1][zero_locs[1]] = 1
        else:
            bottom_rows[0][0] = 1
            bottom_rows[1][1] = 1
        matrix = np.array([self.normal, bottom_rows[0], bottom_rows[1]])
        rand_nums = np.random.uniform(-1, 1, 2)
        b = np.array([0, rand_nums[0], rand_nums[1]]) + np.matmul(matrix, self.center)
        point = np.linalg.lstsq(matrix, b)[0]
        radial = (point - self.center) / np.linalg.norm(point - self.center)
        return radial
    
    def __rotation_matrix(self, n, theta):
        first_row = [np.cos(theta) + n[0]**2*(1-np.cos(theta)), n[0]*n[1]*(1-np.cos(theta))-n[2]*np.sin(theta), n[0]*n[2]*(1-np.cos(theta)) + n[1]*np.sin(theta)]
        second_row = [n[0]*n[1]*(1-np.cos(theta)) + n[2]*np.sin(theta), np.cos(theta) + n[1]**2*(1-np.cos(theta)), n[1]*n[2]*(1-np.cos(theta)) - n[0]*np.sin(theta)]
        third_row = [n[0]*n[2]*(1-np.cos(theta)) - n[1]*np.sin(theta), n[1]*n[2]*(1-np.cos(theta)) + n[0]*np.sin(theta), np.cos(theta) + n[2]**2*(1-np.cos(theta))]
        return np.array([first_row, second_row, third_row])
