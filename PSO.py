import numpy as np
import copy 
#assume minimisation problem
class PSO:
    def __init__(self,input_size, restriction_start, restriction_end,population_size) -> None:
        self.input_size = input_size
        self.restriction_start = restriction_start
        self.restriction_end = restriction_end
        self.population_size = population_size
        self.best_score_per_particle = [np.inf for _ in range(self.population_size)]
        self.best_particles = None
        self.best_score_of_all_particles = np.inf
        self.best_particle_of_all = None
        self.intertia = 0.7
        self.c1 = 0.1
        self.c2 = 0.2
    def generate_start_solution(self):
        self.X = np.random.uniform(self.restriction_start,self.restriction_end,(self.population_size,self.input_size))
        self.best_particles = copy.deepcopy(self.X)
        self.V = np.random.uniform(self.restriction_start,self.restriction_end,(self.population_size,self.input_size)) * 0.001
        return self.X
    def move_one_step(self,scores):  
        #update best scores, based on scores from iteration
        for i in range(len(scores)):
            current_particle = self.X[i]
            if scores[i] < self.best_score_per_particle[i]:
                self.best_score_per_particle[i] = scores[i]
                self.best_particles[i] = current_particle
                if scores[i] < self.best_score_of_all_particles:                   
                    self.best_particle_of_all = current_particle
                    self.best_score_of_all_particles = scores[i]
        #update V
        for j in range(len(self.V)):
            self.V[j] = self.intertia*self.V[j] + self.c1*(self.best_particles[j]-self.X[j]) + self.c2*(self.best_particle_of_all-self.X[j])
        #update X
        new_X = self.X + self.V
        self.X = new_X
        return self.X






