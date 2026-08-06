import numpy as np
from .base import BaseOptimizer

class ParticleSwarm(BaseOptimizer):
    def __init__(self, search_space, pop_size):
        super().__init__(search_space, pop_size)
        self.w = 0.5  # Inertia
        self.c1 = 1.5 # Cognitive (Personal)
        self.c2 = 1.5 # Social (Global)
        
        self.particles = [] 
        self.gbest_pos = None
        self.gbest_score = -float('inf')
        
    def ask(self, num_samples):
        # Initialize particles if empty
        if len(self.particles) == 0:
            for _ in range(self.pop_size):
                pos = np.random.rand(self.dimensions)
                vel = np.random.uniform(-0.1, 0.1, self.dimensions)
                self.particles.append({
                    'pos': pos, 
                    'vel': vel, 
                    'pbest_pos': np.copy(pos), 
                    'pbest_score': -float('inf')
                })
            # Return requested number of samples
            return [p['pos'] for p in self.particles[:num_samples]]
            
        new_vectors = []
        # Update logic
        for i in range(num_samples):
            p = self.particles[i % len(self.particles)]
            
            r1 = np.random.rand(self.dimensions)
            r2 = np.random.rand(self.dimensions)
            
            p['vel'] = (self.w * p['vel'] + 
                        self.c1 * r1 * (p['pbest_pos'] - p['pos']) + 
                        self.c2 * r2 * (self.gbest_pos - p['pos']))
            
            p['pos'] = p['pos'] + p['vel']
            p['pos'] = np.clip(p['pos'], 0.0, 1.0)
            new_vectors.append(np.copy(p['pos']))
            
        return new_vectors

    def tell(self, evaluated_individuals):
        # We assume the evaluated_individuals correspond to the particles we just generated
        for i, ind in enumerate(evaluated_individuals):
            if i >= len(self.particles): break
            p = self.particles[i]
            
            # Update personal best
            if ind.fitness > p['pbest_score']:
                p['pbest_score'] = ind.fitness
                p['pbest_pos'] = np.copy(p['pos'])
                
            # Update global best
            if ind.fitness > self.gbest_score:
                self.gbest_score = ind.fitness
                self.gbest_pos = np.copy(p['pos'])
