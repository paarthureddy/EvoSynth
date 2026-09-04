import numpy as np
from .base import BaseOptimizer


class ParticleSwarm(BaseOptimizer):
    def __init__(self, search_space, pop_size):
        super().__init__(search_space, pop_size)
        self.w = 0.5   # Inertia weight
        self.c1 = 1.5  # Cognitive coefficient (Personal best)
        self.c2 = 1.5  # Social coefficient (Global best)

        self.particles = []
        self.gbest_pos = None
        self.gbest_score = -float('inf')

    def _seed_particles_from_population(self):
        """
        Initializes the particle swarm from self.population (seeded via tell).
        Each Individual in self.population becomes a particle with its vector
        as the starting position. This allows PSO to warm-start from existing
        good configurations instead of random initialization.
        """
        self.particles = []
        for ind in self.population:
            pos = np.copy(ind.vector)
            vel = np.random.uniform(-0.1, 0.1, self.dimensions)
            pbest_score = ind.fitness if ind.fitness is not None else - \
                float('inf')
            self.particles.append({
                'pos': pos,
                'vel': vel,
                'pbest_pos': np.copy(pos),
                'pbest_score': pbest_score
            })
            # Update global best from seed
            if pbest_score > self.gbest_score:
                self.gbest_score = pbest_score
                self.gbest_pos = np.copy(pos)

        # Pad to pop_size with random particles if needed
        while len(self.particles) < self.pop_size:
            pos = np.random.rand(self.dimensions)
            vel = np.random.uniform(-0.1, 0.1, self.dimensions)
            self.particles.append({
                'pos': pos, 'vel': vel,
                'pbest_pos': np.copy(pos), 'pbest_score': -float('inf')
            })

    def ask(self, num_samples):
        # If particles not yet built, seed them from the population (set via tell)
        if len(self.particles) == 0:
            if len(self.population) > 0:
                self._seed_particles_from_population()
            else:
                # Pure random initialization (no seed population)
                for _ in range(self.pop_size):
                    pos = np.random.rand(self.dimensions)
                    vel = np.random.uniform(-0.1, 0.1, self.dimensions)
                    self.particles.append({
                        'pos': pos, 'vel': vel,
                        'pbest_pos': np.copy(pos), 'pbest_score': -float('inf')
                    })
            return [np.copy(p['pos']) for p in self.particles[:num_samples]]

        # Standard PSO velocity + position update
        new_vectors = []
        for i in range(num_samples):
            p = self.particles[i % len(self.particles)]
            r1 = np.random.rand(self.dimensions)
            r2 = np.random.rand(self.dimensions)

            p['vel'] = (
                self.w * p['vel']
                + self.c1 * r1 * (p['pbest_pos'] - p['pos'])
                + self.c2 * r2 * (self.gbest_pos - p['pos'])
            )
            p['pos'] = np.clip(p['pos'] + p['vel'], 0.0, 1.0)
            new_vectors.append(np.copy(p['pos']))

        return new_vectors

    def tell(self, evaluated_individuals):
        """
        Receives evaluated individuals, updates pbest / gbest,
        and syncs self.population so the controller can track PSO's best.
        """
        # First call with seed population: store in self.population
        # (particles will be built on first ask())
        if len(self.particles) == 0:
            self.population = list(evaluated_individuals)
            return

        # Update particles from corresponding evaluations
        for i, ind in enumerate(evaluated_individuals):
            if i >= len(self.particles):
                break
            p = self.particles[i]
            # Update particle's current position to match what was actually evaluated
            p['pos'] = np.copy(ind.vector)

            if ind.fitness > p['pbest_score']:
                p['pbest_score'] = ind.fitness
                p['pbest_pos'] = np.copy(p['pos'])

            if ind.fitness > self.gbest_score:
                self.gbest_score = ind.fitness
                self.gbest_pos = np.copy(p['pos'])

        # Keep self.population in sync (sorted best-first) so controller
        # can read the latest evaluated individuals easily
        self.population.extend(evaluated_individuals)
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)
        self.population = self.population[:self.pop_size]
