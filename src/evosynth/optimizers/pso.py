import numpy as np
from .base import BaseOptimizer
from evosynth.config import get_settings


class ParticleSwarm(BaseOptimizer):
    def __init__(self, search_space, pop_size, rng=None):
        super().__init__(search_space, pop_size, rng)
        p = get_settings().optimizers.pso  # [optimizers.pso] in settings/optimizers.toml
        self.w = p.w    # Inertia weight
        self.c1 = p.c1  # Cognitive coefficient (Personal best)
        self.c2 = p.c2  # Social coefficient (Global best)
        self.init_velocity = p.init_velocity

        self.particles = []
        self.gbest_pos = None
        self.gbest_score = -float('inf')
        self._cursor = 0        # next particle to move (round-robin)
        self._pending = []      # particle index of each vector returned by the last ask

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
            vel = self.rng.uniform(-self.init_velocity, self.init_velocity, self.dimensions)
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
            self.particles.append(self._new_random_particle())

    def _new_random_particle(self):
        pos = self.rng.random(self.dimensions)
        vel = self.rng.uniform(-self.init_velocity, self.init_velocity, self.dimensions)
        return {'pos': pos, 'vel': vel, 'pbest_pos': np.copy(pos), 'pbest_score': -float('inf')}

    def ask(self, num_samples):
        """
        Phase 2.1: never re-proposes the seed population (it is already evaluated): the first ask
        after a seeding tell already moves the particles. Particles are visited round-robin
        (self._cursor), so asking for fewer samples than the swarm size still moves every particle
        in turn; tell() matches results to the particles recorded in self._pending.
        """
        self._pending = []
        if len(self.particles) == 0:
            if len(self.population) > 0:
                self._seed_particles_from_population()
            else:
                # Pure random initialization (no seed population): the random positions are new,
                # so they are proposed as they are.
                self.particles = [self._new_random_particle() for _ in range(self.pop_size)]
                out = []
                for _ in range(num_samples):
                    idx = self._cursor % len(self.particles)
                    self._cursor += 1
                    self._pending.append(idx)
                    out.append(np.copy(self.particles[idx]['pos']))
                return out

        # Standard PSO velocity + position update
        new_vectors = []
        for _ in range(num_samples):
            idx = self._cursor % len(self.particles)
            self._cursor += 1
            p = self.particles[idx]
            r1 = self.rng.random(self.dimensions)
            r2 = self.rng.random(self.dimensions)
            gbest = self.gbest_pos if self.gbest_pos is not None else p['pbest_pos']

            p['vel'] = (
                self.w * p['vel']
                + self.c1 * r1 * (p['pbest_pos'] - p['pos'])
                + self.c2 * r2 * (gbest - p['pos'])
            )
            p['pos'] = np.clip(p['pos'] + p['vel'], 0.0, 1.0)
            self._pending.append(idx)
            new_vectors.append(np.copy(p['pos']))

        return new_vectors

    def inject(self, individuals):
        """Reset the worst particles (lowest personal best) to the injected individuals."""
        individuals = list(individuals)
        if self.particles:
            worst = sorted(range(len(self.particles)), key=lambda i: self.particles[i]['pbest_score'])
            for idx, ind in zip(worst, individuals):
                pos = np.copy(ind.vector)
                self.particles[idx] = {
                    'pos': pos,
                    'vel': self.rng.uniform(-self.init_velocity, self.init_velocity, self.dimensions),
                    'pbest_pos': np.copy(pos), 'pbest_score': ind.fitness,
                }
                if ind.fitness > self.gbest_score:
                    self.gbest_score, self.gbest_pos = ind.fitness, np.copy(pos)
        super().inject(individuals)

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

        # Update the particles that proposed these vectors (recorded by ask, in order)
        for idx, ind in zip(self._pending, evaluated_individuals):
            p = self.particles[idx]
            # Update particle's current position to match what was actually evaluated
            p['pos'] = np.copy(ind.vector)

            if ind.fitness > p['pbest_score']:
                p['pbest_score'] = ind.fitness
                p['pbest_pos'] = np.copy(p['pos'])

            if ind.fitness > self.gbest_score:
                self.gbest_score = ind.fitness
                self.gbest_pos = np.copy(p['pos'])

        self._pending = []

        # Keep self.population in sync (sorted best-first) so controller
        # can read the latest evaluated individuals easily
        self.population.extend(evaluated_individuals)
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)
        self.population = self.population[:self.pop_size]
