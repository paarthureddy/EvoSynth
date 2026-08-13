import numpy as np
import math
from .base import BaseOptimizer

class CMAES(BaseOptimizer):
    def __init__(self, search_space, pop_size, sigma=0.2, elite_ratio=0.4):
        super().__init__(search_space, pop_size)
        self.sigma = sigma
        self.elite_ratio = elite_ratio
        
        # Mean initialized to random position in [0, 1]^D
        self.mean = np.random.rand(self.dimensions)
        # Covariance matrix initialized to Identity
        self.covariance = np.eye(self.dimensions) * 0.05
        
        # For tracking state
        self.population = []
        self._last_ask_vectors = []
        
    def ask(self, num_samples):
        # We need to sample from a multivariate normal distribution
        # using the current mean and covariance matrix
        
        # Ensure covariance is symmetric and positive semi-definite
        self.covariance = (self.covariance + self.covariance.T) / 2.0
        # Add small ridge for numerical stability
        self.covariance += np.eye(self.dimensions) * 1e-8
        
        new_vectors = []
        for _ in range(num_samples):
            # Sample standard normal
            z = np.random.randn(self.dimensions)
            
            # Cholesky decomposition C = L L^T
            try:
                L = np.linalg.cholesky(self.covariance)
            except np.linalg.LinAlgError:
                # Fallback to diagonal if decomposition fails
                L = np.diag(np.sqrt(np.abs(np.diag(self.covariance))))
                
            # y = L * z
            y = L @ z
            
            # x = mean + sigma * y
            x = self.mean + self.sigma * y
            
            # Ensure boundaries [0, 1]
            x = np.clip(x, 0.0, 1.0)
            new_vectors.append(x)
            
        self._last_ask_vectors = new_vectors
        return new_vectors

    def tell(self, evaluated_individuals):
        # If this is the initial population filling
        if len(self.population) < self.pop_size:
            self.population.extend(evaluated_individuals)
            if len(self.population) < self.pop_size:
                return
        else:
            # Replace old population with the newly evaluated one
            self.population = evaluated_individuals
            
        # 1. Sort population by fitness (descending)
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)
        
        # 2. Select top 'mu' elites
        mu = max(1, int(self.pop_size * self.elite_ratio))
        elites = self.population[:mu]
        
        # 3. Calculate weights (exponential decay)
        weights = np.array([math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)])
        weights /= np.sum(weights)
        
        # 4. Update Mean (recombination)
        old_mean = np.copy(self.mean)
        elite_vectors = np.array([ind.vector for ind in elites])
        self.mean = np.sum(elite_vectors * weights[:, np.newaxis], axis=0)
        
        # 5. Update Covariance Matrix
        # Rank-mu update
        c_mu = 0.3 # adaptation factor
        new_cov = np.zeros((self.dimensions, self.dimensions))
        
        for i in range(mu):
            # Difference from old mean scaled by sigma
            dy = (elite_vectors[i] - old_mean) / self.sigma
            # Outer product
            dy_dyT = np.outer(dy, dy)
            new_cov += weights[i] * dy_dyT
            
        self.covariance = (1 - c_mu) * self.covariance + c_mu * new_cov
        
        # 6. Step Size (Sigma) Adaptation
        # Heuristic: adjust based on distance of elites from old mean
        # (A full CMA-ES uses evolution paths, but this works well for our scale)
        avg_dist = 0
        for i in range(mu):
            dist = np.linalg.norm(elite_vectors[i] - old_mean)
            avg_dist += weights[i] * dist
            
        target_dist = self.sigma * math.sqrt(self.dimensions) * 0.5
        
        if avg_dist < target_dist:
            self.sigma = max(0.005, self.sigma * 0.92)
        else:
            self.sigma = min(0.5, self.sigma * 1.02)
