import numpy as np

class CompositeScoringEngine:
    """
    Computes the 5-metric composite score for each algorithm.
    Metrics: Best Fitness, Population Diversity, Performance Trend, Convergence Speed, Computational Efficiency.
    """
    def __init__(self):
        # Weights for the 5 metrics as defined in the system architecture
        self.weights = {
            'fitness': 0.30,
            'speed': 0.25,
            'efficiency': 0.15,
            'diversity': 0.15,
            'trend': 0.15
        }
        
    def calculate_scores(self, algo_stats):
        """
        algo_stats is a dictionary containing the raw metrics for each active algorithm.
        Example: {
            'GA': {'fitness': 85.0, 'speed': +0.5, 'efficiency': 0.9, 'diversity': 0.6, 'trend': 0.4},
            'PSO': ...
        }
        Returns a dictionary of final composite scores between 0 and 1.
        """
        if not algo_stats:
            return {}
            
        metrics = ['fitness', 'speed', 'efficiency', 'diversity', 'trend']
        min_vals = {m: float('inf') for m in metrics}
        max_vals = {m: -float('inf') for m in metrics}
        
        # Find min and max for normalization
        for stats in algo_stats.values():
            for m in metrics:
                val = stats[m]
                if val < min_vals[m]: min_vals[m] = val
                if val > max_vals[m]: max_vals[m] = val
                
        # Normalize and compute weighted sum
        final_scores = {}
        for algo_name, stats in algo_stats.items():
            score = 0.0
            for m in metrics:
                # Avoid division by zero if all algorithms perform identically
                denom = max_vals[m] - min_vals[m]
                norm_val = (stats[m] - min_vals[m]) / denom if denom > 1e-9 else 0.5
                score += self.weights[m] * norm_val
            final_scores[algo_name] = score
            
        return final_scores
