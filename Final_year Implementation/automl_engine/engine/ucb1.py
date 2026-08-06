import math

class UCB1Bandit:
    """
    Multi-Armed Bandit using UCB1 strategy to handle algorithm stagnation.
    Balances Exploitation (average past reward) with Exploration (trying underused algorithms).
    """
    def __init__(self, c=1.414):
        self.c = c # Exploration constant (typically sqrt(2))
        self.total_pulls = 0
        self.stats = {} # Maps 'algo_name' to {'pulls': int, 'total_reward': float}
        
    def register_algorithm(self, algo_name):
        if algo_name not in self.stats:
            self.stats[algo_name] = {'pulls': 0, 'total_reward': 0.0}
            
    def update(self, algo_name, reward):
        """Records the composite score (reward) achieved by an algorithm after deployment."""
        self.register_algorithm(algo_name)
        self.stats[algo_name]['pulls'] += 1
        self.stats[algo_name]['total_reward'] += reward
        self.total_pulls += 1
        
    def select_next(self, available_algorithms):
        """
        Selects the next algorithm to deploy based on the UCB1 formula:
        UCB = avg_reward + c * sqrt(ln(total_pulls) / algorithm_pulls)
        """
        best_score = -float('inf')
        best_algo = None
        
        for algo in available_algorithms:
            self.register_algorithm(algo)
            pulls = self.stats[algo]['pulls']
            
            # If an algorithm has never been used, it gets an infinite exploration bonus
            if pulls == 0:
                return algo
                
            avg_reward = self.stats[algo]['total_reward'] / pulls
            exploration_bonus = self.c * math.sqrt(math.log(self.total_pulls) / pulls)
            
            ucb_score = avg_reward + exploration_bonus
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_algo = algo
                
        return best_algo
