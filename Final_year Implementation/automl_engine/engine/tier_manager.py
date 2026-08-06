import math

class TierManager:
    """
    Manages the structural division of the population into Tier 1 and Tier 2,
    and handles Island-based elite sharing (cooperative co-evolution).
    """
    def __init__(self, total_population, t1_ratio=0.4):
        # Shuffle to ensure unbiased starting distribution despite fitness
        import random
        random.shuffle(total_population)
        
        self.t1_size = int(len(total_population) * t1_ratio)
        self.t2_size = len(total_population) - self.t1_size
        
        self.tier1_pool = total_population[:self.t1_size]
        self.tier2_pool = total_population[self.t1_size:]
        
    def allocate_t1_batch(self, batch_size):
        """Pops and returns a batch of individuals for a Tier 1 algorithm."""
        if len(self.tier1_pool) < batch_size:
            raise ValueError("Not enough individuals in Tier 1 pool.")
        batch = self.tier1_pool[:batch_size]
        self.tier1_pool = self.tier1_pool[batch_size:]
        return batch
        
    def allocate_t2_batch(self, batch_size):
        """Pops and returns a batch of individuals for a Tier 2 island."""
        if len(self.tier2_pool) < batch_size:
            raise ValueError("Not enough individuals in Tier 2 pool.")
        batch = self.tier2_pool[:batch_size]
        self.tier2_pool = self.tier2_pool[batch_size:]
        return batch

    @staticmethod
    def share_elites(island_a_population, island_b_population, top_pct=0.2):
        """
        Executes cooperative co-evolution by swapping the top 20% of individuals 
        between two Tier-2 islands.
        """
        # Sort both populations by fitness descending
        island_a = sorted(island_a_population, key=lambda x: x.fitness, reverse=True)
        island_b = sorted(island_b_population, key=lambda x: x.fitness, reverse=True)
        
        num_share = max(1, int(len(island_a) * top_pct))
        
        # Extract elites
        elites_a = island_a[:num_share]
        elites_b = island_b[:num_share]
        
        # Swap: Island A gets B's elites (replacing A's worst)
        new_island_a = island_a[:-num_share] + elites_b
        # Swap: Island B gets A's elites (replacing B's worst)
        new_island_b = island_b[:-num_share] + elites_a
        
        return new_island_a, new_island_b
