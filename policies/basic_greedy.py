import numpy as np

def basic_greed(world):
    """
    A basic pure greedy policy that picks the arm with the highest success rate so far.
    In case of ties, picks one at random.
    
    Args:
        world (Bandit): The bandit environment to interact with.
    
    Returns:
        None. The function runs until the horizon is reached, then it calls 
        world.store_strat() to record the results.
    """
    while world.turn <= world.h:
        curr_state = world.get_state()

        # Calculate the success rate for each arm (w / n), defaulting to 0 if n = 0.
        perc_chance = {}
        for arm, (n, w) in enumerate(curr_state):
            perc_chance[arm] = w / n if n > 0 else 0
        
        # Find arms with the highest success rate
        max_chance = max(perc_chance.values())
        best_arms = [arm for arm, chance in perc_chance.items() if chance == max_chance]
        
        # Randomly pick one among the best arms
        chosen_arm = np.random.choice(best_arms)

        # Pull the chosen arm
        world.pull(chosen_arm)

    # At the end of the horizon, record performance and reset environment
    world.store_strat("Basic Greedy")
