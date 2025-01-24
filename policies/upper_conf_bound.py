import math
import numpy as np

def ucb(world, c=1.0, temp=False):
    """
    Upper Confidence Bound (UCB) policy for multi-armed bandit problems. 
    Exploration term is set to infinity if arm_i has not been chosen yet to ensure each arm is chosen at least once.

    Args:
        world (Bandit): The bandit environment to interact with.
    
    Returns:
        None. The function runs until the horizon is reached, then it calls 
        world.store_strat() to record the results.
    """
    while world.turn <= world.h:
        curr_state = world.get_state()
        arm_conf = {}

        if temp:
            c = c / math.sqrt(world.turn)     # Decay c over time (Temperature)
        
        for arm, (n, w) in enumerate(curr_state):
            mean_reward = w / n if n > 0 else 0
            t = world.turn
            
            exploration = math.sqrt( c * math.log(t) / n ) if n > 0 else float('inf')
            uncertainty = c * exploration
            conf_val = mean_reward + uncertainty
            arm_conf[arm] = conf_val

        high_val = max(arm_conf.values())
        highest_arms = [arm for arm, val in arm_conf.items() if val == high_val]
        chosen_arm = np.random.choice(highest_arms)
        world.pull(chosen_arm)

    world.store_strat("Upper Confidence Bound (UCB)")