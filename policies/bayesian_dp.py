def bay_dp(world):
    """
    A Bayesian policy that uses dynamic programming (DP) to find the optimal
    arm based on expected future reward.

    Args:
        world (Bandit): The bandit environment to interact with.
    
    Returns:
        None. The function runs until the horizon is reached, then it calls 
        world.store_strat() to record the results.
    """

    def update_state(state, arm, results):
        """
        Update the state given a specific arm to pull outcome.

        Args:
            state (list): List of (n, w) tuples for each arm.
            arm (int): arm index to update.
            results (bool): 'True' for success, 'False' for failure.
        """
        n, w = state[arm]
        state[arm] = (n + 1, w + results)
        return state
    
    def value(val_map, state, remaining_pulls):
        """
        Recursively compute the maximum expected future reward for the current state,
        with a given number of pulls left.

        Args:
            val_map (dict): Memoization dictionary mapping (state_key, remaining_pulls) to (best_val, best_arm_index).
            state (list): List of (n, w) tuples for each arm.
            remaining_pulls (int): Number of pulls left in the horizon.
        
        Return:
            tuple: (best_val, best_arm) representing the maximum expected future reward and the associated arm.
        """
        if remaining_pulls == 0:
            return (0,-1)
        
        # Convert state to a tuple so it can be a dict key.
        state_key = tuple(state)
        if (state_key, remaining_pulls) in val_map:
            return val_map[(state_key, remaining_pulls)]
        
        best_val = float('-inf')
        best_arm = None

        for arm_index, (n, w) in enumerate(state):

            # Bayesian update
            p_i = (w + 1) / (n + 2)                                                   # Posterior probability of success (beta mean for arm_i)

            # Simulate success
            state_success = update_state(state.copy(), arm_index, True)               # Simulate successful arm pull
            value_success = value(val_map, state_success, remaining_pulls - 1)[0]     # Compute future value based on success outcome

            # Simulate failure
            state_failure = update_state(state.copy(), arm_index, False)              # Simulate failed arm pull
            value_failure = value(val_map, state_failure, remaining_pulls - 1)[0]     # Compute future value based on failure outcome

            # Formula : Probability of success * (Immediate Reward + Future Value of Success) + Probability of failure * (Immediate Reward + Future Value of Failure)
            curr_val = p_i * (1 + value_success) + (1 - p_i) * (0 + value_failure)    # Total Expected Value for arm_i

            if curr_val > best_val:                                                   # Update arm with highest Total Expected Value
                best_val = curr_val
                best_arm = arm_index
            
        val_map[(state_key, remaining_pulls)] = (best_val, best_arm)                  # Cache best value for this state and remaining pulls
        return (best_val, best_arm)
    
    val_map = {}                                                                      # Memoization dictionary, persists across turns

    while world.turn <= world.h:                                                      # Run until horizon is reached
        curr_state = world.get_state()                                                # Get current state from environment
        pulls_left = world.h - world.turn + 1
        _, best_arm_index = value(val_map, curr_state, pulls_left)                    # Compute best arm to pull

        if best_arm_index is None or best_arm_index == -1:                            # Check if best_arm_index contains valid arm index
            raise ValueError("No best arm found. Check the value function or state representation.")
        
        world.pull(best_arm_index)                                                    # Interact with real environment

    world.store_strat("Bayesian DP")                                                  # Record performance and reset environment