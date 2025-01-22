import numpy as np

class Bandit:
    """
    A k-armed bandit environment.

    Attributes:
        k (int): Number of arms.
        h (int): Total number of pulls (horizon).
        turn (int): Current turn number.
        scores (list): Stores performance results of different strategies.
        state (list): Internal list of tuples (n, w), where n is pulls, w is total reward.
        history (list): Records interactions (turn, arm, reward).
        true_probs (np.ndarray): The hidden probabilities for each arm.
    """

    def __init__(self, arms, turns, seed=None, dist='uniform', dist_params=None):
        '''
        Initialize a k-armed bandit environment. 
        The environment state is automatically updated each time you pull an arm. 

        Args:
            arms (int): The number of arms (k).
            turns (int): The total number of pulls (horizon).
            seed (int): Random seed used for reproducibility.
            dist (str, optional): Distribution type to generate true reward probabilities
                ('uniform' or 'beta'). Defaults to 'uniform'.
            dist_params (dict, optional): Distribution parameters if using 'beta'.
                E.g., {'a' : 1, 'b' : 1}
        '''
        # set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Validate input
        if arms <= 0 or turns <= 0:
            raise ValueError("The number of arms and turns must be positive integers.")
        
        self.k = arms                     # Number of arms (k)
        self.h = turns                    # Total number of pulls (horizon)
        self.turn = 1                     # Current turn 
        self.scores = []                  # Store payout of different RL strategies
        
        # Internal state: 
        # Each element is a tuple (n, w): n = number of pulls, w = total reward for that arm
        # one tuple per arm
        self.state = [(0, 0) for _ in range(self.k)]  
        
        # History recording: will store (turn_number, arm_index, reward)
        self.history = []
        
        # True reward probabilities (hidden from the learning agent)
        if dist == 'uniform':
            
            # Sample true reward probabilities uniformly between 0 and 1
            self.true_probs = np.random.rand(self.k)
            
        elif dist == 'beta':
            # Use a beta distribution, Set default Beta(1,1) if no parameters provided
            a = 1
            b = 1
            if dist_params is not None:
                a = dist_params.get('a', 1)
                b = dist_params.get('b', 1)
            self.true_probs = np.random.beta(a, b, size=self.k)
        else:
            raise ValueError("Distribution type not recognized. Use 'uniform' or 'beta'.")
    
    def reset(self):
        '''
        Reset the bandit's state and history to the initial condition.
        '''
        self.state = [(0, 0) for _ in range(self.k)]
        self.history = []
        self.turn = 1
    
    def pull(self, arm_index):
        '''
        Simulate pulling a specific arm.
        
        Args:
            arm_index (int): The index of the arm to pull.
            
        Returns:
            reward (int): The reward obtained (1 for success, 0 for failure).
        '''
        # Validate # of turns
        if not (1 <= self.turn <= self.h):
            raise ValueError("turn must be between 1 and h.")
            
        # Validate arm index
        if not (0 <= arm_index < self.k):
            raise ValueError("arm_index must be between 0 and k-1.")
        
        # Determine if the pull results is a success by using a bernoulli distribution
        prob_success = self.true_probs[arm_index]
        reward = np.random.binomial(1, prob_success)
        
        # Update the state for that arm
        n, w = self.state[arm_index]
        new_n = n + 1             # Number of times arm has been pulled after this pull
        new_w = w + reward        # Total reward for arm after this pull
        self.state[arm_index] = (new_n, new_w)
        
        # Record the pull in history (turn, arm_index, reward)
        self.history.append({"turn": self.turn, "arm": arm_index, "payout" : reward})
        
        # Print state after pull
        message = f"Turn {self.turn}: Pulled arm {arm_index}. "
        message += "Result: SUCCESS. " if reward else "Result: FAILURE. "
        message += f"Arm {arm_index} state: (#_of_pulls: {new_n}, total_reward: {new_w}). "
        
        print(message)
        self.turn += 1
        
        return reward
        
    def get_state(self):
        """
        Return the current state as a list of (n, w) tuples.
        """
        return list(self.state)
    
    def get_history(self):
        """
        Return the history of pulls (turn, arm_index, reward).
        """
        return list(self.history)
    
    def store_strat(self, name):
        """
        store the payout of the current strategy and call get_scores().

        Args:
            name (str): A name or label for the strategy.
        """
        payout = sum([score[1] for score in self.state])
        self.scores.append((name, payout))
        
        self.get_scores()                        # Print performance of policy
        self.reset()                             # Reset environment
    
    def get_scores(self):
        """
        prints the payouts of all RL strategies on current bandit environment.
        """
        print('---Strategy Performance---')
        for strategy_name, payout in self.scores:
            print(f"Strategy: {strategy_name} | "
                  f"Total Payout: {payout} | "
                  f"Turns: {self.h}")
        print('---------------------------------')