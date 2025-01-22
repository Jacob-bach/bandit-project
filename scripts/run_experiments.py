"""
run_experiments.py

Example script to demonstrate running different bandit policies
(basic_greedy, Bayesian DP) on the Bandit environment.
"""

from bandit_env import Bandit
from policies import basic_greed, bay_dp

def main():
    # Initialize a bandit environment
    bandit_env = Bandit(
        arms=6,
        turns=15,
        seed=33,
        dist='beta',
        dist_params={'a': 1, 'b': 1}
    )
    print("True reward probabilities: ", bandit_env.true_probs)
    print(" === Basic Greedy Policy === ")

    # Run the basic greedy policy
    basic_greed(bandit_env)

    print(" === Bayesian DP Policy === ")

    # Run the Bayesian DP policy
    bay_dp(bandit_env)


# Run the main function
if __name__ == "__main__":
    main()