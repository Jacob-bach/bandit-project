"""
run_experiments.py

Example script to demonstrate running different bandit policies
(basic_greedy, Bayesian DP) on the Bandit environment.
"""
import time
from bandit_env import Bandit
from policies import basic_greed, bay_dp, ucb

def main():
    start_time = time.time()
    # Initialize a bandit environment
    bandit_env = Bandit(
        arms=6,
        turns=15,
        seed=42,
        dist='beta',
        dist_params={'a': 1, 'b': 1}
    )
    print("True reward probabilities: ", bandit_env.true_probs)

    print(" === Basic Greedy Policy === ")
    # Run the basic greedy policy
    basic_greed(bandit_env)

    print(" === Upper Confidence Bound (UCB) Policy === ")
    # Run the UCB policy
    ucb(bandit_env)

    print(" === Bayesian DP Policy === \n Warning - This policy may take a while to run!")
    # Run the Bayesian DP policy
    bay_dp(bandit_env)

    # Calculate and print the elapsed time
    end_time = time.time()
    min = (end_time - start_time) // 60
    sec = (end_time - start_time) % 60
    print(f"Total Execution Time: {int(min)} min {round(sec)} sec")

# Run the main function
if __name__ == "__main__":
    main()