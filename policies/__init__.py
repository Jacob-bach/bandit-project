"""
policies Package

This directory contains various RL policies (strategies) That can interact with the 
Bandit environment. Each policy typically defines a function that runs until the bandit 
horizon is reached.
"""

# Allows for imports like 'from policies import basic_greed':
from .basic_greedy import basic_greed
from .bayesian_dp import bay_dp
from .upper_conf_bound import ucb