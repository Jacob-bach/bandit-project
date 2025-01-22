# Bandit Environment

the below code implements the k-armed bandit problem which navigates the exploration-exploitation dilemma:

- There are k slot machines standing in front of an agent.
- The agent has h turns, where they must pick a slot machine's arm to pull per turn.
- The slot machine either produces a 1 or 0 as the reward.
- Each slot machine has it's own reward probability distribution unknown to the agent.
- The goal is to maximize the total reward after h turns.

The optimal strategy to the problem challenges the agent to strike a balance of exploring the individual probability distributions of the different machines (exploration) and maximizing profits based on the information acquired so far (exploitation)

Properties:
currently, only supporting a stationary environment. Future updates will occur to allow the option of a non-stationary environment.
- a stationary environment means that the underlying true probabilities do not change over time so in our case, a non-stationary environment would mean that with each turn. The arms underlying probabilities would shift over turns
- As the case with the historical k-bandit problem, our output for a pull is either 'success' or 'failure', in the future there will be an option for a continous variable output so that more policies can be applicable to the bandit environment

![0_l7Ra4R_CpJfc-hjz](https://github.com/user-attachments/assets/e18cee8e-f77f-4043-8f83-e307928a6e5b)

# Motivation

I created this project to help myself understand the core concepts and algorithms described in a Rienforcement Learning Survey. The code is organized in such a way that new policies can be easily added, and experiments can be run from a dedicated scripts folder. 

