## Value Iteration algorithm with FrozenLake MDP environment
The goal of this exercise is to implement both single-core and distributed versions of value iteration (VI) algorithm for solving Markov Decision Processes (MDPs). 
In particular, VI will be applied to MDPs in order to compute policies that optimize expected infinite horizon discounted cummulative reward. 


The VI algorithm is iterative and each iteration produces a newly updated value function based on the value function from the previous iteration. 
This is done by applying the [Bellman backup operator](https://en.wikipedia.org/wiki/Bellman_equation) to previous value function at each state.


![Bellman Backup Equation](https://wikimedia.org/api/rest_v1/media/math/render/svg/ab046e5fb76162018d9bed802c83d9f80a64e7b4)

The pseudocode for the algorithm is as follows:
- Start with V<sub>curr</sub>(s) = 0 for all s
- error = ∞
- While error > ϵ
    - For each state s
        - Compute V<sub>new</sub>(s) using Bellman backup operator
        - π<sub>curr</sub>(s) = action that led to maximum value
    - error = max<sub>s</sub>|V<sub>new</sub>(s) − V<sub>curr</sub>(s)|
- V<sub>curr</sub> = V<sub>new</sub>

[FrozenLake](https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py) is used as the MDP environment.

Following are the State Value image views after running the VI algorithm 3 different maps of sizes 8x8, 16x16, and 32x32 wherein cells represent number of states:
![8x8]()
![16x16]()
![32x32]()

The Jupyter Notebook has synchronous and distributed implementations of VI along with the plots comparing the different approaches.
