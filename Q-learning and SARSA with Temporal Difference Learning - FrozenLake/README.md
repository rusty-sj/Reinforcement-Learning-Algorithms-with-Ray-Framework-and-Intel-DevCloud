## Q-learning and SARSA using Temporal Difference Learning (FrozenLake)
The goal of this exercise is to implement both single-core and distributed versions of of two [temporal difference (TD)](https://en.wikipedia.org/wiki/Temporal_difference_learning) reinforcement learning algorithms, 
Q-Learning and SARSA. In particuar, Q-Learning and SARSA will be run in a Markov Decision Process environment in order to compute policies 
that optimize expected infinite horizon discounted cummulative reward.


[Q-Learning](https://en.wikipedia.org/wiki/Q-learning) and [SARSA](https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action) appear to be extremely similar algorithms at the code level. In particular, their update formulas are as follows:

```SARSA: Q(s, a) ← Q(s, a) + α⋅(r + β.Q(s′, a′) − Q(s, a))```

```Q-Learning: Q(s, a) ← Q(s, a) + α⋅(r + β.maxQ(s′, a′) − Q(s, a))```

where s′ and r are the next state and reward after taking action a in state s. SARSA is an on-policy algorithm, which aims to learn the value of the policy that 
is being used to collect data (we call this the behavior policy). In the case of policy optimization, this behavior policy is an exploration-exploitation policy. 

Rather, Q-Learning is an off-policy algorithm, which can learn from any set of experience tuples, regardless of how they were collected. 
In particular, Q-Learning is not sensitive to what action (if any) was taken in s′, while SARSA must be given the action that was actually taken in s′. Even if 
the exploration policy were completely random, Q-Learning will converge in the limit to an optimal policy under standard assumptions on the learning rate decay.
The off-policy versus on-policy difference can lead the algorithms to sometimes learn different policies in practice. 
The off-policy nature of Q-Learning makes it more directly compatible with distributed implementations compared to SARSA, since the experience need not come from a particular behavior policy. 

[FrozenLake](https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py) is used as the MDP environment.

As seen in the diagram below, the 4-core version of Q-learning performs better than both single core Q-learning and single core SARSA.

<img src="/Q-learning and SARSA with Temporal Difference Learning - FrozenLake/Performance Graphs/Dist-vs-Nondist.png?raw=true" alt="8x8" width="600"/>

However, if we experiment with changing the number of cores in distributed setting, the performance enhances from 2-core to 4-core and then it degrades from 4-core to 8-core. This could probably mean that the communication overhead overwhelms the 8-core version; leading to worse performance.

<img src="/Q-learning and SARSA with Temporal Difference Learning - FrozenLake/Performance Graphs/workers.png" alt="8x8" width="600"/>

[Notebook for map 8x8](https://github.com/rusty-sj/Reinforcement-Learning-Algorithms-with-Ray-Framework-and-Intel-DevCloud/blob/master/Q-learning%20and%20SARSA%20with%20Temporal%20Difference%20Learning%20-%20FrozenLake/TD-Learning-map-8.ipynb)

[Notebook for map 16x16](https://github.com/rusty-sj/Reinforcement-Learning-Algorithms-with-Ray-Framework-and-Intel-DevCloud/blob/master/Q-learning%20and%20SARSA%20with%20Temporal%20Difference%20Learning%20-%20FrozenLake/TD-Learning-map-16.ipynb)


