## Deep Q-Network (CartPole)
The goal of this exercise is to implement and experiment with both single-core and distributed versions of the deep reinforcement learning algorithm 
[Deep Q Networks (DQN)](https://deepmind.com/research/open-source/dqn). 
In particular, DQN is run in the classic RL benchmark Cart-Pole and ablation experiments are run to observe the impact of the different DQN components.

The general DQN algorithm is as follows:

1. Initial “experience replay” data set D
2. Initialize update networkparameters θ
3. Initialize target network parameters θ'
4. Take action according to explore/exploit policy based on θ
5. Add observed transition (s, a, r, s') to D (limit size of D to N)
6. Randomly sample a mini-batch of B transition {s<sub>k</sub>, a<sub>k</sub>, r<sub>k</sub>, s'<sub>k</sub>} from D
7. Perform a Q-learning update on θ based on mini-batch. Use target network for targets target = r + 𝛽. max Q<sub>θ'</sub>(s', a')
8. After every K updates θ' ← θ(refresh target network)
9. Goto 3

[CartPole-v0](https://gym.openai.com/envs/CartPole-v0/) is used as the MDP environment.

The single core(non-distributed) version of the DQN algorithm is experimented with/without usage of replay buffers and target network.

It can be summarized from the experiments that:

a. Updating parameters on randomly sampling mini-batches provides efficient use of memory as well as breaks correlation among updates; 
leading to reduction in variance.

b. Not using a target network results in fluctuated learning and might blow up parameters.

Following are the performance curves for Full DQN in a distributed setting with different combinations of evaluation workers and collector workers:

<p float="left">

Collectors: 2, Evaluators: 4

<img src="/DQN - Cartpole/Performance Curves/dqn_2c_4w.png?raw=true" alt="8x8" width="400"/>
Collectors: 4, Evaluators: 4

<img src="/DQN - Cartpole/Performance Curves/dqn_4c_4w.png?raw=true" alt="16x16" width="400"/>
Collectors: 8, Evaluators: 4

<img src="/DQN - Cartpole/Performance Curves/dqn_8c_4w.png?raw=true" alt="32x32" width="400"/>
Collectors: 16, Evaluators: 4

<img src="/DQN - Cartpole/Performance Curves/dqn_16c_4w.png?raw=true" alt="32x32" width="400"/>
</p>

1. With the decrease in number of collector-workers from 16 to 8 to 4 to 2, we can observe a proportional increment
in runtime. Also notice that this time for distributed DQN is much higher than non-distributed runtime;
we attribute this fact to considerably slower cart-pole environment (Custom-Cartpole) in distributed case.

2. From the above DQN learning curves, it can be noted that after around 5000 epochs, the agent is consistently
performing well with near-maximum rewards.

3. The number of collector-workers shouldn't have an affect on total rewards vs #episodes plot. This is because
the mini-batches formed for Model Server contain i.i.d. samples in all cases and the batch sizes are consistent
too.

The [Jupyter Notebook](https://github.com/rusty-sj/Reinforcement-Learning-Algorithms-with-Ray-Framework-and-Intel-DevCloud/blob/master/DQN%20-%20Cartpole/non_distributed_dqn.ipynb) 
has single core implementation of DQN along with the plots comparing the different variations of DQN.

The [Python file](https://github.com/rusty-sj/Reinforcement-Learning-Algorithms-with-Ray-Framework-and-Intel-DevCloud/blob/master/DQN%20-%20Cartpole/distributed_dqn.py) 
has the code for DQN in a distributed setting.
