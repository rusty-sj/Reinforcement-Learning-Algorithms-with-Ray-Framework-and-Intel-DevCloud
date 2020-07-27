import os
import time
from random import uniform, randint

import matplotlib.pyplot as plt
import ray
import torch
from tqdm import tqdm

from custom_cartpole import CartPoleEnv
from dqn_model import DQNModel
from memory_remote import ReplayBuffer_remote

FloatTensor = torch.FloatTensor

# Set the Env name and action space for CartPole
ENV_NAME = 'CartPole-Dist-v0'
# Move left, Move right
ACTION_DICT = {
    "LEFT": 0,
    "RIGHT": 1
}

# Set result saveing floder
result_floder = ENV_NAME
result_file = ENV_NAME + "/results.txt"
if not os.path.isdir(result_floder):
    os.mkdir(result_floder)
torch.set_num_threads(12)

ray.shutdown()
ray.init(include_webui=False, ignore_reinit_error=True, redis_max_memory=500000000, object_store_memory=5000000000)


def plot_result(total_rewards, learning_num, legend):
    print("\nLearning Performance:\n")
    episodes = []
    for i in range(len(total_rewards)):
        episodes.append(i * learning_num + 1)

    plt.figure(num=1)
    fig, ax = plt.subplots()
    plt.plot(episodes, total_rewards)
    plt.title('performance')
    plt.legend(legend)
    plt.xlabel("Episodes")
    plt.ylabel("total rewards")
    plt.savefig(ENV_NAME + "/dqn_2c_4w.png")
    # plt.show()
    plt.close(fig)


@ray.remote
class Model_Server():
    def __init__(self, env, hyper_params, memory, action_space):
        self.epsilon_decay_steps = hyper_params['epsilon_decay_steps']
        self.final_epsilon = hyper_params['final_epsilon']
        self.batch_size = hyper_params['batch_size']
        self.update_steps = hyper_params['update_steps']
        self.beta = hyper_params['beta']
        self.model_replace_freq = hyper_params['model_replace_freq']
        self.learning_rate = hyper_params['learning_rate']
        self.training_episodes = hyper_params['training_episodes']
        self.test_interval = hyper_params['test_interval']
        self.memory = memory

        self.episode = 0
        self.steps = 0
        self.result_count = 0
        self.next = 0
        self.batch_num = self.training_episodes // self.test_interval

        state = env.reset()
        input_len = len(state)
        output_len = action_space
        self.eval_model = DQNModel(input_len, output_len, learning_rate=hyper_params['learning_rate'])
        self.target_model = DQNModel(input_len, output_len)

        self.results = [0] * (self.batch_num + 1)
        self.previous_q_networks = []

        self.collector_done = False
        self.evaluator_done = False

    def ask_evaluation(self):
        if len(self.previous_q_networks) > self.result_count:
            num = self.result_count
            evaluation_q_network = self.previous_q_networks[num]
            self.result_count += 1
            return evaluation_q_network, False, num
        else:
            if self.episode >= self.training_episodes:
                self.evaluator_done = True
            return [], self.evaluator_done, None

    def get_evaluation_model(self):
        if self.episode >= self.training_episodes:
            self.collector_done = True
        return self.eval_model, self.collector_done

    def replace_with_eval_model(self):
        self.target_model.replace(self.eval_model)

    def get_model_steps(self):
        return self.steps

    def predict_next_eval(self, state, eval_model):
        return eval_model.predict(state)

    def get_predict(self, state):
        return self.eval_model.predict(state)

    def increment_episode(self):
        self.episode += 1

    def increment_model_steps(self):
        self.steps += 1
        return self.steps

    def update_batch(self):

        self.steps += self.update_steps

        if ray.get(self.memory.__len__.remote()) < self.batch_size:  # or self.steps % self.update_steps != 0:
            return

        if self.collector_done:
            return

        batch = ray.get(self.memory.sample.remote(self.batch_size))

        (states, actions, reward, next_states,
         is_terminal) = batch

        states = states
        next_states = next_states
        terminal = FloatTensor([1 if t else 0 for t in is_terminal])
        reward = FloatTensor(reward)
        batch_index = torch.arange(self.batch_size, dtype=torch.long)

        # Current Q Values
        _, q_values = self.eval_model.predict_batch(states)
        q_values = q_values[batch_index, actions]

        # Calculate target
        actions, q_next = self.target_model.predict_batch(next_states)

        q_max, indices = torch.max(q_next, dim=1)

        # INSERT YOUR CODE HERE --- neet to compute 'q_targets' used below
        q_targets = []
        for i, is_term in enumerate(terminal):
            if is_term == 1:
                q_targets.append(reward[i])
            else:
                q_targets.append(reward[i] + self.beta * q_max[i])
        q_targets_tensor = FloatTensor(q_targets)

        # update model
        self.eval_model.fit(q_values, q_targets_tensor)

        if self.episode // self.test_interval + 1 > len(self.previous_q_networks):
            model_id = ray.put(self.eval_model)
            self.previous_q_networks.append(model_id)
        return self.steps

    def add_result(self, reward, num):
        self.results[num] = reward

    def get_results(self):
        return self.results


@ray.remote
def collecting_worker(model_server, env, max_episode_steps, epsilon_decay_steps, final_epsilon, update_steps,
                      model_replace_freq, test_interval, memory, action_space):
    def linear_decrease(initial_value, final_value, curr_steps, final_decay_steps):
        decay_rate = curr_steps / final_decay_steps
        if decay_rate > 1:
            decay_rate = 1
        return initial_value - (initial_value - final_value) * decay_rate

    def greedy_policy_collector(state):
        return ray.get(model_server.get_predict.remote(state))

    def explore_or_exploit_policy(state):
        p = uniform(0, 1)
        # Get decreased epsilon
        self_steps = ray.get(model_server.get_model_steps.remote())
        epsilon = linear_decrease(initial_value=1, final_value=final_epsilon, curr_steps=self_steps,
                                  final_decay_steps=epsilon_decay_steps)

        if p < epsilon:
            # return action
            return randint(0, action_space - 1)
        else:
            # return action
            return greedy_policy_collector(state)

    while True:
        e_model, collect_done = ray.get(model_server.get_evaluation_model.remote())
        if collect_done:
            break
        # if not e_model:
        #     continue

        for episode in tqdm(range(test_interval), desc="Training"):
            state = env.reset()
            done = False
            steps = 0
            model_server.increment_episode.remote()

            while steps < max_episode_steps and not done:
                # INSERT YOUR CODE HERE
                # add experience from explore-exploit policy to memory
                # update the model every 'update_steps' of experience
                # update the target network (if the target network is being used) every 'model_replace_freq' of experiences
                action = explore_or_exploit_policy(state)
                obs_state, reward, done, _ = env.step(action)
                memory.add.remote(state, action, reward, obs_state, done)
                steps += 1
                # model_steps = ray.get(model_server.increment_model_steps.remote())
                state = obs_state

                if steps % update_steps == 0:
                    model_server.update_batch.remote()

                model_steps = ray.get(model_server.get_model_steps.remote())
                if model_steps % model_replace_freq == 0:
                    model_server.replace_with_eval_model.remote()


@ray.remote
def evaluation_worker(model_server, env, max_episode_steps, trials=30):
    def greedy_policy_evaluator(state, eval_model):
        return ray.get(model_server.predict_next_eval.remote(state, eval_model))

    while True:
        model_id, done, num = ray.get(model_server.ask_evaluation.remote())
        eval_model = ray.get(model_id)

        if done:
            break
        total_reward = 0
        if not eval_model:  #
            continue
        for _ in tqdm(range(trials), desc="Evaluating"):
            state = env.reset()
            done = False
            steps = 0

            while steps < max_episode_steps and not done:
                steps += 1
                action = greedy_policy_evaluator(state, eval_model)
                state, reward, done, _ = env.step(action)
                total_reward += reward

        avg_reward = total_reward / trials
        model_server.add_result.remote(avg_reward, num)

        #print(avg_reward)
        f = open(result_file, "a+")
        f.write(str(avg_reward) + "\n")
        f.close()

    return avg_reward


class DQN_agent():

    def __init__(self, env, hyper_params, cw_num, ew_num, memory, action_space):
        self.env = env
        self.max_episode_steps = env._max_episode_steps

        self.model_server = Model_Server.remote(env, hyper_params, memory, action_space)

        self.epsilon_decay_steps = hyper_params['epsilon_decay_steps']
        self.final_epsilon = hyper_params['final_epsilon']
        self.batch_size = hyper_params['batch_size']
        self.update_steps = hyper_params['update_steps']
        self.beta = hyper_params['beta']
        self.model_replace_freq = hyper_params['model_replace_freq']
        self.learning_rate = hyper_params['learning_rate']
        self.training_episodes = hyper_params['training_episodes']
        self.test_interval = hyper_params['test_interval']

        self.cw_num = cw_num
        self.ew_num = ew_num
        self.memory = memory
        self.action_space = action_space
        self.workers_id = []

    def learn_and_evaluate(self):

        # workers_id = []

        for i in range(self.cw_num):
            cw_id = collecting_worker.remote(self.model_server, self.env, self.max_episode_steps,
                                             self.epsilon_decay_steps, self.final_epsilon, self.update_steps,
                                             self.model_replace_freq, self.test_interval,
                                             self.memory, self.action_space)
            self.workers_id.append(cw_id)
        for i in range(self.ew_num):
            ew_id = evaluation_worker.remote(self.model_server, self.env, self.max_episode_steps, self.ew_num)
            self.workers_id.append(ew_id)

        ray.wait(self.workers_id, len(self.workers_id))
        return ray.get(self.model_server.get_results.remote())


hyperparams_CartPole = {
    'epsilon_decay_steps': 100000,
    'final_epsilon': 0.1,
    'batch_size': 32,
    'update_steps': 10,
    'beta': 0.99,
    'model_replace_freq': 2000,
    'learning_rate': 0.0003,
    'training_episodes': 10000,
    'test_interval': 50
}

simulator = CartPoleEnv()
Memory_Server = ReplayBuffer_remote.remote(2000)  # memory_size = 2000

start_time = time.time()
dqn_agent = DQN_agent(simulator, hyperparams_CartPole, cw_num=2, ew_num=4, memory=Memory_Server,
                      action_space=len(ACTION_DICT))
result = dqn_agent.learn_and_evaluate()
run_time = time.time() - start_time
print("Learning time:\n", run_time)
plot_result(result, 50, ["branch_update with target_model"])
