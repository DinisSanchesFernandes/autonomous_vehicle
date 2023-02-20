import tensorflow.keras.optimizers as optim
import tensorflow as tf
import numpy as np
import gym
import pickle

from PPO.Policy import Policy
from PPO.GAE_tf2 import estimate_advantages
from PPO.ppo_step import ppo_step
from PPO.Value import Value
from PPO.zfilter import ZFilter
from PPO.MemoryCollector import MemoryCollector
from PPO.tf2_util import NDOUBLE, TDOUBLE
from PPO.file_util import check_path
import tensorflow as tf
import datetime
import cv2


class PPO:
    def __init__(self,
                 env_id,
                 render=False,
                 num_process=1,
                 min_batch_size=10,
                 lr_p=3e-4,
                 lr_v=3e-4,
                 gamma=0.99,
                 tau=0.95,
                 clip_epsilon=0.2,
                 ppo_epochs=1,
                 ppo_mini_batch_size=0,
                 seed=1,
                 model_path=None
                 ):
        self.env = env_id
        self.gamma = gamma
        self.tau = tau
        self.ppo_epochs = ppo_epochs
        self.ppo_mini_batch_size = ppo_mini_batch_size
        self.clip_epsilon = clip_epsilon
        self.render = render
        self.num_process = num_process
        self.lr_p = lr_p
        self.lr_v = lr_v
        self.min_batch_size = min_batch_size
        self.model_path = model_path
        self.seed = seed
        #self.store_path = store_path
        self._init_model()

    def _init_model(self):

        # Get Environment Data
        #self.env = gym.make(self.env_id)
        num_states = self.env.get_state_space_dimensions()
        num_actions = self.env.get_action_space_dimensions()
        #print("States: ", num_states, "Actions: ", num_actions)

        tf.keras.backend.set_floatx('float64')

        # Seting the seed makes the "random" numbers
        # Not change
        # np.random.seed(self.seed)
        # tf.random.set_seed(self.seed)
        # self.env.seed(self.seed)

        self.policy_net = Policy(num_states, num_actions)
        self.value_net = Value(num_states)
        self.running_state = ZFilter((num_states,), clip=5)

        if self.model_path:
            print("Loading Saved Model {}_ppo_tf2.p".format("Autonomous_Vehicle"))
            self.running_state = pickle.load(
                open('{}/{}_ppo_tf2.p'.format(self.model_path, "Autonomous_Vehicle"), "rb"))
            self.policy_net.load_weights(
                "{}/{}_ppo_tf2_p".format(self.model_path, "Autonomous_Vehicle"))
            self.value_net.load_weights(
                "{}/{}_ppo_tf2_v".format(self.model_path, "Autonomous_Vehicle"))

        self.collector = MemoryCollector(
            self.env, self.policy_net, render=self.render,
            running_state=self.running_state, num_process=self.num_process)

        self.optimizer_p = optim.Adam(learning_rate=self.lr_p, clipnorm=20)
        self.optimizer_v = optim.Adam(learning_rate=self.lr_v)

    def choose_action(self, state):
        """select action"""
        state = np.expand_dims(NDOUBLE(state), 0)
        action, log_prob = self.policy_net.get_action_log_prob(state)
        action = action.numpy()[0]
        return action

    def choose_action_eval(self,state):
        state = np.expand_dims(NDOUBLE(state), 0)
        action = self.policy_net.get_action(state)
        #print(action)
        action = action.numpy()[0]
        #print(action)
        return action

    def eval(self, i_iter, render=False):
        """init model from parameters"""
        state = self.env.reset()
        test_reward = 0
        step = 0
        max_steps = 2048

        if render:
            writer = self.env.render_init()

        while step < max_steps:
            
            step += 1

            if render:
                self.env.render(writer)
            state = self.running_state(state)

            action = self.choose_action_eval(state)
            #print("Action: ",action)
            state, reward, done = self.env.step(action)

            #Show ComputerVision
            #cv2.imshow("Frame",frame)
            #cv2.waitKey(2)
            #print("Reward: ", reward," Angle: ",action[0], "Speed: ", action[1])

            test_reward += reward
            if done:

                if render:
                    self.env.render_close(writer)

                break
        #print(f"Iter: {i_iter}, test Reward: {test_reward}")
        self.env.close()
        return test_reward, step

    def learn(self, writer, i_iter):
        """learn model"""
        memory, log = self.collector.collect_samples(self.min_batch_size)

        print(f"Iter: {i_iter}, num steps: {log['num_steps']}, total reward: {log['total_reward']: .4f}, "
              f"min reward: {log['min_episode_reward']: .4f}, max reward: {log['max_episode_reward']: .4f}, "
              f"average reward: {log['avg_reward']: .4f}, sample time: {log['sample_time']: .4f}")

        # record reward information
        with writer.as_default():
            tf.summary.scalar("total reward", log['total_reward'], i_iter)
            tf.summary.scalar("average reward", log['avg_reward'], i_iter)
            tf.summary.scalar("min reward", log['min_episode_reward'], i_iter)
            tf.summary.scalar("max reward", log['max_episode_reward'], i_iter)
            tf.summary.scalar("num steps", log['num_steps'], i_iter)

        batch = memory.sample()  # sample all items in memory

        batch_state = NDOUBLE(batch.state)
        batch_action = NDOUBLE(batch.action)
        batch_reward = NDOUBLE(batch.reward)
        batch_mask = NDOUBLE(batch.mask)
        batch_log_prob = NDOUBLE(batch.log_prob)[:, None]
        batch_value = tf.stop_gradient(self.value_net(batch_state))

        batch_advantage, batch_return = estimate_advantages(batch_reward, batch_mask, batch_value, self.gamma,
                                                            self.tau)

        log_stats = {}
        if self.ppo_mini_batch_size:
            batch_size = batch_state.shape[0]
            mini_batch_num = batch_size // self.ppo_mini_batch_size

            for e in range(self.ppo_epochs):
                perm = np.random.permutation(batch_size)
                for i in range(mini_batch_num):
                    ind = perm[slice(
                        i * self.ppo_mini_batch_size, min(batch_size, (i + 1) * self.ppo_mini_batch_size))]

                    aux_batch_return = batch_return.numpy()

                    aux_batch_return = aux_batch_return[ind]

                    aux_batch_return = tf.convert_to_tensor(aux_batch_return)

                    aux_batch_advantage = batch_advantage.numpy()[ind]

                    aux_batch_advantage = tf.convert_to_tensor(
                        aux_batch_advantage)

                    log_stats = ppo_step(self.policy_net, self.value_net, self.optimizer_p, self.optimizer_v, 1,
                                         batch_state[ind], batch_action[ind], aux_batch_return, aux_batch_advantage, batch_log_prob[ind],
                                         self.clip_epsilon
                                         )

        else:
            for _ in range(self.ppo_epochs):
                log_stats = ppo_step(self.policy_net, self.value_net, self.optimizer_p, self.optimizer_v, 1,
                                     batch_state, batch_action, batch_return, batch_advantage, batch_log_prob,
                                     self.clip_epsilon)

        with writer.as_default():
            tf.summary.histogram("ratio", log_stats["ratio"], i_iter)
            tf.summary.scalar("policy loss", log_stats["policy_loss"], i_iter)
            tf.summary.scalar("critic loss", log_stats["critic_loss"], i_iter)
        writer.flush()
        return log

    def save(self, save_path):
        """save model"""
        check_path(save_path)
        pickle.dump(self.running_state,
                    open('{}/{}_ppo_tf2.p'.format(save_path, "Autonomous_Vehicle"), 'wb'))
        self.policy_net.save_weights(
            "{}/{}_ppo_tf2_p".format(save_path, "Autonomous_Vehicle"))
        self.value_net.save_weights(
            "{}/{}_ppo_tf2_v".format(save_path, "Autonomous_Vehicle"))
