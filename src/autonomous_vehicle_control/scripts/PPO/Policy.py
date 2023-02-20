import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow_probability.python.distributions import Normal


class Policy(tf.keras.Model):
    def __init__(
        self,
        dim_state,
        dim_action,
        dim_hidden=300,
        activation=tf.nn.relu,
        log_std=1,
    ):
        super(Policy, self).__init__(dim_state, dim_action, dim_hidden)

        # Store parameters of the neural network
        self.dim_action = dim_action
        self.dim_state = dim_state
        self.dim_hidden = dim_hidden

        # Define the neural network hidden and output layer    
        self.policy = tf.keras.models.Sequential(
            [
                layers.Dense(self.dim_hidden, activation=activation),
                layers.Dense(self.dim_hidden, activation=activation),
                layers.Dense(self.dim_action, activation=tf.nn.sigmoid),
            ]
        )

        # Define the shape of the input layer
        self.policy.build(input_shape=(None, self.dim_state))

        # Define the standard deviation variable 
        self.log_std = tf.Variable(
            name="action_log_std",
            initial_value=tf.zeros((1, dim_action), dtype=tf.float64) * log_std,
            trainable=True,
        )

    @tf.function
    def _get_dist(self, states):

        # Get the output of the neural network
        mean = self.policy(states)
        mean = tf.cast(mean,tf.float64)

        # Creates an array of deviation
        # For each action
        log_std = tf.ones_like(mean) * self.log_std

        # The deviation is now converted 
        # Converted from logarithic to normal
        std = tf.exp(log_std)
        return mean, std

    @tf.function
    def call(self, states, **kwargs):
        
        # Gets mean from NN and std value
        mean, std = self._get_dist(states)

        # Creates distributions to sample a action
        # With the values retrieved 
        dist = Normal(mean, std)

        # Sample a value from the normal distribution
        action = dist.sample()

        # Sums all the logarithmic probabilities of all the taken actions 
        log_prob = tf.reduce_sum(dist.log_prob(action), -1)

        # Return action and
        # Logarithmic probability of executing that action
        return action, log_prob

    @tf.function
    def get_action(self,states):
        actions = self.policy(states)
        actions = tf.cast(actions,tf.float64)
        return actions

    @tf.function
    def get_action_log_prob(self, states):
        return self.call(states)

    @tf.function
    def get_values(self,states):

        return self.policy(states), self.log_std

    @tf.function
    def get_action_log_prob(self, states):
        return self.call(states)

    @tf.function
    def get_log_prob(self, states, actions):
        mean, std = self._get_dist(states)
        dist = Normal(mean, std)
        #tf.print("LogProb: ",dist.log_prob(actions))
        log_prob = tf.reduce_sum(dist.log_prob(actions), -1)
        #tf.print(log_prob)
        return log_prob

    @tf.function
    def get_entropy(self, states):
        mean, std = self._get_dist(states)
        dist = Normal(mean, std)
        return tf.reduce_sum(dist.entropy(), -1)

