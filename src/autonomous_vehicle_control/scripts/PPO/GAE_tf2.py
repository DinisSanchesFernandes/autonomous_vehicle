#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/3 下午6:48
import numpy as np
import tensorflow as tf
from PPO.tf2_util import NDOUBLE, TDOUBLE


def estimate_advantages(rewards, masks, values, gamma, tau, eps=1e-8):
    batch_size = rewards.shape[0]
    deltas = np.zeros((batch_size,1), dtype=NDOUBLE)
    advantages = np.zeros((batch_size,1), dtype=NDOUBLE)

    # Set the auxiliar variables
    prev_value = 0
    prev_advantage = 0

    # Iterate the batch backawards
    for i in reversed(range(batch_size)):
        

        # Calculate delta
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]

        # Calculate advantage
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_value = values[i]
        prev_advantage = advantages[i]

    # The return is used to calculate the loss of the critic 
    returns = values + advantages
    
    # Normalize the advantage
    advantages = (advantages - advantages.mean()) / (advantages.std() + eps)

    # Return the advantage and the returns as tensors 
    return tf.convert_to_tensor(advantages, dtype=TDOUBLE), tf.convert_to_tensor(returns, dtype=TDOUBLE)


