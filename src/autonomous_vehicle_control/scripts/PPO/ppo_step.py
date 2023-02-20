#!/usr/bin/env python
# Created at 2020/1/22
import tensorflow as tf


@tf.function
def ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, optim_value_iternum, states, actions,
             returns, advantages, old_log_probs, clip_epsilon, entropy_coeff=1e-3):
    
    # Update Critic

    # Create loss object
    critic_loss_fn = tf.keras.losses.MeanSquaredError()
    
    # Cycle gradient update optim_value_iternum times 
    for _ in range(optim_value_iternum):

        # Gradient tape stores the data needed to calculate gradient
        with tf.GradientTape() as tape:

            # Make value estimation for each state
            values_pred = value_net(states)

            # Calculate critic loss
            value_loss = critic_loss_fn(returns, y_pred=values_pred)

        # Get the gradients 
        grads = tape.gradient(value_loss, value_net.trainable_variables)

        # Apply gradients using 
        optimizer_value.apply_gradients(
            grads_and_vars=zip(grads, value_net.trainable_variables))

    # Update policy

    # Gradient tape stores the data needed to calculate gradient
    with tf.GradientTape() as tape:

        # Get logarithmic probabilities of executing each action
        log_probs = tf.expand_dims(policy_net.get_log_prob(states, actions), axis=-1)
        
        # Calculate the ratio
        ratio = tf.exp(log_probs - old_log_probs)
        
        # Calculate surrogate objective 1
        surr1 = ratio * advantages

        # Calculate surrogate objective 2
        surr2 = tf.clip_by_value(
            ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        
        # Calculate entropy
        entropy = tf.reduce_mean(policy_net.get_entropy(states))

        # Calculate the loss 
        policy_loss = - tf.reduce_mean(tf.minimum(surr1, surr2)) - entropy_coeff * entropy

    # Get the gradients
    grads = tape.gradient(policy_loss, policy_net.trainable_variables)
    
    # Apply gradients using optimizer
    optimizer_policy.apply_gradients(
        grads_and_vars=zip(grads, policy_net.trainable_variables))

    return {"ratio": ratio,
            "critic_loss": value_loss,
            "policy_loss": policy_loss,
            "policy_entropy": entropy
            }
