#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/19 下午2:47
import math
import multiprocessing
import time
#a
import numpy as np
import tensorflow as tf
import cv2
from PPO.Memory import Memory
from PPO.tf2_util import NDOUBLE, TDOUBLE


def collect_samples(
    pid, queue, env, policy, render, running_state, min_batch_size
):
    #print("Init Dic Mem")
    log = dict()
    memory = Memory()
    num_steps = 0
    num_episodes = 0

    counter_episodes = 0
    total_iter = 0
    mean_iter = 0
    


    min_episode_reward = float("inf")
    max_episode_reward = float("-inf")
    total_reward = 0

    # Cycle finishes when buffer (min_batch_size) is full
    while num_steps < min_batch_size:
        
        # Reset the environment and retrieve first state
        state = env.reset()
        episode_reward = 0

        # Apply zfilter to the state
        if running_state:
            state = running_state(state)

        # Max iterations in episode
        for t in range(10000):

            # Periodically data is diplayed to the user
            if not num_steps %500:

                print("Memory Collector Iter: ", num_steps)
                print("Episode Iteration Mean: ",mean_iter)

            # Make the state a tensor to input it in neural network
            state_tensor = tf.expand_dims(
                tf.convert_to_tensor(state, dtype=TDOUBLE), axis=0
            )

            # Process action and log_prob to store it
            action, log_prob = policy.get_action_log_prob(state_tensor)
            
            # Convert from tensor to numpy value
            action = action.numpy()[0]
            log_prob = log_prob.numpy()[0]
            env.rate.sleep()

            # Use the action returned to step in environment
            next_state, reward, done = env.step(action)
            episode_reward += reward

            # Aplly z filter
            if running_state:
                next_state = running_state(next_state)

            # Apply mask
            mask = 0 if done else 1
            
            # Store tuple 
            memory.push(state, action, reward, next_state, mask, log_prob)
            num_steps += 1

            # Break if terminal state or buffer is full
            if done or num_steps >= min_batch_size:
                break

            # Next state become the current state
            state = next_state

            final_iteration = t

        #num_steps += (t + 1)
        num_episodes += 1
        total_reward += episode_reward
        min_episode_reward = min(episode_reward, min_episode_reward)
        max_episode_reward = max(episode_reward, max_episode_reward)
        #print("Mean: ",mean_iter)
        counter_episodes += 1 
        total_iter += final_iteration

        mean_iter =  total_iter / counter_episodes   

    #print("Finished Memory Collector")

    log["num_steps"] = num_steps
    log["num_episodes"] = num_episodes
    log["total_reward"] = total_reward
    log["avg_reward"] = total_reward / num_episodes
    log["max_episode_reward"] = max_episode_reward
    log["min_episode_reward"] = min_episode_reward
    

    if queue is not None:
        queue.put([pid, memory, log])
    else:
        return memory, log


def merge_log(log_list):
    log = dict()
    log["total_reward"] = sum([x["total_reward"] for x in log_list])
    log["num_episodes"] = sum([x["num_episodes"] for x in log_list])
    log["num_steps"] = sum([x["num_steps"] for x in log_list])
    log["avg_reward"] = log["total_reward"] / log["num_episodes"]
    log["max_episode_reward"] = max(
        [x["max_episode_reward"] for x in log_list]
    )
    log["min_episode_reward"] = min(
        [x["min_episode_reward"] for x in log_list]
    )

    return log


class MemoryCollector:
    def __init__(
        self, env, policy, render=False, running_state=None, num_process=1
    ):
        self.env = env
        self.policy = policy
        self.running_state = running_state
        self.render = render
        self.num_process = num_process

    def collect_samples(self, min_batch_size):

        #print("Enter memory Collector")

        if self.num_process == 1:

            t_start = time.time()
            #print("Start Collect Sample")
            memory, log = collect_samples(1,None, self.env, self.policy, False, self.running_state, min_batch_size)
            t_end = time.time()
            log["sample_time"] = t_end - t_start
            return memory ,log
        else:
            
            t_start = time.time()
            process_batch_size = int(math.floor(min_batch_size / self.num_process))
            # Init FIFO to store Processes Data
            queue = multiprocessing.Queue()
            # Init Workers List
            workers = []

            # don't render other parallel processes
            #print("Start Workers Init")
            for i in range(self.num_process - 1):
                worker_args = (
                    i + 1,
                    queue,
                    self.env,
                    self.policy,
                    False,
                    self.running_state,
                    process_batch_size,
                )


                # Set Workers in List
                workers.append(
                    multiprocessing.Process(
                        target=collect_samples, args=worker_args
                    )
                )
            #print("Exit Workers Init")

            # Start All Multithreading Workers
            #print("Start Workers")
            for worker in workers:
                worker.start()
            #print("Exit Workers")

            # Start Collecting samples with the script 
            # With Paralel Threads Running
            #print("Start  Workers Sample")
            memory, log = collect_samples(
                0,
                None,
                self.env,
                self.policy,
                self.render,
                self.running_state,
                process_batch_size,
            )

            # Init None Arrays to fill with FIFO Data 
            # From all Workers Experiences 
            worker_logs = [None] * len(workers)
            worker_memories = [None] * len(workers)
            #print("Exit Workers Sample")

            # Get Data From Workers FIFO
            #print("Start Get Data Workers")
            for _ in workers:
                pid, worker_memory, worker_log = queue.get()
                worker_memories[pid - 1] = worker_memory
                worker_logs[pid - 1] = worker_log
            #print("Exit Get Data Workers")

            # concat all memories
            #print("Start Concat Memories")
            for worker_memory in worker_memories:
                memory.append(worker_memory)
            #print("Exit Concat Memories")

            # Merge The data From Workers With Data from Script
            #print("Start Merge")
            if self.num_process > 1:
                log_list = [log] + worker_logs
                log = merge_log(log_list)

            #print("Stop Merge")
            t_end = time.time()
            log["sample_time"] = t_end - t_start

            #print("Exit memory Collector")

        return memory, log
