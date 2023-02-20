#! /usr/bin/env python3

from AutonomousVehicle.EnvironmentClass import Environment
from PPO.PPO import PPO
import rospy
import tensorflow as tf
import datetime


def main_talker():

    env = Environment()
    env.environment_init()

    # To Use older parameters change string of OldModelPath to the desired
    # To create new set of parameters change Old ModelPath to the desired Path
    # Paths Relative to "Data/"
    # OldModelPath = "Data/PPO/2022-09-28 15:50:48.742388/ModelData"
    OldModelPath = "Data/PPO/Autonomous_VehicleModelAlpha_4.1/ModelData"

    ppo = PPO(env_id=env,
        render=False,
              num_process=1,
              model_path=OldModelPath
              )

    i_iter = 1
    max_eval_iter = 5
    Reward_sum = 0
    Reward_record = -100



    for eval in range(max_eval_iter):

        print("Eval: ", eval)
        Reward = ppo.eval(i_iter, render=True)
        print("Reward: ",Reward)

    print("------------------------")


if __name__ == '__main__':

    try:
        main_talker()
    except rospy.ROSInterruptException:
        pass
