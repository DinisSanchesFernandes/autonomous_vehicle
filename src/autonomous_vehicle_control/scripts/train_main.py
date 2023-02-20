#! /usr/bin/env python3

from AutonomousVehicle.EnvironmentClass import Environment
from PPO.PPO import PPO
import rospy
import tensorflow as tf
import datetime
import time

# Hyperparameters
#
#   Learning Rate
#       => Strenght of each gradientupdate step
#       => Should be increased if reward does not consistently increase
#       => Should be increased if learning is unstable
#       => 1e-5 <=> 1e-3
#
#   Clip Epsilon
#       => Clips the Advantage to limit the excecively optimistic 
#       => 0.2 <=> 0.3
#
#   Gamma (gamma)
#       => Is used in the Advantage calculos to consider future rewards
#       => The biggers the gamma the most important will the futures rewards be
#
#   Batch size (ppo_mini_batch_size)
#       => Data used for each gradient update
#       => 32 - 512 
#   
#   Buffer Size (min_batch_size)
#       => Data stored before the learning step 
#       => 2048 - 409600
#
#   Number of Epochs (ppo_epochs)
#       => Number of passes throught buffer during gradient descent
#       => 3 - 10
#



def main_talker():


    env = Environment()
    env.environment_init()

    '''
    tf.compat.v1.enable_eager_execution()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    '''
    # To Use older parameters change string of OldModelPath to the desired
    # To create new set of parameters change Old ModelPath to the desired Path
    # Paths Relative to "Data/"
    OldModelPath = None
    #OldModelPath = None
    now = datetime.datetime.now()
    NewModelStorePath = "Data/" + "PPO/" + str(now)
    
    if OldModelPath == None:
        ModelStorePath = NewModelStorePath
        old_model_path = None
    else:
        ModelStorePath = OldModelPath
        old_model_path = OldModelPath 
    writer = tf.summary.create_file_writer(ModelStorePath + "/TensorBoardData")


    ppo = PPO(env_id = env, 
        render=False, 
        num_process=1,
         
        lr_p = 1e-5, 
        lr_v = 1e-5, 
        gamma = 0.99, 
        tau = 0.95, 
        clip_epsilon = 0.2, 
        min_batch_size = 4096,
        ppo_mini_batch_size = 128, 
        ppo_epochs = 10, 
        model_path = old_model_path
        )

     
    max_iter = 1000
    eval_iter = 1
    save_iter = 1

    # Eval Param
    validation_final = 10


    for i_iter in range(1, max_iter + 1):
        
        print("------------------------")
        print("Iteration: ", i_iter)
        now = datetime.datetime.now()
        print("Time: ", str(now))
        print("------------------------")
        print("------------------------")
        print("Learn")
        ppo.learn(writer, i_iter)
        print("------------------------")

        

        if i_iter % eval_iter == 0:

            validation_step = 0

            r, eval_steps = ppo.eval(i_iter, render = True)
            print("Validation Step ",validation_step)
            print("Reward: ",r)
            print("Eval Steps: ", eval_steps)

            ''''
            print("------------------------")
            print("Eval")
            while validation_step < validation_final: 
                validation_step += 1
                r, eval_steps = ppo.eval(i_iter, render = True)
                print("Validation Step ",validation_step)
                print("Reward: ",r)
                print("Eval Steps: ", eval_steps)
                if r ==  -10:
                    validation_step -= 1
                elif eval_steps < 2000 or r < 500:
                    break

            print("------------------------")            
            '''

        if i_iter % save_iter == 0:

            print("------------------------")
            print("Save")
            ppo.save(ModelStorePath + "/ModelData")
            #ppo.save(ModelStorePath)
            print("------------------------")

        ''''
        if validation_step >= validation_final:
            print("Training Finished")
            break
        '''

if __name__ == '__main__':

    try:
        main_talker()
    except rospy.ROSInterruptException:
        pass
