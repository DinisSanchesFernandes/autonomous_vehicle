from sre_constants import SUCCESS
import gym
#import click
from matplotlib.cbook import ls_mapper
import tensorflow as tf
from MemoryCollector import MemoryCollector
from PPO import PPO
from Memory import Memory 
from Policy import Policy

#env = gym.make("Pendulum-v1")
#num_states = env.observation_space.shape[0]
#num_actions = 1
#s = env.reset()

#ppo = PPO("Pendulum-v1")








def main(env_id, render=False, num_process=1, lr_p=3e-4, lr_v=3e-4, gamma=0.99, tau=0.95, epsilon=0.2, batch_size= 1500,
         ppo_mini_batch_size = 250, ppo_epochs=10, max_iter=1000, eval_iter=10, save_iter=1, model_path="TrainedModels", log_path="TensorboarFiles", seed=1):

    AlgorithmType = "Hyperparameters6"
    loadModelFlag = False


    # Path To Store Tensorboard Data 
    # Path Name: <Algorithm>_<Environment>_<Specification>
    
    ModelStorePath = "Data/" + AlgorithmType 

    #base_dir = log_path + env_id + "/PPO_exp{}".format(seed)
    writer = tf.summary.create_file_writer(ModelStorePath + "/TensorBoardData")

    if loadModelFlag == True:
        model_path = "Data/" + AlgorithmType + "/ModelData"

    else:
        model_path = None

    ppo = PPO(env_id=env_id,
              render=render,
              num_process=num_process,
              min_batch_size=batch_size,
              lr_p=lr_p,
              lr_v=lr_v,
              gamma=gamma,
              tau=tau,
              clip_epsilon=epsilon,
              ppo_epochs=ppo_epochs,
              ppo_mini_batch_size=ppo_mini_batch_size,
              seed=seed,
              model_path = model_path,
              #model_path = None
              #log_path = StoreFile + log_file 
              )

    Success_Cnt = 0
    

    for i_iter in range(1, max_iter + 1):
    
        ppo.learn(writer, i_iter)


        

        if i_iter % eval_iter == 0:
            
            Reward = ppo.eval(i_iter, render=render)
            
            if(Reward >= -300):
                print("=========>Success: ",Success_Cnt)
                Success_Cnt += 1
            else:
                Success_Cnt += 0

            if Success_Cnt >= 10:
                print("------------------Finished------------------")
                break

        if i_iter % save_iter == 0:
            ppo.save(ModelStorePath + "/ModelData")




if __name__ == '__main__':
    main("Pendulum-v1")