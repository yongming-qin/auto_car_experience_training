from gym_torcs import TorcsEnv
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
from keras.engine.training import collect_trainable_weights
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit

import signal
import sys

PI= 3.14159265359

OU = OU()       #Ornstein-Uhlenbeck Process

class DriverExample(object):
    '''What the driver is intending to do (i.e. send to the server).
    Composes something like this for the server:
    (accel 1)(brake 0)(gear 1)(steer 0)(clutch 0)(focus 0)(meta 0) or
    (accel 1)(brake 0)(gear 1)(steer 0)(clutch 0)(focus -90 -45 0 45 90)(meta 0)'''
    def __init__(self):
        self.R = {'brake':0}

    def action(self, s_t, a_t):
        '''This is only an example. It will get around the track but the
        correct thing to do is write your own `drive()` function.'''
        target_speed=100
        # S: angle, track (19), trackPos, speedX, speedY, speedZ, wheelSpinVel/100.0 (4), rpm
        S = {}
            # value are processed in gym_torcs.py/make_observation while these are not processed
            #   in snakeoil3_gym.py. The controller we use is from snakeoil3_gym.py
            #   Thus, revert back.
        S['angle'] = s_t[0] * 3.1416
        S['trackPos'] = s_t[20]
        S['speedX'] = s_t[21] * 300.
        S['wheelSpinVel'] = s_t[24:28]

        self.R['accel'] = a_t[1]


        # Steer To Corner
        self.R['steer'] = S['angle']*10 / PI
        # Steer To Center
        self.R['steer'] -= S['trackPos']*.10

        # Throttle Control
        if S['speedX'] < target_speed - (self.R['steer']*50):
            self.R['accel'] += .01
        else:
            self.R['accel'] -= .01
        if S['speedX']<10:
            self.R['accel'] += 1/(S['speedX']+.1)

        # Traction Control System
        if ((S['wheelSpinVel'][2]+S['wheelSpinVel'][3]) -
            (S['wheelSpinVel'][0]+S['wheelSpinVel'][1]) > 5):
            self.R['accel']-= .2
        
        self.clip_to_limits() # get rid of absurd values

        #print("------------------------------------------")
        #print("angle: ", S['angle'], "speedX: ", S['speedX'], "trackPos: ", S['trackPos'])
        #print("steer: ", self.R['steer'], "accel: ", self.R['accel'], "brake: ", self.R['brake'])

        return [self.R['steer'], self.R['accel'], self.R['brake']]

    def clip(self,v,lo,hi):
        if v<lo: return lo
        elif v>hi: return hi
        else: return v
    
    def clip_to_limits(self): 
        self.R['steer']= self.clip(self.R['steer'], -1, 1)
        self.R['brake']= self.clip(self.R['brake'], 0, 1)
        self.R['accel']= self.clip(self.R['accel'], 0, 1)

def playGame(train_indicator=1):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 3  #Steering/Acceleration/Brake
    state_dim = 29  #of sensors input

    np.random.seed(1337)

    vision = False

    EXPLORE = 10000.
    INDEPENDENT = 3000.
    episode_count = 2000
    max_steps = 100000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    p = 0 # probability of choosing actor

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=True,gear_change=False)
    driver = DriverExample()

    #Now load the weight
    folder = "../pre/pre_aa"
    print("Now we load the weight")
    try:
        actor.model.load_weights(folder+"actormodel.h5")
        critic.model.load_weights(folder+"criticmodel.h5")
        actor.target_model.load_weights(folder+"actormodel.h5")
        critic.target_model.load_weights(folder+"criticmodel.h5")
        print("Weight load successfully")
    except:
        print("----------------------------Cannot find the weight-------------------------------")

    print("TORCS Experiment Start.")
    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
     
        total_reward = 0.
        for j in range(max_steps):
            loss = 0 
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim]) # steer, accel, brake
            noise_t = np.zeros([1,action_dim])
            
            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            
            # Regulate the rate of driver and learning of brake
            p += 1.0 / INDEPENDENT
            if p > 0.5:
                p = 0.5
            choice = np.random.binomial(1, p)

            #noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.60, 0.30)
            #noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.5 , 1.00, 0.10)
            #noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1 , 1.00, 0.05)

            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0, 0, 0.1) # steer
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  1 , 0.6, 0.10) # accel
            # noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], 0 , 0, 0.1) # brake

            #The following code do the stochastic brake
            if False:
                if p < 0.5: # after 1/2 INDEPENDENT steps, no expolration of brake
                    if random.random() <= 0.1:
                        print("***********************Now we apply the brake***************************")
                        noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  1.00 , -0.1, 0.05)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

            # choose between two results. From always using driver action to always using actor action 
            a_t_driver = driver.action(s_t, a_t[0])
            if choice == 0:
                a_t[0] = a_t_driver

            ob, r_t, done, info = env.step(a_t[0])
            
            s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        
            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer
            
            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])  
           
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
       
            if (train_indicator):
                loss += critic.model.train_on_batch([states,actions], y_t) 
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1
        
            print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)
        
            step += 1
            if done:
                break

        if np.mod(i, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("fly_actormodel.h5", overwrite=True)
                with open("fly_actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("fly_criticmodel.h5", overwrite=True)
                with open("fly_criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.end()  # This is for shutting down TORCS
    print("Finish.")

def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    # Generate a Torcs environment
    env = TorcsEnv(vision=False, throttle=True, gear_change=False)
    env.end()
    sys.exit(0)

if __name__ == "__main__":
    # if ctrl c is pressed, close env too
    signal.signal(signal.SIGINT, signal_handler)

    playGame()
