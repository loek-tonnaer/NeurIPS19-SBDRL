from env import Env
import numpy as np
from tqdm import trange
import time
import math
import pickle as pk
import random
import pymunk
from constants import *
import torch


print(pymunk.version)

agent_parameters = {
    'radius': 15,
    'speed': 10,
    'rotation_speed' : math.pi/8,
    'living_penalty': 0,
    'position': (30,30),
    'angle': 0,
    'sensors': [
      
        {
           'nameSensor' : 'proximity_test',
           'typeSensor': 'proximity',
           'fovResolution': 64,
           'fovRange': 300,
           'fovAngle': math.pi ,
           'bodyAnchor': 'body',
           'd_r': 0,
           'd_theta': 0,
           'd_relativeOrientation': 0,
           'display': False,
        }
        
       
    ],
    'actions': ['forward', 'turn_left', 'turn_right', 'left', 'right', 'backward'],
    'measurements': ['health', 'poisons', 'fruits'],
    'texture': {
        'type': 'color',
        'c': (255, 255, 255)
    },
    'normalize_measurements': False,
    'normalize_states': False,
    'normalize_rewards': False
}

env_parameters = {
    'map':False,
    'n_rooms': 2,
    'display': False,
    'horizon': 10001,
    'shape': (84, 84),
    'mode': 'time',
    'poisons': {
        'number': 0,
        'positions': 'random',
        'size': 10,
        'reward': -10,
        'respawn': True,
        'texture': {
            'type': 'color',
            'c': (255, 255, 255),
        }
    },
    'fruits': {
        'number': 0,
        'positions': 'random',
        'size': 10,
        'reward': 10,
        'respawn': True,
        'texture': {
            'type': 'color',
            'c': (255, 150, 0),
        }
    },
    'obstacles': [
       
    ],
    'walls_texture': {
        'type': 'color',
        'c': (1, 1, 1)
    },
    'agent': agent_parameters
}


env = Env(**env_parameters)
n = len(agent_parameters['actions'])
meas, sens = None, None
prev_sens = None

dataset = []

action = {}

longtrans = 0
lattrans = 0
rot = 0

action_save = []
start = time.time()
done = False
for i in trange(1):
    while not done:
        if sens is not None:
            prev_sens = sens.copy()
            
        if np.random.binomial(1, 0.5, size=None) == 0:

            action_binary = (2*np.random.binomial(1, 0.5, size=None) -1)
            action_tosave = action_binary+1

            #print('lateral', action_tosave)

            action['longitudinal_velocity'] = 0
            action['lateral_velocity'] = action_binary*0.5
            action['angular_velocity'] = 0

        else:

            action_binary = (2*np.random.binomial(1, 0.5, size=None) -1)
            action_tosave = action_binary+2

            #print('longitudinal', action_tosave)

            action['longitudinal_velocity'] = action_binary*0.5
            action['lateral_velocity'] = 0
            action['angular_velocity'] = 0

        action_save.append(action_tosave)

            
        sens, r, done, meas = env.step(action)

        
    env.reset()
    done = False
end = time.time()
print(end - start)


def translate(action):
    if action == 1:
        res = 'up'
    if action == -1:
        res = 'down'
    if action == 0:
        res = 'left'
    if action == 2:
        res = 'right'

    return res


from PIL import Image
import os
inputs = []
for i,elem in enumerate(os.listdir('images/')):
    im = np.asarray(Image.open('images/' + str(i+1)+'.png').convert('RGB'))
    inputs.append(im)
inputs = np.array(inputs).reshape(-1,64,64,3)
inputs = inputs.astype('float32') / 255.

action_save = action_save[1:]

np.save('inputs', inputs[:-1])
np.save('actions', np.array(action_save))







