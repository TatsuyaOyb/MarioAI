"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import torch
from env import create_train_env
from model import PPO
import torch.nn.functional as F
from collections import deque
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

from IPython import display
import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

# from pyvirtualdisplay import Display
# _display = Display(visible=False, size=(600, 600))
# _ = _display.start()

import matplotlib.pyplot as plt
from matplotlib import animation

def display_color_movie(img_color,info):
    dpi = 72
    interval = 50 # ms

    plt.figure(figsize=(240/dpi,256/dpi),dpi=dpi)  # 修正
    patch = plt.imshow(img_color[0])
    plt.axis=('off')
    animate = lambda i: patch.set_data(img_color[i])
    ani = animation.FuncAnimation(plt.gcf(),animate,frames=len(img_color),interval=interval)
    #ani.save("./result.mp4", writer="ffmpeg", dpi=300)
    if info["flag_get"]:
        ani.save("./result_goal.gif", writer="pillow")
    else:
        ani.save("./result.gif", writer="pillow")
    #display.display(display.HTML(ani.to_jshtml()))
    plt.close()

def eval(opt, global_model, num_states, num_actions, last_episode):
    torch.manual_seed(123)
    if opt.action_type == "right":
        actions = RIGHT_ONLY
    elif opt.action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT
    env = create_train_env(opt.world, opt.stage, actions)
    # color env
    env_color = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(opt.world, opt.stage))
    env_color = JoypadSpace(env_color, actions)  # 行動空間を制限 0. 右に歩く、 1. 右方向にジャンプ

    local_model = PPO(num_states, num_actions)
    if torch.cuda.is_available():
        local_model.cuda()
    local_model.eval()
    state = torch.from_numpy(env.reset())
    if torch.cuda.is_available():
        state = state.cuda()
    done = True
    curr_step = 0
    actions = deque(maxlen=opt.max_actions)
    
    # display
    # prev_screen = env.render(mode='rgb_array')
    # plt.imshow(prev_screen)
    img_color=[]
    state_color = env_color.reset()
    while True:
        curr_step += 1
        if done:
            local_model.load_state_dict(global_model.state_dict())
        logits, value = local_model(state)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, done, info = env.step(action)

        # Uncomment following lines if you want to save model whenever level is completed
        if info["flag_get"]:
            print("GOAL!!!")
            torch.save(local_model.state_dict(),
                       "{}/smb_ppo_goal_{}_{}_{}".format(opt.saved_path, opt.world, opt.stage, curr_step+last_episode))

        #env.render()
        
        # 使えない場合
        #screen = env.render(mode='rgb_array')
        # screen = env.render(mode='rgb_array')
        # plt.imshow(screen)
        # ipythondisplay.clear_output(wait=True)
        # ipythondisplay.display(plt.gcf())
        
        #
        for i in range(4):
            next_state_color, _, done, _ = env_color.step(action)
            if done:
                break
        img_color.append(np.stack(state_color,axis=0))
        state_color = next_state_color

        actions.append(action)
        if curr_step > opt.num_global_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        if done:
            img_color.append(np.stack(state_color,axis=0))
            display_color_movie(img_color, info)
            img_color = []
            curr_step = 0
            actions.clear()
            state = env.reset()
            state_color = env_color.reset()
        state = torch.from_numpy(state)
        if torch.cuda.is_available():
            state = state.cuda()
    
