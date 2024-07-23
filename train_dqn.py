import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import math
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from argparse import ArgumentParser
from collections import Counter
from DoubleDueling_dqn import *
from rnn import *
from lstm import *
from datetime import datetime
from Alstm import *
from dynamicLstm import *


# 设置随机种子
seed = 15
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# # 定义超参数
epsilon = 1.0  # 初始探索因子
epsilon_min = 0.01  # 最小探索因子
epsilon_decay = 0.99
# epsilon_decay = (epsilon - epsilon_min) / 5000  # 线性递减值
# 折扣因子
gamma = 0.99
# 学习率
lr = 0.01
# 状态维度
n_states = 11
# 动作维度
n_actions = 4
# 目标网络更迭步数
TRARGE_REPLACE_INTER = 10
# 经验回放池大小
memory_capacity = 50000
# 迭代次数
epochs = 2000
# 采样大小
sampling_size = 100
# 批量大小
batch_size = 1000
# 一轮游戏长度
episode_length = 24
# strategy
strategy_list = ["ucb", "boltzmann", "entropy", "noise","epsilon"]
strategy = strategy_list[3]

data_time = "616"
# 消费6行为模拟器
# simulator_model = f"../model4/best_model_LSTM_{data_time}.pth"
simulator_model = f"../modelDynamic/best_model_dwa.pth"
# 数据地址
data_path = f"../data/DQN_data_{data_time}.csv"
# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模拟器
# simulator = torch.load(simulator_model)
simulator = torch.load(simulator_model, map_location=torch.device('cpu'))
simulator.to(device)

# 加载数据
data = pd.read_csv(data_path)
# 使用用户ID进行分组，对每个用户的数据取第一行（即最早的数据）作为代表。获取每个用户的初始状态。
init = data.groupby("顾客ID").first()
init = init.reset_index()

# 标准化
data_array = data.values
data_mean = data_array.mean(axis=0)
data_std = data_array.std(axis=0)
data_mean = data_mean[1:]
data_std = data_std[1:]
data_mean = np.array([0] + list(data_mean))
data_std = np.array([1] + list(data_std))
normalized = (init.values - data_mean) * 1.0 / data_std

action_mean = data_mean[n_states + 1]
action_std = data_std[n_states + 1]

reward_mean = data_mean[-1]
reward_std = data_std[-1]
print(action_mean)
print(action_std)
print(reward_mean)
print(reward_std)
# 提取初始状态
init_bundle = normalized[:, 1:n_states + 1]

print(simulator)
# 随机采样
def inital_state_random_sample():
    index = np.random.choice(init_bundle.shape[0] - 1, sampling_size, replace=False)
    return torch.FloatTensor(init_bundle[index]).to(device)


# 创建DQN模型
dqn = DoubleDuelingDQN(memory_capacity=memory_capacity, TRARGE_REPLACE_INTER=TRARGE_REPLACE_INTER,
                       batch_size=batch_size, n_states=n_states, n_actions=n_actions,
                       lr=lr, device=device, epsilon=epsilon, epsilon_min=epsilon_min,
                       epsilon_decay=epsilon_decay, gamma=gamma)

# 存放奖励和动作均值
score = []
mean_actions = []

for i_episode in range(epochs):
    # 初始化平均奖励
    ave_ep_score = 0
    # 采样初始状态
    init_state = inital_state_random_sample()
    s = init_state  # shape(100,9)
    # 初始化分数矩阵
    ep_score = torch.zeros(init_state.shape[0], device=device)
    # 初始化隐藏状态
    h_state = None
    # 初始化动作记录
    ep_actions = []

    # 开始迭代
    for i in range(episode_length):
        a = dqn.choose_action(s, strategy)
        a = torch.tensor(a, dtype=torch.float32).to(device)

        # 记录动作
        ep_actions.extend(a.cpu().numpy())

        a_for_simulator = torch.unsqueeze(torch.squeeze((a - action_mean) / action_std), dim=1)
        input_tensor = torch.unsqueeze(torch.cat((s, a_for_simulator), dim=1), dim=1)
        length = torch.ones(s.size(0), device=device)
        length = torch.ones(s.size(0), device=device).cpu().long()  # 转换为 CPU 张量

        pred_s, pre_r, h_state = simulator(input_tensor, length, h_state)

        s_ = pred_s.data.squeeze()
        r = pre_r.data
        r = r * reward_std + reward_mean

        # a 100,1  r 100,1  s 100,9  s_ 100,9
        dqn.store_transition(s, a.unsqueeze(1), r.squeeze(1), s_)

        ep_score += r.squeeze()
        if dqn.memory_counter > memory_capacity:
            dqn.learn()

        s = s_

    ave_ep_score = ep_score.mean().item()/100
    print(i_episode, ave_ep_score, np.mean(ep_actions))
    score.append(ave_ep_score)
    mean_actions.append(np.mean(ep_actions))


# 计算移动平均和置信区间
window_size = 50

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def standard_error(data, window_size):
    return np.std(data) / np.sqrt(window_size)

smoothed_losses = moving_average(score, window_size)
std_err = standard_error(score, window_size)
confidence_interval = 1.96 * std_err

lower_bound = smoothed_losses - confidence_interval
upper_bound = smoothed_losses + confidence_interval


# 获取当前日期和时间
now = datetime.now()

# 格式化当前日期和时间
formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")

# 绘制图形
x = np.arange(window_size - 1, len(score))

plt.figure(figsize=(12, 6))
plt.plot(score, alpha=0.3, label='Original Score')
plt.plot(x, smoothed_losses, label='Smoothed Score', color='red')
plt.fill_between(x, lower_bound, upper_bound, color='red', alpha=0.2, label='95% Confidence Interval')
plt.xlabel('Epoch')
plt.ylabel('Average Score')
plt.legend()
plt.savefig(f'../model_711/img/LSTM_{data_time}_{formatted_now}_{strategy}_1.png')
plt.show()

# 绘制奖励图像和动作均值图像
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(score)
plt.ylabel('Average Score')
plt.xlabel('Episode')
plt.title('Average Score Over Episodes')

plt.subplot(1, 2, 2)
plt.plot(mean_actions)
plt.ylabel('Mean Action')
plt.xlabel('Episode')
plt.title('Mean Action Over Episodes')

plt.tight_layout()
plt.savefig(f'../model_711/img/LSTM_{data_time}_{formatted_now}_{strategy}_2.png')
plt.show()

# 在最后一个epoch记录3个消费者每一天的行为策略
num_consumers = 3
days = 30
final_actions = np.zeros((num_consumers, days))

# 随机选择3个消费者
consumer_indices = np.random.choice(init_bundle.shape[0], num_consumers, replace=False)
selected_states = torch.FloatTensor(init_bundle[consumer_indices]).to(device)

# 初始化隐藏状态
h_state = None

# 创建一个空的 final_actions 张量，用于存储每个消费者在每天的行动
final_actions = torch.zeros((num_consumers, days), dtype=torch.float32, device=device)

for day in range(days):
    actions = dqn.choose_action(selected_states)
    final_actions[:, day] = actions
    actions = torch.tensor(actions, dtype=torch.float32).to(device)  # 移动到设备上

    # 对 actions 使用 clone().detach() 替代 torch.tensor，以避免警告
    # actions = torch.tensor(actions, dtype=torch.float32).clone().detach().to(device)

    a_for_simulator = torch.unsqueeze(torch.squeeze((actions - action_mean) / action_std), dim=1)
    input_tensor = torch.unsqueeze(torch.cat((selected_states, a_for_simulator), dim=1), dim=1)

    # 确保 lengths 的大小与 input_tensor 的 batch_size 一致
    length = torch.ones(input_tensor.size(0), dtype=torch.long, device=device)

    # 将 input_tensor 和 length 转移到同一设备上
    input_tensor = input_tensor.to(device)
    length = length.to(device)

    # 使用模拟器进行预测
    pred_s, pre_r, h_state = simulator(input_tensor, length.cpu(), h_state)  # 将 lengths 张量移动到 CPU 上
    selected_states = pred_s.data.squeeze()

# 绘制3个消费者的行为策略折线图
plt.figure(figsize=(12, 8))
for i in range(num_consumers):
    plt.plot(final_actions[i].cpu().numpy(), label=f'Consumer {i+1}')  # 转移到 CPU 上绘制图像

plt.xlabel('Day')
plt.ylabel('Action')
plt.title('Behavior Strategy of 3 Consumers Over 30 Days')
plt.legend()
plt.savefig(f'../model_711/img/LSTM_{data_time}_{formatted_now}_{strategy}_3.png')
plt.show()

