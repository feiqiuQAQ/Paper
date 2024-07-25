import torch
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from utils.mkdir_utils import *


def train(epochs, model, train_loader, val_loader, loss_func, optimizer, model_name, num_features, device):
    model.to(device)
    train_loss_record = []
    val_loss_record = []
    h_state = None
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for step, (b_x, lengths, b_y) in enumerate(train_loader):
            b_x, b_y = b_x.to(device), b_y.to(device)
            state_prediction, reward_prediction, _ = model(b_x, lengths, h_state)

            state_mask = torch.zeros(state_prediction.size(), dtype=torch.bool, device=device)
            reward_mask = torch.zeros(reward_prediction.size(), dtype=torch.bool, device=device)

            for batch_id, item in enumerate(lengths):
                state_mask[batch_id][:item] = True
                reward_mask[batch_id][:item] = True

            state_loss = loss_func(state_prediction, b_y[:, :, : num_features], state_mask)
            reward_loss = loss_func(reward_prediction, b_y[:, :, -1].unsqueeze(2), reward_mask)

            if model_name == 'fixed':
                loss = 0.5 * state_loss + 0.5 * reward_loss
            elif model_name == 'dw':
                loss = model.loss(state_prediction, reward_prediction, b_y, state_mask, reward_mask)
            else:
                loss = model.loss(state_prediction, reward_prediction, b_y, state_mask, reward_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        train_loss_record.append(epoch_loss)

        print(f"Epoch: {epoch}, | Average train loss ({model_name}): {epoch_loss}")

        model.eval()
        val_epoch_loss = 0

        with torch.no_grad():
            for step, (b_x, lengths, b_y) in enumerate(val_loader):
                b_x, b_y = b_x.to(device), b_y.to(device)
                state_prediction, reward_prediction, _ = model(b_x, lengths, h_state)

                state_mask = torch.zeros(state_prediction.size(), dtype=torch.bool, device=device)
                reward_mask = torch.zeros(reward_prediction.size(), dtype=torch.bool, device=device)

                for batch_id, item in enumerate(lengths):
                    state_mask[batch_id][:item] = True
                    reward_mask[batch_id][:item] = True

                state_loss = loss_func(state_prediction, b_y[:, :, : num_features], state_mask)
                reward_loss = loss_func(reward_prediction, b_y[:, :, -1].unsqueeze(2), reward_mask)

                if model_name == 'fixed':
                    loss = 0.5 * state_loss + 0.5 * reward_loss
                elif model_name == 'dw':
                    loss = model.loss(state_prediction, reward_prediction, b_y, state_mask, reward_mask)
                else:
                    loss = model.loss(state_prediction, reward_prediction, b_y, state_mask, reward_mask)

                val_epoch_loss += loss.item()

        val_epoch_loss /= len(val_loader)
        val_loss_record.append(val_epoch_loss)

        print(f"Epoch: {epoch}, | Average validation loss ({model_name}): {val_epoch_loss}")

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model, f'result/models/{model_name}_best.pth')
            print(f"Saved best {model_name} model with validation loss: {best_val_loss}")

    torch.save(model, f'result/models/{model_name}_{epochs}.pth')

    plot(train_loss_record, val_loss_record, model_name)


def plot(train_loss_record, val_loss_record, model_name):
    # 设置seaborn风格
    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))  # 设置图形的尺寸
    plt.plot(train_loss_record, label=f'Training loss ({model_name})', color='blue', marker='o', linestyle='-')
    plt.plot(val_loss_record, label=f'Validation loss ({model_name})', color='red', marker='x', linestyle='--')

    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training and Validation Loss Over Epochs', fontsize=16)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)  # 添加网格线
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

    plt.savefig(f'./result/img/loss_{model_name}.png')
    plt.show()
    print('保存成功')


def load_and_preprocess_data(data_path, num_features):
    data = pd.read_csv(data_path)
    # init = data.groupby("顾客ID").first().reset_index()
    init = data.groupby("编号").first().reset_index()
    data_array = data.values
    data_mean = data_array.mean(axis=0)
    data_std = data_array.std(axis=0)

    data_mean = np.array([0] + list(data_mean[1:]))
    data_std = np.array([1] + list(data_std[1:]))

    normalized = (init.values - data_mean) / data_std
    action_mean = data_mean[len(data_mean) - 2]
    action_std = data_std[len(data_std) - 2]
    reward_mean = data_mean[-1]
    reward_std = data_std[-1]

    init_bundle = normalized[:, 1:num_features + 1]

    return init_bundle, action_mean, action_std, reward_mean, reward_std


def inital_state_random_sample(init_bundle, sampling_size, device):
    index = np.random.choice(init_bundle.shape[0], sampling_size, replace=False)
    return torch.FloatTensor(init_bundle[index]).to(device)


def train_model(dqn, simulator, init_bundle, action_mean, action_std, reward_mean, reward_std, epochs, episode_length,
                sampling_size, device, strategy):
    score = []
    mean_actions = []

    for i_episode in range(epochs):
        ave_ep_score = 0
        init_state = inital_state_random_sample(init_bundle, sampling_size, device)
        s = init_state
        ep_score = torch.zeros(init_state.shape[0], device=device)
        h_state = None
        ep_actions = []

        for i in range(episode_length):
            a = dqn.choose_action(s, strategy)
            a = torch.tensor(a, dtype=torch.float32).to(device)

            ep_actions.extend(a.cpu().numpy())

            a_for_simulator = torch.unsqueeze(torch.squeeze((a - action_mean) / action_std), dim=1)
            input_tensor = torch.unsqueeze(torch.cat((s, a_for_simulator), dim=1), dim=1)
            length = torch.ones(s.size(0), device=device).cpu().long()

            pred_s, pre_r, h_state = simulator(input_tensor, length, h_state)

            s_ = pred_s.data.squeeze()
            r = pre_r.data * reward_std + reward_mean

            dqn.store_transition(s, a.unsqueeze(1), r.squeeze(1), s_)

            ep_score += r.squeeze()
            if dqn.memory_counter > dqn.memory_capacity:
                dqn.learn()

            s = s_

        ave_ep_score = ep_score.mean().item() / 10
        print(i_episode, ave_ep_score, np.mean(ep_actions))
        score.append(ave_ep_score)
        mean_actions.append(np.mean(ep_actions))

    return score, mean_actions


def plot_dqn(score, mean_actions,  strategy, save_img_path):
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

    save_img_path = mkdir(save_img_path)

    plt.figure(figsize=(12, 6))
    x = np.arange(window_size - 1, len(score))
    plt.plot(score, alpha=0.3, label='Original Score')
    plt.plot(x, smoothed_losses, label='Smoothed Score', color='red')
    plt.fill_between(x, lower_bound, upper_bound, color='red', alpha=0.2, label='95% Confidence Interval')
    plt.xlabel('Epoch')
    plt.ylabel('Average Score')
    plt.legend()
    plt.savefig(f'{save_img_path}/loss.png')
    plt.show()

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
    plt.savefig(f'{save_img_path}/{strategy}_loss_and_action.png')
    plt.show()


def record_behavior(dqn, init_bundle, action_mean, action_std, device, simulator, num_consumers=3, days=7):
    consumer_indices = np.random.choice(init_bundle.shape[0], num_consumers, replace=False)
    selected_states = torch.FloatTensor(init_bundle[consumer_indices]).to(device)

    h_state = None
    final_actions = torch.zeros((num_consumers, days), dtype=torch.float32, device=device)

    for day in range(days):
        actions = dqn.choose_action(selected_states)
        final_actions[:, day] = actions
        actions = torch.tensor(actions, dtype=torch.float32).to(device)

        a_for_simulator = torch.unsqueeze(torch.squeeze((actions - action_mean) / action_std), dim=1)
        input_tensor = torch.unsqueeze(torch.cat((selected_states, a_for_simulator), dim=1), dim=1)

        length = torch.ones(input_tensor.size(0), dtype=torch.long, device=device)
        input_tensor = input_tensor.to(device)
        length = length.to(device)

        pred_s, _, h_state = simulator(input_tensor, length.cpu(), h_state)
        selected_states = pred_s.data.squeeze()

    plt.figure(figsize=(12, 8))
    for i in range(num_consumers):
        plt.plot(final_actions[i].cpu().numpy(), label=f'Consumer {i + 1}')

    plt.xlabel('Day')
    plt.ylabel('Action')
    plt.title('Behavior Strategy of 3 Consumers Over 30 Days')
    plt.legend()
    plt.savefig(f'result/img/3.png')
    plt.show()



