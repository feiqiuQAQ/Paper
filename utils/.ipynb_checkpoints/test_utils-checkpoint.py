import pickle
import random
import numpy as np
from matplotlib import pyplot as plt

from datas.DataLoder import MyDataSet
from datas.padsequence import PadSequence
from models.lstm_dw import *
from utils.mkdir_utils import *


def test_model(test_data_path, model_path, num_features, save_test_path):
    pkl_file = open(test_data_path, 'rb')
    test_batch_data = pickle.load(pkl_file)
    pkl_file.close()
    pkl_file.close()
    test_data = MyDataSet(test_batch_data, num_features=num_features)
    test_loader = DataLoader(dataset=test_data,
                             batch_size=16,
                             shuffle=True,
                             collate_fn=PadSequence(),
                             )

    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()  # 设置模型为评估模式
    print(model)
    # 初始化 h_state
    num_layers = 1
    hidden_size = 50
    h_state = (torch.zeros(num_layers, 1, hidden_size), torch.zeros(num_layers, 1, hidden_size))

    true_rewards = []
    pred_rewards = []
    mse_loss = 0

    with torch.no_grad():
        for step, (b_x, lengths, b_y) in enumerate(test_loader):
            b_x, b_y = b_x, b_y
            state_prediction, reward_prediction, h_state = model(b_x, lengths, h_state)

            true_reward = b_y[:, :, -1].squeeze()
            pred_reward = reward_prediction.squeeze()

            true_rewards.append(true_reward.numpy())
            pred_rewards.append(pred_reward.numpy())

            mse_loss += torch.mean((true_reward - pred_reward) ** 2)

    mse_loss /= len(test_loader)
    print("Mean Squared Error Loss:", mse_loss.item())

    plt.plot(list(true_rewards[0]), label='True Rewards')
    plt.plot(list(pred_rewards[0]), label='Predicted Rewards')
    save_test_path = mkdir(save_test_path)
    plt.xlabel('Day')
    plt.ylabel('Reward')
    plt.title('True and Predicted Rewards')
    plt.savefig(f'{save_test_path}/test_lstm_fixed.png')
    plt.legend()
    plt.show()


def record_behavior_all(dqn, init_bundle, action_mean, action_std, device, simulator,  save_img_path, days=24):
    # 使用所有个体的状态
    selected_states = torch.FloatTensor(init_bundle).to(device)

    h_state = None
    # 记录每天所有个体的行为
    all_actions = torch.zeros((init_bundle.shape[0], days), dtype=torch.float32, device=device)

    for day in range(days):
        actions = dqn.choose_action(selected_states)
        all_actions[:, day] = actions

        actions = torch.tensor(actions, dtype=torch.float32).to(device)

        a_for_simulator = torch.unsqueeze((actions - action_mean) / action_std, dim=1)
        input_tensor = torch.cat((selected_states, a_for_simulator), dim=1)

        length = torch.ones(input_tensor.size(0), dtype=torch.long, device=device)
        input_tensor = torch.unsqueeze(input_tensor, dim=1)  # 添加批次维度
        length = length.to(device)

        pred_s, _, h_state = simulator(input_tensor, length, h_state)
        selected_states = pred_s.squeeze()

    # 计算每天的行为均值
    mean_actions = all_actions.mean(dim=0)
    save_img_path = mkdir(save_img_path)
    # 绘制整个群体的行为均值
    plt.figure(figsize=(12, 8))
    plt.plot(mean_actions.cpu().numpy(), label='Mean Action')

    plt.xlabel('Day')
    plt.ylabel('Action')
    plt.title('Mean Behavior Strategy of All Consumers Over 30 Days')
    plt.legend()
    plt.savefig(f'{save_img_path}/mean_behavior.png')
    plt.show()
