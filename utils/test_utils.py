import pickle
import random
import numpy as np
from matplotlib import pyplot as plt

from datas.DataLoder import MyDataSet
from datas.padsequence import PadSequence
from models.lstm_dw import *


def test_model(test_data_path, model_path, num_features):
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

    plt.xlabel('Day')
    plt.ylabel('Reward')
    plt.title('True and Predicted Rewards')
    plt.savefig(f'result/img/test_lstm_fixed.png')
    plt.legend()
    plt.show()
