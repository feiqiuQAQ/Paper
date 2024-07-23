import torch


class Config:
    # 输入特征数 包含action
    input_size = 9
    LR = 0.005
    batch_size = 64
    hidden_size = 50
    epochs = 1000
    training_data_path = "./data/train_new_723.pkl"
    save_path = "./result/models"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'fixed'
    # 状态数
    num_features = 8
    # 测试参数
    test_data_path = './data/test_new_723.pkl'
    model_path = './result/models/fixed_best.pth'
# --------------------------------------------------------------------------------------
    # dqn参数
    epsilon = 1
    epsilon_min = 0.01
    epsilon_decay = 0.99
    gamma = 0.99
    dqn_lr = 0.01
    n_states = 8
    n_actions = 4
    TRARGE_REPLACE_INTER = 10
    memory_capacity = 5000
    dqn_epochs = 2000
    sampling_size = 10
    episode_length = 10
    dqn_batch_size = 100
    # strategy
    strategy_list = ["ucb", "boltzmann", "entropy", "noise", "epsilon"]
    strategy = strategy_list[4]
    data_time = "723"
    # 消费行为模拟器
    simulator_model = f"./result/models/fixed_best.pth"

    # 数据地址
    data_path = f"./data/DQN_data_new_{data_time}.csv"

    # 加载模拟器
    simulator = torch.load(simulator_model, map_location=torch.device('cpu'))


