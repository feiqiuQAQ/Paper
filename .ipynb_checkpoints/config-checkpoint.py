import torch
from utils.mkdir_utils import *

def get_now():
    # 获取当前日期
    today = datetime.today()
    
    # 格式化为“年-月-日”格式
    formatted_date = today.strftime("%Y-%m-%d")
    return formatted_date

class Config:
    # 输入特征数 包含action
    input_size = 12
    LR = 0.005
    batch_size = 64
    hidden_size = 50
    epochs = 300
    training_data_path = "./data/train_year_814.pkl"
    
    save_path = mkdir("./result/models")
    
    save_test_path = "./result/img/test/year_814"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'dwa'
    # 状态数
    num_features = 11
    # 测试参数
    test_data_path = './data/test_year_814.pkl'
    
    model_path = f'{save_path}/{model_name}_best.pth'

    # --------------------------------------------------------------------------------------
    # dqn参数
    epsilon = 1
    epsilon_min = 0.01
    epsilon_decay = 0.99
    gamma = 0.99
    dqn_lr = 0.01
    n_states = 11
    n_actions = 5
    TRARGE_REPLACE_INTER = 10
    memory_capacity = 14000
    dqn_epochs = 3000
    sampling_size = 30
    episode_length = 24
    dqn_batch_size = 300
    # strategy
    strategy_list = ["ucb", "boltzmann", "entropy", "noise", "epsilon"]
    strategy = strategy_list[4]

    # 消费行为模拟器
    simulator_model = model_path

    # 结果图保存
    save_img_path = f"./result/img/train_dqn"

    # 数据地址
    data_path = f"./data/DQN_data_year_814.csv"

    # 加载模拟器
    # simulator = torch.load(simulator_model, map_location=torch.device('cpu'))


