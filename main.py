import pickle

from config import Config
from sklearn.model_selection import train_test_split
from datas.DataLoder import *
from datas.padsequence import *
from models.lstm_base import *
from models.lstm_dw import *
from models.lstm_dwa import *
from models.DoubleDueling_dqn import *
from models.DuelingNet import *
from utils.train_utils import *
from utils.test_utils import *
import argparse


def env_train(training_data_path, model_name, epochs, num_features, input_size, hidden_size, LR, batch_size):
    with open(training_data_path, 'rb') as pkl_file:
        train_batch_data = pickle.load(pkl_file)

    train_data, val_data = train_test_split(list(train_batch_data.items()), test_size=0.1, random_state=42)
    train_data = dict(train_data)
    val_data = dict(val_data)

    print(len(train_data))  # Training set size
    print(len(val_data))  # Validation set size

    train_ds = MyDataSet(train_data, num_features)
    val_ds = MyDataSet(val_data, num_features)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_loader = DataLoader(dataset=train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=PadSequence(),
                              num_workers=2)

    val_loader = DataLoader(dataset=val_ds,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=PadSequence(),
                            num_workers=2)

    model_fixed = MyLstmWithTaskAttention(input_size=input_size, hidden_size=hidden_size, num_features=num_features)
    model_dynamic = MyModelWithDynamicWeights(input_size=input_size, hidden_size=hidden_size, num_features=num_features)
    model_dwa = MyModelWithDWA(input_size=input_size, hidden_size=hidden_size, num_features=num_features)

    optimizer_fixed = torch.optim.Adam(model_fixed.parameters(), lr=LR)
    optimizer_dynamic = torch.optim.Adam(model_dynamic.parameters(), lr=LR)
    optimizer_dwa = torch.optim.Adam(model_dwa.parameters(), lr=LR)

    loss_func = MaskedMSE()

    if model_name == 'fixed':

        train(epochs=epochs, model=model_fixed, model_name=model_name, train_loader=train_loader, val_loader=val_loader,
              loss_func=loss_func,
              optimizer=optimizer_fixed, device=device, num_features=num_features)
    elif model_name == 'dw':
        train(epochs=epochs, model=model_dynamic, model_name=model_name, train_loader=train_loader,
              val_loader=val_loader,
              loss_func=loss_func,
              optimizer=optimizer_dynamic, device=device, num_features=num_features)
    else:
        train(epochs=epochs, model=model_dwa, model_name=model_name, train_loader=train_loader,
              val_loader=val_loader,
              loss_func=loss_func,
              optimizer=optimizer_dwa, device=device, num_features=num_features)


def dqn_train(memory_capacity, TRARGE_REPLACE_INTER, dqn_batch_size,
              n_states, n_actions, dqn_lr, device, epsilon, epsilon_min,
              epsilon_decay, gamma, save_img_path):
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

    # 加载数据
    init_bundle, action_mean, action_std, reward_mean, reward_std = load_and_preprocess_data(data_path, n_states)

    # 加载模拟器
    simulator.to(device)

    # 创建DQN模型
    dqn = DoubleDuelingDQN(
        memory_capacity=memory_capacity, TRARGE_REPLACE_INTER=TRARGE_REPLACE_INTER, batch_size=dqn_batch_size,
        n_states=n_states, n_actions=n_actions, lr=dqn_lr, device=device, epsilon=epsilon,
        epsilon_min=epsilon_min, epsilon_decay=epsilon_decay, gamma=gamma
    )

    print(reward_mean)
    # 训练模型
    score, mean_actions = train_model(
        dqn, simulator, init_bundle, action_mean, action_std, reward_mean, reward_std,
        epochs=epochs, episode_length=episode_length, sampling_size=sampling_size, device=device, strategy=strategy
    )

    # 绘制结果
    plot_dqn(score, mean_actions, strategy, save_img_path=save_img_path)

    # 记录行为策略
    record_behavior_all(dqn, init_bundle, action_mean, action_std, device, simulator=simulator,
                        save_img_path=save_img_path)


if __name__ == '__main__':

    conf = Config()
    # env_train 参数
    batch_size = conf.batch_size
    input_size = conf.input_size
    hidden_size = conf.hidden_size
    LR = conf.LR
    epochs = conf.epochs
    num_features = conf.num_features
    device = conf.device
    test_data_path = conf.test_data_path
    model_path = conf.model_path
    training_data_path = conf.training_data_path
    save_test_path = conf.save_test_path
    # env_train(training_data_path=training_data_path, model_name=model_name, epochs=epochs, num_features=num_features, input_size=input_size, hidden_size=hidden_size, LR=LR, batch_size=batch_size)

    # ----------------------------------------------------------------------------------------
    # dqn_train 参数
    epsilon = conf.epsilon
    epsilon_min = conf.epsilon_min
    epsilon_decay = conf.epsilon_decay
    gamma = conf.gamma
    dqn_lr = conf.dqn_lr
    n_states = conf.n_states
    n_actions = conf.n_actions
    # 目标网络更迭步数
    TRARGE_REPLACE_INTER = conf.TRARGE_REPLACE_INTER

    memory_capacity = conf.memory_capacity
    dqn_epochs = conf.dqn_epochs
    sampling_size = conf.sampling_size
    episode_length = conf.episode_length

    dqn_batch_size = conf.dqn_batch_size

    strategy = conf.strategy

    data_path = conf.data_path
    save_img_path = conf.save_img_path

    simulator = conf.simulator


    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='这是一个示例程序，用于展示如何接收命令行参数。')

    # 添加参数
    parser.add_argument('--w', type=str, help='输入训练类型')
    parser.add_argument('--t', type=str, default='train', help='这是一个可选参数，默认值为 default_value')
    parser.add_argument('--m', type=str, default='fixed', help='这是一个可选参数，默认值为 default_value')

    # 解析命令行参数
    args = parser.parse_args()
    model_name = args.m

    if args.w == "env":
        if args.t == "train":
            print(f'开始训练环境模拟器，模型为{model_name}')
            print('-------------------------------------------------------------------------------------')
            env_train(training_data_path=training_data_path, model_name=model_name, epochs=epochs,
                      num_features=num_features, input_size=input_size, hidden_size=hidden_size,
                      LR=LR, batch_size=batch_size)
        else:
            print('开始测试环境模拟器')
            test_model(test_data_path=test_data_path, model_path=model_path, num_features=num_features,
                       save_test_path=save_test_path)
    else:
        print('开始训练DQN')
        dqn_train(memory_capacity=memory_capacity, TRARGE_REPLACE_INTER=TRARGE_REPLACE_INTER,
                  dqn_batch_size=dqn_batch_size, n_states=n_states, n_actions=n_actions,
                  dqn_lr=dqn_lr, device=device, epsilon=epsilon, epsilon_min=epsilon_min,
                  epsilon_decay=epsilon_decay, gamma=gamma, save_img_path=save_img_path)


