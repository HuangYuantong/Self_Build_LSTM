import torch
from torch import nn


# 自实现LSTM结构单层基础
class LSTM_Base(nn.Module):
    # 输入维度、输出维度
    def __init__(self, input_size, hidden_size):
        super(LSTM_Base, self).__init__()
        # 遗忘门（旧记忆的占比）
        # f
        self.linear_if = nn.Linear(input_size, hidden_size)
        self.linear_hf = nn.Linear(hidden_size, hidden_size)
        self.sigmoid_f = nn.Sigmoid()
        # 输入门（新的记忆）
        # i
        self.linear_ii = nn.Linear(input_size, hidden_size)
        self.linear_hi = nn.Linear(hidden_size, hidden_size)
        self.sigmoid_i = nn.Sigmoid()
        # g
        self.linear_ig = nn.Linear(input_size, hidden_size)
        self.linear_hg = nn.Linear(hidden_size, hidden_size)
        self.tanh_g = nn.Tanh()
        # 输出门（由新记忆构造输出）
        # o
        self.linear_io = nn.Linear(input_size, hidden_size)
        self.linear_ho = nn.Linear(hidden_size, hidden_size)
        self.sigmoid_o = nn.Sigmoid()
        self.tanh_o = nn.Tanh()
        # 需要存储一些网络规模有关数据
        self.hidden_size = hidden_size  # 中间变量维度（输出维度）

    def forward(self, X, state, n_step, batch_size, device):
        # 获取隐层结点初始值
        hidden_state = state[0]
        cell_state = state[1]
        outputs = torch.zeros((n_step, batch_size, self.hidden_size)).to(device)  # 最后一层所有结点的输出

        # 一层共n_step个结点（句子长度）
        for step in range(n_step):
            x_i = X[step]
            # 遗忘门（旧记忆的占比）
            f = self.sigmoid_f(self.linear_if(x_i) + self.linear_hf(hidden_state))
            # 输入门（新的记忆）
            i = self.sigmoid_i(self.linear_ii(x_i) + self.linear_hi(hidden_state))
            g = self.tanh_g(self.linear_ig(x_i) + self.linear_hg(hidden_state))
            cell_state = f * cell_state + i * g
            # 输出门
            o = self.sigmoid_o(self.linear_io(x_i) + self.linear_ho(hidden_state))
            hidden_state = o * self.tanh_o(cell_state)

            # 将所有n_step个结点，值保存在outputs中
            outputs[step] = hidden_state
        # 在每一层的最后一个结点，值保存在outputs_h、outputs_c中
        h_n = hidden_state
        c_n = cell_state
        return outputs, (h_n, c_n)


# 自实现双层LSTM结构（串联）
class LSTM(nn.Module):
    # 输入维度、输出维度
    def __init__(self, input_size, hidden_size, *, num_layers=1):
        super(LSTM, self).__init__()
        # 第一层
        self.LSTM_one = LSTM_Base(input_size=input_size, hidden_size=hidden_size)
        # 第二层（若有）
        if num_layers == 2:
            self.LSTM_two = LSTM_Base(input_size=hidden_size, hidden_size=hidden_size)
        # 需要存储一些网络规模有关数据
        self.num_layers = num_layers  # 网络层数
        self.hidden_size = hidden_size  # 中间变量维度（输出维度）

    def forward(self, X, state=None):
        # 根据输入X，确定所运行设备
        device = X.device
        # 获取并设置各维长度
        [n_step, batch_size, _] = X.shape
        num_layers = self.num_layers
        hidden_size = self.hidden_size
        # 若state未传入参数，则使用默认初始化
        if state is None:
            hidden_state = torch.zeros((num_layers, batch_size, hidden_size)).to(device)
            cell_state = torch.zeros((num_layers, batch_size, hidden_size)).to(device)
        else:
            hidden_state = state[0].to(device)
            cell_state = state[1].to(device)
        # 输出存储
        h_n = torch.zeros((num_layers, batch_size, hidden_size)).to(device)  # 每一层最后一个结点的输出
        c_n = torch.zeros((num_layers, batch_size, hidden_size)).to(device)  # 每一层最后一个结点的记忆

        # 第一层
        outputs1, (h_n1, c_n1) = self.LSTM_one(X, (hidden_state[0], cell_state[0]), n_step, batch_size, device)
        h_n[0], c_n[0] = h_n1, c_n1
        # 第二层
        if num_layers == 2:
            outputs2, (h_n2, c_n2) = self.LSTM_two(outputs1, (hidden_state[1], cell_state[1]), n_step, batch_size, device)
            h_n[1], c_n[1] = h_n2, c_n2
            outputs = outputs2
        else:
            outputs = outputs1
        return outputs, (h_n, c_n)
