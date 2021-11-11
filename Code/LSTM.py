import torch
from torch import nn


# 自实现LSTM结构
class LSTM(nn.Module):
    # 输入维度、输出维度
    def __init__(self, input_size, hidden_size, *, num_layers=1):
        super(LSTM, self).__init__()
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

        outputs = torch.zeros((n_step, batch_size, hidden_size)).to(device)  # 最后一层所有结点的输出
        h_n = torch.zeros((num_layers, batch_size, hidden_size)).to(device)  # 每一层最后一个结点的输出
        c_n = torch.zeros((num_layers, batch_size, hidden_size)).to(device)  # 每一层最后一个结点的记忆
        # # 但nn中的LSTM会分别使用X、state的设备，不进行统一
        # hidden_state = state[0]
        # cell_state = state[1]
        # h_n = hidden_state.clone()
        # c_n = cell_state.clone()

        # 第layer层
        for layer in range(num_layers):
            hidden_state_layer = hidden_state[layer]
            cell_state_layer = cell_state[layer]
            # 一层共n_step个结点（句子长度）
            for step in range(n_step):
                x_i = X[step]
                # 遗忘门（旧记忆的占比）
                f = self.sigmoid_f(self.linear_if(x_i) + self.linear_hf(hidden_state_layer))
                # 输入门（新的记忆）
                i = self.sigmoid_i(self.linear_ii(x_i) + self.linear_hi(hidden_state_layer))
                g = self.tanh_g(self.linear_ig(x_i) + self.linear_hg(hidden_state_layer))
                cell_state_layer = f * cell_state_layer + i * g
                # 输出门
                o = self.sigmoid_o(self.linear_io(x_i) + self.linear_ho(hidden_state_layer))
                hidden_state_layer = o * self.tanh_o(cell_state_layer)

                # 在最后一层时，需要将所有n_step个结点，值保存在outputs中
                if layer is num_layers - 1:
                    outputs[step] = hidden_state_layer
            # 在每一层的最后一个结点，值保存在outputs_h、outputs_c中
            h_n[layer] = hidden_state_layer
            c_n[layer] = cell_state_layer
        return outputs, (h_n, c_n)
