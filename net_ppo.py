import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        
        # 图像特征提取 - 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # 计算卷积输出大小
        conv_output_size = self._get_conv_output_size(640, 640)
        
        # LSTM层
        self.lstm_input_size = conv_output_size
        self.lstm_hidden_size = 256
        self.lstm = nn.LSTM(self.lstm_input_size, self.lstm_hidden_size, batch_first=True)
        
        # Actor头部 - 动作输出
        self.fc_actor_move = nn.Linear(self.lstm_hidden_size, 2)  # 移动/不移动
        self.fc_actor_angle = nn.Linear(self.lstm_hidden_size, 360)  # 移动方向
        self.fc_actor_info = nn.Linear(self.lstm_hidden_size, 9)  # 信息动作
        self.fc_actor_attack = nn.Linear(self.lstm_hidden_size, 11)  # 攻击动作
        self.fc_actor_type = nn.Linear(self.lstm_hidden_size, 3)  # 动作类型
        self.fc_actor_arg1 = nn.Linear(self.lstm_hidden_size, 360)  # 参数1
        self.fc_actor_arg2 = nn.Linear(self.lstm_hidden_size, 100)  # 参数2
        self.fc_actor_arg3 = nn.Linear(self.lstm_hidden_size, 5)  # 参数3
        
        # Critic头部 - 价值输出
        self.fc_critic = nn.Linear(self.lstm_hidden_size, 1)
        
        # 初始化权重
        self._initialize_weights()
        
    def _get_conv_output_size(self, height, width):
        dummy_input = torch.zeros(1, 3, height, width)
        with torch.no_grad():
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        return x.view(x.size(0), -1).size(1)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, hidden_state=None):
        x = x.to(next(self.parameters()).device)
        
        # 卷积层处理图像
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        # LSTM处理序列
        if hidden_state is not None:
            x, new_hidden = self.lstm(x.unsqueeze(1), hidden_state)
        else:
            x, new_hidden = self.lstm(x.unsqueeze(1))
        x = x.squeeze(1)
        
        # Actor输出 - 动作概率分布
        move_logits = self.fc_actor_move(x)
        angle_logits = self.fc_actor_angle(x)
        info_logits = self.fc_actor_info(x)
        attack_logits = self.fc_actor_attack(x)
        type_logits = self.fc_actor_type(x)
        arg1_logits = self.fc_actor_arg1(x)
        arg2_logits = self.fc_actor_arg2(x)
        arg3_logits = self.fc_actor_arg3(x)
        
        # Critic输出 - 状态价值
        value = self.fc_critic(x)
        
        return (move_logits, angle_logits, info_logits, attack_logits, 
                type_logits, arg1_logits, arg2_logits, arg3_logits), value, new_hidden
    
    def get_action_probs(self, logits_list, actions):
        """计算给定动作的概率"""
        probs = []
        for i, logits in enumerate(logits_list):
            action = actions[:, i]
            prob = F.softmax(logits, dim=1).gather(1, action.unsqueeze(1))
            probs.append(prob)
        return torch.prod(torch.cat(probs, dim=1), dim=1)
    
    def get_log_probs(self, logits_list, actions):
        """计算给定动作的对数概率"""
        log_probs = []
        for i, logits in enumerate(logits_list):
            action = actions[:, i]
            log_prob = F.log_softmax(logits, dim=1).gather(1, action.unsqueeze(1))
            log_probs.append(log_prob)
        return torch.sum(torch.cat(log_probs, dim=1), dim=1)