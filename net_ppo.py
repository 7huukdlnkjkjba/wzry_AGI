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
        
        # 状态向量编码
        # 状态向量包含: hp, mp, skill_cd(3), summoner_cd(2), item_cd, level, gold, minimap_features(10)
        self.state_dim = 20
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # 融合图像特征和状态特征
        self.fusion_dim = conv_output_size + 128
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # LSTM层
        self.lstm_input_size = 512
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
    
    def _encode_state_vector(self, state_info):
        """
        编码游戏状态向量
        
        Args:
            state_info: 游戏状态字典，包含:
                - hp_percent: 血量百分比 (0-1)
                - mp_percent: 蓝量百分比 (0-1)
                - skill_cd: 技能冷却字典 {1: cd1, 2: cd2, 3: cd3}
                - summoner_cd: 召唤师技能冷却 {'skill_1': cd1, 'skill_2': cd2}
                - item_cd: 装备技能冷却
                - level: 等级
                - gold: 金币
                - minimap: 小地图信息 {'allies': [], 'enemies': []}
        
        Returns:
            state_vector: 编码后的状态向量 (batch_size, state_dim)
        """
        batch_size = 1
        
        # 基础状态
        hp = state_info.get('hp_percent', 1.0)
        mp = state_info.get('mp_percent', 1.0)
        level = state_info.get('level', 1.0) / 15.0  # 归一化
        gold = state_info.get('gold', 0.0) / 20000.0  # 归一化
        item_cd = state_info.get('item_cd', 0.0) / 60.0  # 归一化
        
        # 技能冷却
        skill_cds = state_info.get('skill_cd', {1: 0, 2: 0, 3: 0})
        skill1_cd = skill_cds.get(1, 0.0) / 60.0  # 归一化
        skill2_cd = skill_cds.get(2, 0.0) / 60.0
        skill3_cd = skill_cds.get(3, 0.0) / 60.0
        
        # 召唤师技能冷却
        summoner_cds = state_info.get('summoner_cd', {'skill_1': 0, 'skill_2': 0})
        summoner1_cd = summoner_cds.get('skill_1', 0.0) / 300.0  # 归一化
        summoner2_cd = summoner_cds.get('skill_2', 0.0) / 300.0
        
        # 小地图信息
        minimap = state_info.get('minimap', {'allies': [], 'enemies': []})
        allies = minimap.get('allies', [])
        enemies = minimap.get('enemies', [])
        
        # 小地图特征：友方数量、敌方数量、最近友方距离、最近敌方距离等
        num_allies = min(len(allies), 5) / 5.0
        num_enemies = min(len(enemies), 5) / 5.0
        avg_enemy_dist = 0.5  # 简化处理
        avg_ally_dist = 0.5
        
        # 构建状态向量 (20维)
        state_vector = torch.tensor([
            hp, mp, level, gold, item_cd,
            skill1_cd, skill2_cd, skill3_cd,
            summoner1_cd, summoner2_cd,
            num_allies, num_enemies,
            avg_ally_dist, avg_enemy_dist,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # 预留维度
        ], dtype=torch.float32)
        
        # 扩展为batch维度
        state_vector = state_vector.unsqueeze(0).repeat(batch_size, 1)
        
        return state_vector
    
    def forward(self, x, hidden_state=None, state_info=None):
        """
        前向传播
        
        Args:
            x: 图像输入 (batch, 3, H, W)
            hidden_state: LSTM隐藏状态
            state_info: 游戏状态字典
        
        Returns:
            (logits_list), value, new_hidden
        """
        x = x.to(next(self.parameters()).device)
        
        # 卷积层处理图像
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        image_features = x.view(x.size(0), -1)
        
        # 编码状态向量
        if state_info is not None:
            state_vector = self._encode_state_vector(state_info)
            state_vector = state_vector.to(next(self.parameters()).device)
            state_features = self.state_encoder(state_vector)
        else:
            # 如果没有状态信息，使用零向量
            batch_size = x.size(0)
            state_features = torch.zeros(batch_size, 128).to(next(self.parameters()).device)
        
        # 融合图像特征和状态特征
        fused_features = torch.cat([image_features, state_features], dim=1)
        x = self.fusion_layer(fused_features)
        
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