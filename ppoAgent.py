import os

import cv2
import numpy as np
import torch
from torch import optim, nn
from torch.distributions import Categorical

from argparses import device, args, globalInfo
from net_ppo import ActorCritic
from action_mask import ActionMask
from feather_cnn import FeatherCNNManager


class PPOAgent:
    def __init__(self, use_feather_cnn=False):
        torch.backends.cudnn.enabled = False
        
        self.action_sizes = [2, 360, 9, 11, 3, 360, 100, 5]
        self.device = device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.epsilon = args.epsilon  # PPO clip参数
        self.dual_ppo_c = args.dual_ppo_c  # dual-PPO的c参数
        self.learning_rate = args.learning_rate
        self.clip_param = args.clip_param
        
        self.steps_done = 0
        self.ppo_epochs = args.ppo_epochs
        self.entropy_coef = args.entropy_coef
        self.value_loss_coef = args.value_loss_coef
        
        # 是否使用FeatherCNN加速推理
        self.use_feather_cnn = use_feather_cnn
        self.feather_manager = FeatherCNNManager.get_instance()
        
        # Actor-Critic网络
        self.policy_net = ActorCritic().to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # 旧策略网络用于计算重要性采样比率
        self.old_policy_net = ActorCritic().to(self.device)
        self.update_old_policy()
        
        # Action Mask机制
        self.action_mask = ActionMask()
        
        if args.model_path and os.path.exists(args.model_path):
            self.policy_net.load_state_dict(torch.load(args.model_path))
            self.old_policy_net.load_state_dict(self.policy_net.state_dict())
            print(f"Model loaded from {args.model_path}")
    
    def export_to_onnx(self, output_path):
        """导出模型为ONNX格式，用于FeatherCNN推理"""
        dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
        # 创建一个模拟的状态字典
        dummy_state_info = {
            'hp_percent': 1.0,
            'mp_percent': 1.0,
            'skill_cd': {1: 0, 2: 0, 3: 0},
            'summoner_cd': {'skill_1': 0, 'skill_2': 0},
            'item_cd': 0,
            'level': 1,
            'gold': 0,
            'minimap': {'allies': [], 'enemies': []}
        }
        torch.onnx.export(
            self.policy_net,
            (dummy_input, None, dummy_state_info),
            output_path,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input', 'hidden', 'state_info'],
            output_names=['output_0', 'output_1', 'output_2', 'output_3', 
                         'output_4', 'output_5', 'output_6', 'output_7', 'value'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output_0': {0: 'batch_size'},
                'output_1': {0: 'batch_size'},
                'output_2': {0: 'batch_size'},
                'output_3': {0: 'batch_size'},
                'output_4': {0: 'batch_size'},
                'output_5': {0: 'batch_size'},
                'output_6': {0: 'batch_size'},
                'output_7': {0: 'batch_size'},
                'value': {0: 'batch_size'}
            }
        )
        print(f"Model exported to ONNX: {output_path}")
    
    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")
    
    def update_old_policy(self):
        """更新旧策略网络"""
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())
    
    def update_game_state(self, game_state):
        """更新游戏状态用于action mask"""
        self.action_mask.update_game_state(game_state)
    
    def select_action(self, state, hidden_state=None, game_state=None):
        """选择动作，返回动作、对数概率和价值"""
        # 更新游戏状态用于action mask
        if game_state is not None:
            self.update_game_state(game_state)
        
        # 使用FeatherCNN加速推理
        if self.use_feather_cnn:
            return self._select_action_feather(state, game_state)
        
        # 默认使用PyTorch推理
        return self._select_action_pytorch(state, hidden_state, game_state)
    
    def _select_action_pytorch(self, state, hidden_state=None, game_state=None):
        """使用PyTorch进行动作选择，支持游戏状态"""
        tmp_state = self.preprocess_image(state).unsqueeze(0)
        self.policy_net.eval()
        
        with torch.no_grad():
            # 传递游戏状态给网络
            logits_list, value, new_hidden = self.policy_net(tmp_state, hidden_state, game_state)
            
            # 应用action mask
            logits_list = self.action_mask.apply_mask_to_logits(logits_list)
            
            actions = []
            log_probs = []
            
            for logits in logits_list:
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                actions.append(action.item())
                log_probs.append(log_prob)
            
            total_log_prob = torch.sum(torch.stack(log_probs))
        
        return actions, total_log_prob.item(), value.item(), new_hidden
    
    def _select_action_feather(self, state, game_state=None):
        """使用FeatherCNN加速进行动作选择"""
        # 预处理图像
        input_data = self.preprocess_image_feather(state)
        
        # 使用FeatherCNN推理（FeatherCNN暂不支持状态向量，使用PyTorch fallback）
        print("Warning: FeatherCNN doesn't support state vector yet, falling back to PyTorch")
        return self._select_action_pytorch(state, None, game_state)
    
    def preprocess_image_feather(self, image, target_size=(640, 640)):
        """预处理图像用于FeatherCNN"""
        resized_image = cv2.resize(image, target_size)
        # FeatherCNN期望的输入格式 (H, W, C) -> (C, H, W) -> (1, C, H, W)
        input_data = resized_image.transpose(2, 0, 1).astype(np.float32)
        input_data = np.expand_dims(input_data, 0)
        return input_data
    
    def preprocess_image(self, image, target_size=(640, 640)):
        """预处理图像"""
        resized_image = cv2.resize(image, target_size)
        tensor_image = torch.from_numpy(resized_image).float().permute(2, 0, 1)
        return tensor_image.to(device)
    
    def compute_advantages(self, rewards, values, next_values, dones, gamma=0.99, lam=0.95):
        """计算优势函数（GAE）"""
        advantages = []
        advantage = 0.0
        
        # 从后向前计算
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantage = delta + gamma * lam * (1 - dones[t]) * advantage
            advantages.insert(0, advantage)
        
        # 标准化优势
        advantages = torch.tensor(advantages, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 计算目标价值 (TD target)
        returns = advantages + torch.tensor(values, device=self.device)
        
        return advantages, returns
    
    def dual_ppo_loss(self, log_probs, old_log_probs, advantages, values, returns):
        """
        dual-PPO损失函数
        公式: Et[max(min(r_t(θ)Â_t, clip(r_t(θ),1−ε,1+ε)Â_t), cÂ_t)]
        
        其中:
        - r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t) 是重要性采样比率
        - Â_t 是优势函数
        - c 是dual-PPO的下界参数
        """
        # 计算重要性采样比率
        ratio = torch.exp(log_probs - old_log_probs)
        
        # 标准PPO裁剪目标
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
        
        # 标准PPO的min操作
        clipped_surrogate = torch.min(surr1, surr2)
        
        # dual-PPO的改进：添加第二个下界 max(..., c*A_t)
        # 当A_t < 0时，防止策略更新幅度过大
        dual_surrogate = torch.max(clipped_surrogate, self.dual_ppo_c * advantages)
        
        # 策略损失（取负号因为我们要最大化目标）
        policy_loss = -torch.mean(dual_surrogate)
        
        # 价值损失
        value_loss = self.value_loss_coef * nn.MSELoss()(values, returns)
        
        # 总损失
        total_loss = policy_loss + value_loss
        
        return total_loss, policy_loss, value_loss
    
    def train(self, states, actions, log_probs, rewards, values, next_values, dones, state_infos=None):
        """
        PPO训练，支持游戏状态
        
        Args:
            states: 图像状态列表
            actions: 动作列表
            log_probs: 旧策略的对数概率
            rewards: 奖励列表
            values: 价值列表
            next_values: 下一个状态价值列表
            dones: 完成标志列表
            state_infos: 游戏状态信息列表
        """
        # 计算优势函数
        advantages, returns = self.compute_advantages(rewards, values, next_values, dones)
        
        # 将数据转换为张量
        actions = torch.tensor(actions, device=self.device)
        old_log_probs = torch.tensor(log_probs, device=self.device)
        states = torch.stack([self.preprocess_image(s) for s in states]).to(device)
        
        # 如果状态信息为None，创建默认值
        if state_infos is None:
            state_infos = [None] * len(states)
        
        # 多次迭代优化
        for epoch in range(self.ppo_epochs):
            self.policy_net.train()
            
            total_loss_sum = 0.0
            policy_loss_sum = 0.0
            value_loss_sum = 0.0
            
            # 逐个处理每个样本（因为每个样本有不同的状态信息）
            for i in range(len(states)):
                # 前向传播，传递状态信息
                logits_list, value, _ = self.policy_net(states[i:i+1], None, state_infos[i])
                
                # 计算当前策略的对数概率
                current_log_prob = self.policy_net.get_log_probs(logits_list, actions[i:i+1])
                
                # 计算dual-PPO损失
                total_loss, policy_loss, value_loss = self.dual_ppo_loss(
                    current_log_prob, old_log_probs[i:i+1], advantages[i:i+1], 
                    value.squeeze(), returns[i:i+1]
                )
                
                # 累加损失
                total_loss_sum += total_loss
                policy_loss_sum += policy_loss
                value_loss_sum += value_loss
            
            # 平均损失
            total_loss_avg = total_loss_sum / len(states)
            policy_loss_avg = policy_loss_sum / len(states)
            value_loss_avg = value_loss_sum / len(states)
            
            # 优化
            self.optimizer.zero_grad()
            total_loss_avg.backward()
            self.optimizer.step()
            
            print(f"Epoch {epoch+1}/{self.ppo_epochs}, Loss: {total_loss_avg.item():.4f}, "
                  f"Policy Loss: {policy_loss_avg.item():.4f}, Value Loss: {value_loss_avg.item():.4f}")
        
        # 更新旧策略
        self.update_old_policy()
        self.steps_done += 1
