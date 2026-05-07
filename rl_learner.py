import json
import threading
import time
from collections import deque

import torch
import torch.nn as nn

from argparses import args, device
from net_ppo import ActorCritic
from ppoAgent import PPOAgent


class RLLearner:
    """
    分布式强化学习Learner组件
    
    负责：
    - 接收来自多个AI Server的数据
    - 并行计算梯度
    - 同步梯度取均值
    - 更新策略网络
    - 将更新后的策略分发给AI Server
    """
    
    def __init__(self):
        self.device = device
        self.agent = PPOAgent()
        
        # 梯度累积缓冲区
        self.gradient_buffer = deque(maxlen=args.num_servers or 10)
        
        # 策略版本
        self.policy_version = 0
        
        # 锁
        self.gradient_lock = threading.Lock()
        self.policy_lock = threading.Lock()
        
        # 训练状态
        self.is_training = True
        
        # 统计信息
        self.total_samples = 0
        self.update_count = 0
        
        # 启动训练线程
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()
    
    def _training_loop(self):
        """训练主循环"""
        while self.is_training:
            # 等待足够的梯度积累
            while len(self.gradient_buffer) < args.min_gradients or not self._has_enough_data():
                time.sleep(0.1)
                if not self.is_training:
                    return
            
            # 执行一次训练更新
            self._perform_update()
    
    def _has_enough_data(self):
        """检查是否有足够的训练数据"""
        # 检查全局memory是否有足够数据
        from globalInfo import globalInfo
        return globalInfo.is_memory_bigger_batch_size_ppo()
    
    def _perform_update(self):
        """执行策略更新"""
        # 获取训练数据
        from globalInfo import globalInfo
        transitions = globalInfo.random_batch_size_memory_ppo()
        
        # 提取数据
        states = [t.state for t in transitions]
        actions = [t.action for t in transitions]
        log_probs = [t.log_prob for t in transitions]
        rewards = [t.reward for t in transitions]
        values = [t.value for t in transitions]
        next_values = [t.next_value for t in transitions]
        dones = [t.done for t in transitions]
        
        # 执行PPO训练
        self.agent.train(states, actions, log_probs, rewards, values, next_values, dones)
        
        # 更新策略版本
        with self.policy_lock:
            self.policy_version += 1
        
        # 清空梯度缓冲区
        with self.gradient_lock:
            self.gradient_buffer.clear()
        
        self.update_count += 1
        self.total_samples += len(states)
        
        print(f"[RL Learner] Update {self.update_count} completed, total samples: {self.total_samples}")
    
    def receive_gradient(self, gradient_data):
        """
        接收来自AI Server的梯度数据
        :param gradient_data: 包含梯度信息的字典
        """
        with self.gradient_lock:
            self.gradient_buffer.append(gradient_data)
    
    def receive_sample(self, sample_data):
        """
        接收来自Dispatch Module的采样数据
        :param sample_data: 采样数据
        """
        from globalInfo import globalInfo
        globalInfo.store_transition_ppo(
            sample_data['state'],
            sample_data['action'],
            sample_data['log_prob'],
            sample_data['reward'],
            sample_data['value'],
            sample_data['next_value'],
            sample_data['done']
        )
    
    def get_policy(self):
        """获取当前策略"""
        with self.policy_lock:
            policy_state = self.agent.policy_net.state_dict()
            version = self.policy_version
        
        return {
            'version': version,
            'state_dict': policy_state
        }
    
    def broadcast_policy(self):
        """广播策略到所有AI Server"""
        # 通过Dispatch Module广播
        from dispatch_module import DispatchModule
        dispatch = DispatchModule.get_instance()
        dispatch.broadcast_policy(self.get_policy())
    
    def save_model(self, path=None):
        """保存模型"""
        if path is None:
            path = f'src/wzry_ai_ppo_v{self.policy_version}.pt'
        self.agent.save_model(path)
    
    def stop(self):
        """停止训练"""
        self.is_training = False
        if self.training_thread.is_alive():
            self.training_thread.join()
    
    def get_stats(self):
        """获取统计信息"""
        return {
            'policy_version': self.policy_version,
            'update_count': self.update_count,
            'total_samples': self.total_samples,
            'buffer_size': len(self.gradient_buffer)
        }
