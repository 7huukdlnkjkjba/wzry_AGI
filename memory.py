import time
import threading
from collections import namedtuple
import random

# Transition - 一个命名元组(named tuple)用于表示环境中的单次状态迁移(single transition)。
#     该类的作用本质上是将状态-动作对[(state, action) pairs]映射到他们的下一个结果，即[(next_state, action) pairs]，
# 其中的 状态(state)是指从屏幕上获得的差分图像块(screen diifference image)。
#
# ReplayMemory - 一个大小有限的循环缓冲区，用于保存最近观察到的迁移(transition)。

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

# PPOMemory - PPO专用的记忆单元，支持存储对数概率、价值和状态信息
PPOMemory = namedtuple('PPOMemory',
                       ('state', 'action', 'log_prob', 'reward', 'value', 'next_value', 'done', 'state_info'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PPORolloutBuffer(object):
    """
    PPO的Rollout缓冲区，用于存储一个episode的完整轨迹数据
    
    内部实现为内存高效的循环队列，支持各种长度的样本以及基于生成时间的数据样本
    
    特性：
    - 基于生成时间的数据样本管理
    - 支持多服务器数据合并
    - 线程安全操作
    - 支持游戏状态信息存储
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.position = 0
        self.size = 0
        
        # 时间戳索引（用于基于时间的数据管理）
        self.timestamps = [0] * capacity
        
        # 服务器来源记录
        self.server_ids = [""] * capacity
        
        # 锁
        self.lock = threading.Lock()

    def push(self, state, action, log_prob, reward, value, next_value, done, state_info=None, timestamp=None, server_id=""):
        """保存一个轨迹样本，支持状态信息"""
        with self.lock:
            self.buffer[self.position] = PPOMemory(state, action, log_prob, reward, value, next_value, done, state_info)
            self.timestamps[self.position] = timestamp or time.time()
            self.server_ids[self.position] = server_id
            
            if self.size < self.capacity:
                self.size += 1
            
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """随机采样一个批次"""
        with self.lock:
            if self.size == 0:
                return []
            indices = random.sample(range(self.size), min(batch_size, self.size))
            return [self.buffer[i] for i in indices]

    def sample_by_time_range(self, start_time, end_time):
        """按时间范围采样"""
        with self.lock:
            samples = []
            for i in range(self.size):
                pos = (self.position - self.size + i) % self.capacity
                if start_time <= self.timestamps[pos] <= end_time:
                    samples.append(self.buffer[pos])
            return samples

    def sample_by_server(self, server_id):
        """按服务器来源采样"""
        with self.lock:
            samples = []
            for i in range(self.size):
                pos = (self.position - self.size + i) % self.capacity
                if self.server_ids[pos] == server_id:
                    samples.append(self.buffer[pos])
            return samples

    def get_all(self):
        """获取所有样本"""
        with self.lock:
            if self.size == 0:
                return []
            result = []
            for i in range(self.size):
                pos = (self.position - self.size + i) % self.capacity
                result.append(self.buffer[pos])
            return result

    def get_recent(self, count):
        """获取最近的count个样本"""
        with self.lock:
            if self.size == 0:
                return []
            count = min(count, self.size)
            result = []
            for i in range(count):
                pos = (self.position - 1 - i) % self.capacity
                result.append(self.buffer[pos])
            return result[::-1]  # 保持时间顺序

    def clear(self):
        """清空缓冲区"""
        with self.lock:
            self.buffer = [None] * self.capacity
            self.timestamps = [0] * self.capacity
            self.server_ids = [""] * self.capacity
            self.position = 0
            self.size = 0

    def get_statistics(self):
        """获取统计信息"""
        with self.lock:
            if self.size == 0:
                return {
                    'size': 0,
                    'server_distribution': {},
                    'oldest_timestamp': 0,
                    'newest_timestamp': 0
                }
            
            server_dist = {}
            oldest_ts = float('inf')
            newest_ts = 0
            
            for i in range(self.size):
                pos = (self.position - self.size + i) % self.capacity
                server = self.server_ids[pos]
                server_dist[server] = server_dist.get(server, 0) + 1
                oldest_ts = min(oldest_ts, self.timestamps[pos])
                newest_ts = max(newest_ts, self.timestamps[pos])
            
            return {
                'size': self.size,
                'server_distribution': server_dist,
                'oldest_timestamp': oldest_ts,
                'newest_timestamp': newest_ts
            }

    def __len__(self):
        with self.lock:
            return self.size
