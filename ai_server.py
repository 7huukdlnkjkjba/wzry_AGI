import threading
import time
import uuid

import cv2
import numpy as np

from paddleocr_parser import PaddleOCRManager
from reward_system import RewardSystem


class AIServer:
    """
    AI Server组件 - 游戏环境和AI之间的交互逻辑
    
    负责：
    - 从游戏中收集state
    - 预测英雄行为
    - 产生训练数据
    
    一台AI服务器绑定一个CPU内核
    """
    
    def __init__(self, server_id=None):
        self.server_id = server_id or str(uuid.uuid4())[:8]
        self.device_id = None
        
        # 连接状态
        self.is_connected = False
        self.is_running = False
        
        # 当前策略版本
        self.policy_version = -1
        
        # 数据收集统计
        self.samples_collected = 0
        self.episodes_completed = 0
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 游戏工具和环境（延迟初始化）
        self.tool = None
        self.env = None
        self.reward_util = None
        self.agent = None
        self.start_check = None
        
        # 游戏状态解析器（使用PaddleOCR）
        self.state_parser = PaddleOCRManager.get_instance(use_gpu=False)
        
        # 奖励系统
        self.reward_system = RewardSystem()
        
        # 本地轨迹缓冲区
        self.rollout_buffer = []
        
        # 数据发送回调
        self.on_data_ready = None
        
        print(f"[AI Server {self.server_id}] Created")
    
    def initialize(self, tool, env, reward_util, agent, start_check):
        """初始化游戏环境"""
        self.tool = tool
        self.env = env
        self.reward_util = reward_util
        self.agent = agent
        self.start_check = start_check
        self.is_connected = True
        print(f"[AI Server {self.server_id}] Initialized")
    
    def start(self):
        """启动AI Server"""
        if not self.is_connected:
            print(f"[AI Server {self.server_id}] Not connected, cannot start")
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        print(f"[AI Server {self.server_id}] Started")
    
    def stop(self):
        """停止AI Server"""
        self.is_running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join()
        print(f"[AI Server {self.server_id}] Stopped")
    
    def _run_loop(self):
        """主运行循环"""
        while self.is_running:
            try:
                self._collect_data()
            except Exception as e:
                print(f"[AI Server {self.server_id}] Error: {e}")
                time.sleep(1)
    
    def _collect_data(self):
        """收集游戏数据"""
        # 获取当前图像
        state = self.tool.screenshot_window()
        if state is None:
            time.sleep(0.01)
            return
        
        # 保存状态图像用于游戏状态解析
        self.last_state_image = state.copy()
        
        # 检查游戏是否开始
        check_result = self.start_check.get_max_label(state)
        if check_result != 'started':
            time.sleep(0.1)
            return
        
        print(f"[AI Server {self.server_id}] Game started, collecting data...")
        
        # 重置轨迹缓冲区
        self.rollout_buffer = []
        hidden_state = None
        prev_value = 0.0
        
        # 收集数据直到游戏结束
        while self.is_running:
            # 更新状态图像
            self.last_state_image = state.copy()
            
            # 获取游戏状态（用于action mask）
            game_state = self._get_game_state()
            
            # 选择动作
            action, log_prob, value, hidden_state = self.agent.select_action(
                state, hidden_state, game_state
            )
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 获取额外奖励
            extra_reward = self._calculate_additional_reward(info)
            total_reward = reward + extra_reward
            
            # 存储轨迹数据
            self.rollout_buffer.append({
                'state': state.copy(),
                'action': action,
                'log_prob': log_prob,
                'reward': total_reward,
                'value': value,
                'next_value': prev_value,
                'done': done,
                'server_id': self.server_id,
                'timestamp': time.time()
            })
            
            prev_value = value
            self.samples_collected += 1
            
            # 检查游戏是否结束
            if done == 1:
                self.episodes_completed += 1
                
                # 打印回合统计
                episode_summary = self.reward_system.get_episode_summary()
                print(f"[AI Server {self.server_id}] Episode {self.episodes_completed} Summary: {episode_summary}")
                
                # 处理最后一个样本
                if self.rollout_buffer:
                    self.rollout_buffer[-1]['next_value'] = 0.0
                
                # 发送数据
                self._send_data()
                
                # 重置奖励系统
                self.reward_system.reset()
                
                break
            
            state = next_state
            
            # 定期发送数据（避免内存积压）
            if len(self.rollout_buffer) >= 100:
                self._send_partial_data()
    
    def _get_game_state(self):
        """获取游戏状态信息（用于action mask）"""
        # 使用游戏状态解析器从画面中解析状态
        if hasattr(self, 'last_state_image') and self.last_state_image is not None:
            try:
                parsed_state = self.state_parser.parse(self.last_state_image)
                
                return {
                    'skill_cd': parsed_state['skill_cd'],
                    'summoner_cd': parsed_state['summoner_cd'],
                    'item_cd': parsed_state['item_cd'],
                    'mana_enough': parsed_state['mana_enough'],
                    'casting': parsed_state['is_casting'],
                    'on_cooldown': any([v > 0 for v in parsed_state['skill_cd'].values()]),
                    'hp_percent': parsed_state['hp_percent'],
                    'mp_percent': parsed_state['mp_percent'],
                    'level': parsed_state['level'],
                    'gold': parsed_state['gold']
                }
            except Exception as e:
                print(f"[AI Server {self.server_id}] State parsing error: {e}")
        
        # 默认返回空状态
        return {
            'skill_cd': {1: 0, 2: 0, 3: 0},
            'summoner_cd': {'flash': 0, 'heal': 0},
            'item_cd': 0,
            'mana_enough': True,
            'casting': False,
            'on_cooldown': False,
            'hp_percent': 1.0,
            'mp_percent': 1.0,
            'level': 15,
            'gold': 10000
        }
    
    def _calculate_additional_reward(self, info):
        """计算额外奖励（使用RewardSystem）"""
        try:
            # 构建游戏状态字典
            game_state = {}
            
            # 从解析的状态中提取信息
            if hasattr(self, 'last_state_image') and self.last_state_image is not None:
                parsed_state = self.state_parser.parse(self.last_state_image)
                
                game_state['hp_percent'] = parsed_state.get('hp_percent', 1.0)
                game_state['mp_percent'] = parsed_state.get('mp_percent', 1.0)
                game_state['gold'] = parsed_state.get('gold', 0)
                game_state['level'] = parsed_state.get('level', 1)
            
            # 从环境info中提取信息
            if info:
                game_state['kills'] = info.get('kills', 0)
                game_state['assists'] = info.get('assists', 0)
                game_state['deaths'] = info.get('deaths', 0)
                game_state['last_hits'] = info.get('last_hits', 0)
                game_state['towers_destroyed'] = info.get('towers_destroyed', 0)
                game_state['damage_dealt'] = info.get('damage_dealt', 0)
                game_state['damage_taken'] = info.get('damage_taken', 0)
            
            # 使用奖励系统计算奖励
            additional_reward = self.reward_system.calculate_reward(game_state)
            
            return additional_reward
            
        except Exception as e:
            print(f"[AI Server {self.server_id}] Reward calculation error: {e}")
            return 0.0
    
    def _send_data(self):
        """发送完整轨迹数据"""
        if self.on_data_ready and self.rollout_buffer:
            self.on_data_ready(self.rollout_buffer)
            print(f"[AI Server {self.server_id}] Sent {len(self.rollout_buffer)} samples")
        self.rollout_buffer = []
    
    def _send_partial_data(self):
        """发送部分数据"""
        if self.on_data_ready and self.rollout_buffer:
            # 保留最后几个样本（用于计算next_value）
            send_data = self.rollout_buffer[:-5]
            self.rollout_buffer = self.rollout_buffer[-5:]
            self.on_data_ready(send_data)
    
    def update_policy(self, policy_data):
        """更新策略"""
        if policy_data['version'] > self.policy_version:
            self.policy_version = policy_data['version']
            self.agent.policy_net.load_state_dict(policy_data['state_dict'])
            self.agent.old_policy_net.load_state_dict(policy_data['state_dict'])
            print(f"[AI Server {self.server_id}] Updated policy to version {self.policy_version}")
    
    def get_status(self):
        """获取服务器状态"""
        return {
            'server_id': self.server_id,
            'is_connected': self.is_connected,
            'is_running': self.is_running,
            'policy_version': self.policy_version,
            'samples_collected': self.samples_collected,
            'episodes_completed': self.episodes_completed
        }
