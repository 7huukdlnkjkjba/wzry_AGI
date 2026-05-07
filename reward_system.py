import numpy as np


class RewardSystem:
    """
    可扩展的奖励系统
    
    支持多种奖励类型：
    - 击杀奖励
    - 补刀奖励
    - 推塔奖励
    - 生存奖励
    - 经济奖励
    - 伤害奖励
    - 治疗奖励
    - 团队合作奖励
    """
    
    def __init__(self):
        # 奖励权重配置
        self.reward_weights = {
            # 击杀相关
            'kill': 10.0,           # 击杀英雄
            'assist': 3.0,          # 助攻
            'death': -5.0,          # 死亡
            
            # 补刀相关
            'last_hit_creep': 1.0,  # 补刀小兵
            'last_hit_jungle': 1.5, # 击杀野怪
            
            # 推塔相关
            'tower_destroy': 20.0,  # 摧毁防御塔
            'inhibitor_destroy': 30.0,  # 摧毁水晶
            'nexus_destroy': 100.0, # 摧毁基地
            
            # 生存相关
            'survival_time': 0.01,  # 每帧生存奖励
            'avoid_damage': 0.5,    # 成功躲避伤害
            
            # 经济相关
            'gold_earned': 0.01,    # 获得金币
            'item_purchase': 0.1,   # 购买装备
            
            # 战斗相关
            'damage_dealt': 0.01,   # 造成伤害
            'damage_taken': -0.005, # 受到伤害
            'healing_done': 0.02,   # 治疗量
            
            # 团队合作
            'objective_control': 2.0,    # 控制目标
            'vision_score': 0.1,         # 视野得分
            'ward_placed': 1.0,         # 放置眼位
            
            # 游戏进度
            'game_progress': 0.1,   # 游戏进度奖励
            'victory': 1000.0,      # 胜利奖励
            'defeat': -500.0,       # 失败惩罚
        }
        
        # 累积统计
        self.episode_stats = {
            'kills': 0,
            'assists': 0,
            'deaths': 0,
            'last_hits': 0,
            'towers_destroyed': 0,
            'total_damage': 0,
            'total_healing': 0,
            'gold_earned': 0,
            'survival_time': 0,
        }
        
        # 上一帧状态（用于计算增量）
        self.prev_state = {}
    
    def reset(self):
        """重置奖励系统状态"""
        self.episode_stats = {k: 0 for k in self.episode_stats.keys()}
        self.prev_state = {}
    
    def calculate_reward(self, current_state):
        """
        计算当前帧的奖励
        :param current_state: 当前游戏状态字典
        :return: 总奖励值
        """
        total_reward = 0.0
        
        # 计算各项奖励
        rewards = {
            'kill': self._calculate_kill_reward(current_state),
            'assist': self._calculate_assist_reward(current_state),
            'death': self._calculate_death_reward(current_state),
            'last_hit_creep': self._calculate_last_hit_reward(current_state),
            'tower_destroy': self._calculate_tower_reward(current_state),
            'survival_time': self._calculate_survival_reward(current_state),
            'damage_dealt': self._calculate_damage_reward(current_state),
            'damage_taken': self._calculate_damage_taken_reward(current_state),
            'gold_earned': self._calculate_gold_reward(current_state),
        }
        
        # 累加奖励
        for reward_type, value in rewards.items():
            if reward_type in self.reward_weights:
                total_reward += value * self.reward_weights[reward_type]
        
        # 更新累积统计
        self._update_stats(current_state)
        
        # 更新上一帧状态
        self.prev_state = current_state.copy()
        
        return total_reward
    
    def _calculate_kill_reward(self, state):
        """计算击杀奖励"""
        current_kills = state.get('kills', 0)
        prev_kills = self.prev_state.get('kills', 0)
        return max(0, current_kills - prev_kills)
    
    def _calculate_assist_reward(self, state):
        """计算助攻奖励"""
        current_assists = state.get('assists', 0)
        prev_assists = self.prev_state.get('assists', 0)
        return max(0, current_assists - prev_assists)
    
    def _calculate_death_reward(self, state):
        """计算死亡惩罚"""
        current_deaths = state.get('deaths', 0)
        prev_deaths = self.prev_state.get('deaths', 0)
        return max(0, current_deaths - prev_deaths)
    
    def _calculate_last_hit_reward(self, state):
        """计算补刀奖励"""
        current_last_hits = state.get('last_hits', 0)
        prev_last_hits = self.prev_state.get('last_hits', 0)
        return max(0, current_last_hits - prev_last_hits)
    
    def _calculate_tower_reward(self, state):
        """计算推塔奖励"""
        current_towers = state.get('towers_destroyed', 0)
        prev_towers = self.prev_state.get('towers_destroyed', 0)
        return max(0, current_towers - prev_towers)
    
    def _calculate_survival_reward(self, state):
        """计算生存奖励"""
        return 1  # 每帧生存奖励
    
    def _calculate_damage_reward(self, state):
        """计算造成伤害奖励"""
        current_damage = state.get('damage_dealt', 0)
        prev_damage = self.prev_state.get('damage_dealt', 0)
        return max(0, current_damage - prev_damage)
    
    def _calculate_damage_taken_reward(self, state):
        """计算受到伤害惩罚"""
        current_damage = state.get('damage_taken', 0)
        prev_damage = self.prev_state.get('damage_taken', 0)
        return max(0, current_damage - prev_damage)
    
    def _calculate_gold_reward(self, state):
        """计算金币奖励"""
        current_gold = state.get('gold', 0)
        prev_gold = self.prev_state.get('gold', 0)
        return max(0, current_gold - prev_gold)
    
    def _update_stats(self, state):
        """更新累积统计"""
        for key in self.episode_stats.keys():
            if key in state:
                self.episode_stats[key] = state[key]
    
    def get_episode_summary(self):
        """获取回合统计摘要"""
        return {
            'total_kills': self.episode_stats['kills'],
            'total_assists': self.episode_stats['assists'],
            'total_deaths': self.episode_stats['deaths'],
            'kda': self._calculate_kda(),
            'total_last_hits': self.episode_stats['last_hits'],
            'towers_destroyed': self.episode_stats['towers_destroyed'],
            'total_damage': self.episode_stats['total_damage'],
            'total_healing': self.episode_stats['total_healing'],
            'gold_earned': self.episode_stats['gold_earned'],
        }
    
    def _calculate_kda(self):
        """计算KDA"""
        deaths = self.episode_stats['deaths'] or 1
        return (self.episode_stats['kills'] + self.episode_stats['assists']) / deaths
    
    def set_reward_weight(self, reward_type, weight):
        """设置奖励权重"""
        if reward_type in self.reward_weights:
            self.reward_weights[reward_type] = weight
    
    def get_reward_weights(self):
        """获取当前奖励权重配置"""
        return self.reward_weights.copy()