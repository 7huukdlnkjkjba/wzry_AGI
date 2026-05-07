import numpy as np


class ActionMask:
    """
    Action Mask机制 - 根据专家经验过滤不合理、受限制的动作
    
    动作类型说明：
    - move_action: 移动动作 (0: 不移动, 1: 移动)
    - angle_action: 移动方向 (0-359度)
    - info_action: 信息动作 (0: 无操作, 1-8: 各种信号)
    - attack_action: 攻击动作 (0: 无操作, 1-10: 各种攻击)
    - action_type: 动作类型 (0-2)
    - arg1: 参数1 (技能方向等)
    - arg2: 参数2
    - arg3: 参数3
    """
    
    def __init__(self):
        # 技能冷却时间（单位：帧）
        self.skill_cd = {
            1: 0,  # 1技能冷却
            2: 0,  # 2技能冷却
            3: 0   # 3技能冷却
        }
        
        # 召唤师技能冷却
        self.summoner_cd = {
            'flash': 0,
            'heal': 0
        }
        
        # 装备技能冷却
        self.item_cd = 0
        
        # 蓝量是否充足
        self.mana_enough = True
        
        # 是否正在施法
        self.casting = False
        
        # 是否在冷却中
        self.on_cooldown = False
    
    def update_game_state(self, game_state):
        """
        更新游戏状态信息
        :param game_state: 包含技能冷却、蓝量、施法状态等信息的字典
        """
        if 'skill_cd' in game_state:
            self.skill_cd.update(game_state['skill_cd'])
        
        if 'summoner_cd' in game_state:
            self.summoner_cd.update(game_state['summoner_cd'])
        
        if 'item_cd' in game_state:
            self.item_cd = game_state['item_cd']
        
        if 'mana_enough' in game_state:
            self.mana_enough = game_state['mana_enough']
        
        if 'casting' in game_state:
            self.casting = game_state['casting']
        
        if 'on_cooldown' in game_state:
            self.on_cooldown = game_state['on_cooldown']
    
    def get_move_mask(self):
        """获取移动动作的mask"""
        mask = np.ones(2, dtype=np.float32)
        # 如果正在施法，禁止移动
        if self.casting:
            mask[1] = 0.0  # 禁止移动
        return mask
    
    def get_angle_mask(self):
        """获取移动方向的mask"""
        # 默认所有方向都允许
        mask = np.ones(360, dtype=np.float32)
        return mask
    
    def get_info_mask(self):
        """获取信息动作的mask"""
        mask = np.ones(9, dtype=np.float32)
        
        # 升级技能需要满足等级条件（简化处理）
        # 0: 无操作, 1-2: 购买装备, 3-5: 信号, 6-8: 升级技能
        # 默认允许购买装备和发送信号
        # 升级技能根据等级判断（这里简化为全部允许）
        
        return mask
    
    def get_attack_mask(self):
        """获取攻击动作的mask"""
        mask = np.ones(11, dtype=np.float32)
        
        # 0: 无操作
        # 1: 普通攻击
        # 2: 攻击小兵
        # 3: 攻击塔
        # 4: 回城
        # 5: 恢复
        # 6: 装备技能
        # 7: 召唤师技能
        # 8: 1技能
        # 9: 2技能
        # 10: 3技能
        
        # 技能冷却检查
        if self.skill_cd.get(1, 0) > 0 or not self.mana_enough:
            mask[8] = 0.0  # 1技能不可用
        
        if self.skill_cd.get(2, 0) > 0 or not self.mana_enough:
            mask[9] = 0.0  # 2技能不可用
        
        if self.skill_cd.get(3, 0) > 0 or not self.mana_enough:
            mask[10] = 0.0  # 3技能不可用
        
        # 召唤师技能冷却检查
        if self.summoner_cd.get('flash', 0) > 0:
            mask[7] = 0.0  # 召唤师技能不可用
        
        # 装备技能冷却检查
        if self.item_cd > 0:
            mask[6] = 0.0  # 装备技能不可用
        
        # 回城需要在安全区域（简化处理）
        # mask[4] = 1.0  # 回城
        
        # 恢复需要有足够的血量/蓝量（简化处理）
        # mask[5] = 1.0  # 恢复
        
        return mask
    
    def get_action_type_mask(self):
        """获取动作类型的mask"""
        mask = np.ones(3, dtype=np.float32)
        return mask
    
    def get_arg1_mask(self):
        """获取参数1的mask"""
        mask = np.ones(360, dtype=np.float32)
        return mask
    
    def get_arg2_mask(self):
        """获取参数2的mask"""
        mask = np.ones(100, dtype=np.float32)
        return mask
    
    def get_arg3_mask(self):
        """获取参数3的mask"""
        mask = np.ones(5, dtype=np.float32)
        return mask
    
    def get_all_masks(self):
        """获取所有动作的mask"""
        return [
            self.get_move_mask(),
            self.get_angle_mask(),
            self.get_info_mask(),
            self.get_attack_mask(),
            self.get_action_type_mask(),
            self.get_arg1_mask(),
            self.get_arg2_mask(),
            self.get_arg3_mask()
        ]
    
    def apply_mask_to_logits(self, logits_list):
        """
        将mask应用到logits上，通过设置负无穷来禁止某些动作
        :param logits_list: 各动作头的logits列表
        :return: 应用mask后的logits列表
        """
        masked_logits = []
        masks = self.get_all_masks()
        
        for logits, mask in zip(logits_list, masks):
            # 将mask为0的位置设置为负无穷
            mask_tensor = logits.new_tensor(mask)
            masked_logit = logits + (mask_tensor - 1) * 1e10  # mask=0时变为-inf
            masked_logits.append(masked_logit)
        
        return masked_logits