import cv2
import numpy as np


class GameStateParser:
    """
    游戏状态解析模块 - 从游戏画面中解析各种状态信息
    
    解析内容：
    - 血量和最大血量
    - 蓝量和最大蓝量
    - 技能冷却时间
    - 召唤师技能冷却
    - 装备技能冷却
    - 等级和经验
    - 金币数量
    - Buff/Debuff状态
    - 小地图信息
    """
    
    def __init__(self):
        # 屏幕区域配置（相对于640x640图像）
        self.regions = {
            # 玩家血条区域
            'hp_bar': {'x': 180, 'y': 580, 'w': 150, 'h': 12},
            'hp_bg': {'x': 180, 'y': 595, 'w': 150, 'h': 4},
            
            # 玩家蓝条区域
            'mp_bar': {'x': 180, 'y': 600, 'w': 150, 'h': 8},
            'mp_bg': {'x': 180, 'y': 610, 'w': 150, 'h': 4},
            
            # 技能图标区域（1-3技能 + 召唤师技能）
            'skill_1': {'x': 480, 'y': 520, 'w': 50, 'h': 50},
            'skill_2': {'x': 540, 'y': 520, 'w': 50, 'h': 50},
            'skill_3': {'x': 600, 'y': 520, 'w': 50, 'h': 50},
            'summoner': {'x': 420, 'y': 580, 'w': 45, 'h': 45},
            
            # 装备技能区域
            'item_skill': {'x': 600, 'y': 150, 'w': 45, 'h': 45},
            
            # 等级区域
            'level': {'x': 350, 'y': 580, 'w': 25, 'h': 25},
            
            # 金币区域
            'gold': {'x': 50, 'y': 580, 'w': 60, 'h': 25},
            
            # 小地图区域
            'minimap': {'x': 540, 'y': 50, 'w': 90, 'h': 90},
        }
        
        # 颜色阈值配置
        self.color_thresholds = {
            'hp_red': ([0, 100, 100], [20, 255, 255]),      # 红色血量
            'hp_green': ([40, 100, 100], [80, 255, 255]),    # 绿色血量（回血时）
            'mp_blue': ([90, 100, 100], [130, 255, 255]),    # 蓝色蓝量
            'cd_gray': ([0, 0, 0], [180, 50, 100]),          # 冷却灰色遮罩
            'ready_green': ([40, 100, 100], [80, 255, 255]), # 就绪绿色边框
        }
        
        # 技能冷却模板
        self.skill_cd_templates = self._load_cd_templates()
        
        # 初始化状态
        self.reset()
    
    def _load_cd_templates(self):
        """加载冷却时间数字模板"""
        templates = {}
        # 这里可以添加数字识别模板
        # 实际应用中需要使用OCR或模板匹配来识别冷却数字
        return templates
    
    def reset(self):
        """重置状态"""
        self.last_state = {}
        self.skill_cd_cache = {
            1: 0, 2: 0, 3: 0,
            'summoner': 0,
            'item': 0
        }
    
    def parse(self, image):
        """
        解析游戏画面
        :param image: 游戏画面图像 (BGR格式)
        :return: 游戏状态字典
        """
        state = {
            'hp': 0,
            'max_hp': 100,
            'hp_percent': 0.0,
            'mp': 0,
            'max_mp': 100,
            'mp_percent': 0.0,
            'level': 1,
            'gold': 0,
            'skill_cd': {
                1: 0,
                2: 0,
                3: 0
            },
            'summoner_cd': {
                'flash': 0,
                'heal': 0
            },
            'item_cd': 0,
            'is_casting': False,
            'mana_enough': True,
            'buff_list': [],
            'debuff_list': [],
            'minimap_info': {}
        }
        
        # 解析血条
        state['hp_percent'] = self._parse_hp_bar(image)
        state['hp'] = int(state['hp_percent'] * 1000)
        state['max_hp'] = 1000
        
        # 解析蓝条
        state['mp_percent'] = self._parse_mp_bar(image)
        state['mp'] = int(state['mp_percent'] * 500)
        state['max_mp'] = 500
        state['mana_enough'] = state['mp_percent'] > 0.1
        
        # 解析技能冷却
        state['skill_cd'][1] = self._parse_skill_cd(image, 'skill_1')
        state['skill_cd'][2] = self._parse_skill_cd(image, 'skill_2')
        state['skill_cd'][3] = self._parse_skill_cd(image, 'skill_3')
        
        # 解析召唤师技能冷却
        state['summoner_cd']['flash'] = self._parse_skill_cd(image, 'summoner')
        
        # 解析装备技能冷却
        state['item_cd'] = self._parse_skill_cd(image, 'item_skill')
        
        # 解析等级
        state['level'] = self._parse_level(image)
        
        # 解析金币
        state['gold'] = self._parse_gold(image)
        
        # 解析施法状态
        state['is_casting'] = self._detect_casting(image)
        
        # 更新缓存
        self.last_state = state
        
        return state
    
    def _parse_hp_bar(self, image):
        """解析血条百分比"""
        region = self.regions['hp_bar']
        hp_region = image[region['y']:region['y']+region['h'], region['x']:region['x']+region['w']]
        
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(hp_region, cv2.COLOR_BGR2HSV)
        
        # 检测红色血量区域
        lower_red, upper_red = self.color_thresholds['hp_red']
        mask_red = cv2.inRange(hsv, np.array(lower_red), np.array(upper_red))
        
        # 检测绿色血量区域（回血时）
        lower_green, upper_green = self.color_thresholds['hp_green']
        mask_green = cv2.inRange(hsv, np.array(lower_green), np.array(upper_green))
        
        # 合并掩码
        mask = cv2.bitwise_or(mask_red, mask_green)
        
        # 计算血量百分比
        if mask.shape[0] * mask.shape[1] > 0:
            percent = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
            return max(0.0, min(1.0, percent))
        
        return 1.0  # 默认满血
    
    def _parse_mp_bar(self, image):
        """解析蓝条百分比"""
        region = self.regions['mp_bar']
        mp_region = image[region['y']:region['y']+region['h'], region['x']:region['x']+region['w']]
        
        hsv = cv2.cvtColor(mp_region, cv2.COLOR_BGR2HSV)
        
        lower_blue, upper_blue = self.color_thresholds['mp_blue']
        mask = cv2.inRange(hsv, np.array(lower_blue), np.array(upper_blue))
        
        if mask.shape[0] * mask.shape[1] > 0:
            percent = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
            return max(0.0, min(1.0, percent))
        
        return 1.0  # 默认满蓝
    
    def _parse_skill_cd(self, image, skill_name):
        """解析技能冷却时间"""
        region = self.regions[skill_name]
        skill_region = image[region['y']:region['y']+region['h'], region['x']:region['x']+region['w']]
        
        # 检测灰色冷却遮罩
        hsv = cv2.cvtColor(skill_region, cv2.COLOR_BGR2HSV)
        lower_gray, upper_gray = self.color_thresholds['cd_gray']
        mask = cv2.inRange(hsv, np.array(lower_gray), np.array(upper_gray))
        
        # 计算遮罩百分比（对应冷却进度）
        total_pixels = mask.shape[0] * mask.shape[1]
        gray_pixels = np.sum(mask > 0)
        
        if gray_pixels > total_pixels * 0.1:
            # 有冷却遮罩，估算冷却时间
            cd_percent = gray_pixels / total_pixels
            
            # 假设最大冷却时间（需要根据具体英雄调整）
            max_cd = self._get_max_cd(skill_name)
            
            # 估算剩余冷却时间
            estimated_cd = int(cd_percent * max_cd)
            
            return estimated_cd
        
        return 0  # 无冷却
    
    def _get_max_cd(self, skill_name):
        """获取技能最大冷却时间（基于经验值）"""
        cd_map = {
            'skill_1': 8,
            'skill_2': 12,
            'skill_3': 40,
            'summoner': 300,  # 闪现默认300秒
            'item_skill': 60
        }
        return cd_map.get(skill_name, 10)
    
    def _parse_level(self, image):
        """解析等级"""
        region = self.regions['level']
        level_region = image[region['y']:region['y']+region['h'], region['x']:region['x']+region['w']]
        
        # 使用简单的模板匹配或OCR来识别等级
        # 这里简化处理，返回默认等级
        return 15  # 假设满级
    
    def _parse_gold(self, image):
        """解析金币数量"""
        region = self.regions['gold']
        gold_region = image[region['y']:region['y']+region['h'], region['x']:region['x']+region['w']]
        
        # 使用OCR识别金币数字
        # 这里简化处理，返回默认值
        return 10000  # 假设10000金币
    
    def _detect_casting(self, image):
        """检测是否正在施法"""
        # 检测施法动画特效（简化处理）
        return False
    
    def parse_minimap(self, image):
        """解析小地图信息"""
        region = self.regions['minimap']
        minimap = image[region['y']:region['y']+region['h'], region['x']:region['x']+region['w']]
        
        info = {
            'ally_positions': [],
            'enemy_positions': [],
            'tower_positions': [],
            'objective_positions': []
        }
        
        # 检测不同颜色的点
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        
        # 检测友方英雄（蓝色）
        lower_blue = np.array([90, 100, 100])
        upper_blue = np.array([130, 255, 255])
        ally_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # 检测敌方英雄（红色）
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([20, 255, 255])
        enemy_mask = cv2.inRange(hsv, lower_red, upper_red)
        
        # 找到轮廓
        ally_contours, _ = cv2.findContours(ally_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        enemy_contours, _ = cv2.findContours(enemy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in ally_contours:
            if cv2.contourArea(cnt) > 5:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    info['ally_positions'].append((cx, cy))
        
        for cnt in enemy_contours:
            if cv2.contourArea(cnt) > 5:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    info['enemy_positions'].append((cx, cy))
        
        return info
    
    def get_debug_info(self):
        """获取调试信息"""
        return {
            'regions': self.regions,
            'last_state': self.last_state,
            'skill_cd_cache': self.skill_cd_cache
        }


class GameStateParserManager:
    """
    游戏状态解析管理器
    """
    
    _instance = None
    
    def __init__(self):
        self.parser = GameStateParser()
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = GameStateParserManager()
        return cls._instance
    
    def parse(self, image):
        """解析游戏状态"""
        return self.parser.parse(image)
    
    def parse_minimap(self, image):
        """解析小地图"""
        return self.parser.parse_minimap(image)