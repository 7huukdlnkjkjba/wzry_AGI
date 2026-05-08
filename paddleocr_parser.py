import cv2
import numpy as np
import time
from typing import Dict, Tuple, Optional
import threading


class PaddleOCRParser:
    """
    使用PaddleOCR（ppocronnx）解析游戏状态
    
    功能：
    - 识别血量/蓝量数值
    - 识别技能冷却时间
    - 解析小地图信息
    - 识别金币/等级
    """
    
    def __init__(self, use_gpu=False, model_dir=None):
        self.use_gpu = use_gpu
        self.model_dir = model_dir or 'models/ocr'
        self.ocr_engine = None
        self.lock = threading.Lock()
        
        # 延迟加载OCR引擎
        self._ocr_initialized = False
        
        # ROI区域配置（屏幕百分比）
        self.roi_config = {
            'hp_bar': (0.02, 0.88, 0.12, 0.92),  # 血条区域
            'mp_bar': (0.02, 0.92, 0.12, 0.96),  # 蓝条区域
            'hp_text': (0.12, 0.88, 0.18, 0.92),  # 血量数值
            'mp_text': (0.12, 0.92, 0.18, 0.96),  # 蓝量数值
            'skill_1': (0.68, 0.85, 0.72, 0.90),  # 1技能
            'skill_2': (0.73, 0.85, 0.77, 0.90),  # 2技能
            'skill_3': (0.78, 0.85, 0.82, 0.90),  # 3技能
            'summoner_1': (0.83, 0.85, 0.87, 0.90),  # 召唤师技能1
            'summoner_2': (0.88, 0.85, 0.92, 0.90),  # 召唤师技能2
            'minimap': (0.88, 0.02, 0.98, 0.18),  # 小地图
            'gold': (0.50, 0.02, 0.60, 0.05),  # 金币
            'level': (0.48, 0.02, 0.50, 0.05),  # 等级
        }
        
        print("[PaddleOCRParser] Created (OCR engine will be loaded on first use)")
    
    def _init_ocr(self):
        """初始化OCR引擎（延迟加载）"""
        if self._ocr_initialized:
            return
        
        with self.lock:
            if self._ocr_initialized:
                return
            
            try:
                from paddleocr import PaddleOCR
                
                print("[PaddleOCRParser] Initializing PaddleOCR...")
                self.ocr_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang='ch',
                    use_gpu=self.use_gpu,
                    show_log=False,
                    det_model_dir=f'{self.model_dir}/det',
                    rec_model_dir=f'{self.model_dir}/rec',
                    cls_model_dir=f'{self.model_dir}/cls'
                )
                self._ocr_initialized = True
                print("[PaddleOCRParser] OCR engine initialized successfully")
                
            except ImportError:
                print("[PaddleOCRParser] PaddleOCR not installed, falling back to image analysis")
                self._ocr_initialized = True
            except Exception as e:
                print(f"[PaddleOCRParser] OCR initialization failed: {e}")
                self._ocr_initialized = True
    
    def _crop_roi(self, image: np.ndarray, roi: Tuple[float, float, float, float]) -> np.ndarray:
        """裁剪ROI区域"""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = roi
        x1, y1 = int(x1 * w), int(y1 * h)
        x2, y2 = int(x2 * w), int(y2 * h)
        return image[y1:y2, x1:x2]
    
    def _parse_bar_percentage(self, bar_image: np.ndarray) -> float:
        """解析血条/蓝条百分比"""
        if bar_image.size == 0:
            return 1.0
        
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(bar_image, cv2.COLOR_BGR2HSV)
        
        # 定义绿色范围（血条）
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        
        # 定义蓝色范围（蓝条）
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # 检测绿色和蓝色
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask = cv2.bitwise_or(mask_green, mask_blue)
        
        # 计算填充比例
        total_pixels = mask.shape[1]
        filled_pixels = np.sum(mask > 0, axis=1)
        max_filled = np.max(filled_pixels) if len(filled_pixels) > 0 else 0
        
        return max_filled / total_pixels if total_pixels > 0 else 1.0
    
    def _parse_text(self, image: np.ndarray) -> str:
        """使用OCR识别文本"""
        if image.size == 0:
            return ""
        
        self._init_ocr()
        
        if self.ocr_engine is None:
            return ""
        
        try:
            result = self.ocr_engine.ocr(image, cls=True)
            if result and len(result) > 0 and result[0]:
                texts = [line[1][0] for line in result[0]]
                return " ".join(texts)
        except Exception as e:
            print(f"[PaddleOCRParser] OCR error: {e}")
        
        return ""
    
    def _parse_number(self, text: str) -> int:
        """从文本中提取数字"""
        import re
        numbers = re.findall(r'\d+', text)
        return int(numbers[0]) if numbers else 0
    
    def _parse_skill_cooldown(self, skill_image: np.ndarray) -> float:
        """解析技能冷却时间"""
        if skill_image.size == 0:
            return 0.0
        
        # 检测灰色遮罩（冷却状态）
        gray = cv2.cvtColor(skill_image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # 如果平均亮度较低，说明有灰色遮罩（冷却中）
        if mean_brightness < 80:
            # 尝试识别冷却数字
            text = self._parse_text(skill_image)
            cooldown = self._parse_number(text)
            return float(cooldown)
        
        return 0.0
    
    def _parse_minimap(self, minimap_image: np.ndarray) -> Dict[str, list]:
        """解析小地图信息"""
        if minimap_image.size == 0:
            return {'allies': [], 'enemies': []}
        
        result = {'allies': [], 'enemies': []}
        
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(minimap_image, cv2.COLOR_BGR2HSV)
        
        # 绿色范围（友方）
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        
        # 红色范围（敌方）
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        
        # 检测友方
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_green:
            if cv2.contourArea(cnt) > 10:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    result['allies'].append((cx, cy))
        
        # 检测敌方
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_red:
            if cv2.contourArea(cnt) > 10:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    result['enemies'].append((cx, cy))
        
        return result
    
    def parse(self, image: np.ndarray) -> Dict:
        """
        解析游戏状态
        
        Args:
            image: 游戏画面（BGR格式）
        
        Returns:
            包含游戏状态的字典
        """
        if image is None or image.size == 0:
            return self._get_default_state()
        
        state = {}
        
        try:
            # 解析血量
            hp_bar = self._crop_roi(image, self.roi_config['hp_bar'])
            hp_text = self._crop_roi(image, self.roi_config['hp_text'])
            state['hp_percent'] = self._parse_bar_percentage(hp_bar)
            state['hp_text'] = self._parse_text(hp_text)
            
            # 解析蓝量
            mp_bar = self._crop_roi(image, self.roi_config['mp_bar'])
            mp_text = self._crop_roi(image, self.roi_config['mp_text'])
            state['mp_percent'] = self._parse_bar_percentage(mp_bar)
            state['mp_text'] = self._parse_text(mp_text)
            state['mana_enough'] = state['mp_percent'] > 0.3
            
            # 解析技能冷却
            state['skill_cd'] = {}
            for i in range(1, 4):
                skill_key = f'skill_{i}'
                skill_img = self._crop_roi(image, self.roi_config[skill_key])
                state['skill_cd'][i] = self._parse_skill_cooldown(skill_img)
            
            # 解析召唤师技能
            state['summoner_cd'] = {}
            for i in range(1, 3):
                summoner_key = f'summoner_{i}'
                summoner_img = self._crop_roi(image, self.roi_config[summoner_key])
                state['summoner_cd'][f'skill_{i}'] = self._parse_skill_cooldown(summoner_img)
            
            # 解析装备技能
            state['item_cd'] = 0.0  # 简化处理
            
            # 解析小地图
            minimap = self._crop_roi(image, self.roi_config['minimap'])
            state['minimap'] = self._parse_minimap(minimap)
            
            # 解析金币和等级
            gold_img = self._crop_roi(image, self.roi_config['gold'])
            level_img = self._crop_roi(image, self.roi_config['level'])
            state['gold'] = self._parse_number(self._parse_text(gold_img))
            state['level'] = self._parse_number(self._parse_text(level_img))
            
            # 判断是否在施法中
            state['is_casting'] = False  # 需要更复杂的检测
            
            # 判断是否在冷却中
            state['on_cooldown'] = any([v > 0 for v in state['skill_cd'].values()])
            
        except Exception as e:
            print(f"[PaddleOCRParser] Parse error: {e}")
            return self._get_default_state()
        
        return state
    
    def _get_default_state(self) -> Dict:
        """返回默认状态"""
        return {
            'hp_percent': 1.0,
            'hp_text': '',
            'mp_percent': 1.0,
            'mp_text': '',
            'mana_enough': True,
            'skill_cd': {1: 0, 2: 0, 3: 0},
            'summoner_cd': {'skill_1': 0, 'skill_2': 0},
            'item_cd': 0,
            'minimap': {'allies': [], 'enemies': []},
            'gold': 0,
            'level': 1,
            'is_casting': False,
            'on_cooldown': False
        }


class PaddleOCRManager:
    """PaddleOCR解析器管理器（单例模式）"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, use_gpu=False, model_dir=None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.parser = PaddleOCRParser(use_gpu, model_dir)
        return cls._instance
    
    @classmethod
    def get_instance(cls, use_gpu=False, model_dir=None):
        """获取单例实例"""
        return cls(use_gpu, model_dir)
    
    def parse(self, image: np.ndarray) -> Dict:
        """解析游戏状态"""
        return self.parser.parse(image)