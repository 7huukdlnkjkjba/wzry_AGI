import json
import threading
import time
from collections import deque

import zlib


class DispatchModule:
    """
    Dispatch Module - 数据收集和分发中心
    
    负责：
    - 从多个AI Server收集数据
    - 压缩、打包数据
    - 将数据传送到Memory Pool
    - 将策略从RL Learner分发给AI Server
    
    采用单例模式
    """
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = DispatchModule()
        return cls._instance
    
    def __init__(self):
        # AI Server列表
        self.servers = {}
        
        # 数据缓冲区
        self.data_buffer = deque(maxlen=100000)
        
        # 数据处理线程
        self.is_running = True
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
        # 策略分发线程
        self.policy_thread = threading.Thread(target=self._policy_distribution_loop, daemon=True)
        self.policy_thread.start()
        
        # RL Learner引用
        self.learner = None
        
        # 统计信息
        self.total_data_received = 0
        self.total_data_sent = 0
        self.compression_ratio = 0.0
        
        print("[Dispatch Module] Initialized")
    
    def register_server(self, server):
        """注册AI Server"""
        server_id = server.server_id
        self.servers[server_id] = server
        server.on_data_ready = self._on_server_data_ready
        print(f"[Dispatch Module] Registered AI Server: {server_id}")
    
    def unregister_server(self, server_id):
        """注销AI Server"""
        if server_id in self.servers:
            del self.servers[server_id]
            print(f"[Dispatch Module] Unregistered AI Server: {server_id}")
    
    def _on_server_data_ready(self, data):
        """处理来自AI Server的数据"""
        # 压缩数据
        compressed_data = self._compress_data(data)
        
        # 添加到缓冲区
        with self._lock:
            self.data_buffer.append(compressed_data)
        
        self.total_data_received += len(data)
    
    def _compress_data(self, data):
        """压缩数据"""
        # 转换为JSON字符串
        json_str = json.dumps(data)
        
        # 使用zlib压缩
        compressed = zlib.compress(json_str.encode('utf-8'))
        
        # 计算压缩率
        original_size = len(json_str.encode('utf-8'))
        compressed_size = len(compressed)
        if original_size > 0:
            self.compression_ratio = (1 - compressed_size / original_size) * 100
        
        return compressed
    
    def _decompress_data(self, compressed_data):
        """解压数据"""
        try:
            decompressed = zlib.decompress(compressed_data)
            return json.loads(decompressed.decode('utf-8'))
        except Exception as e:
            print(f"[Dispatch Module] Decompression error: {e}")
            return None
    
    def _process_loop(self):
        """数据处理主循环"""
        while self.is_running:
            # 检查是否有数据需要处理
            if not self.data_buffer:
                time.sleep(0.01)
                continue
            
            # 批量处理数据
            batch_size = min(100, len(self.data_buffer))
            batch = []
            
            with self._lock:
                for _ in range(batch_size):
                    compressed_data = self.data_buffer.popleft()
                    data = self._decompress_data(compressed_data)
                    if data:
                        batch.extend(data)
            
            # 发送到Memory Pool
            self._send_to_memory_pool(batch)
            
            self.total_data_sent += len(batch)
    
    def _send_to_memory_pool(self, data):
        """发送数据到Memory Pool"""
        from globalInfo import globalInfo
        
        for item in data:
            globalInfo.store_transition_ppo(
                item['state'],
                item['action'],
                item['log_prob'],
                item['reward'],
                item['value'],
                item['next_value'],
                item['done']
            )
    
    def _policy_distribution_loop(self):
        """策略分发循环"""
        last_version = -1
        
        while self.is_running:
            if self.learner is None:
                time.sleep(1)
                continue
            
            # 获取最新策略
            policy = self.learner.get_policy()
            
            # 如果策略有更新，分发到所有服务器
            if policy['version'] > last_version:
                self._broadcast_policy(policy)
                last_version = policy['version']
            
            time.sleep(0.5)
    
    def _broadcast_policy(self, policy):
        """广播策略到所有AI Server"""
        for server_id, server in self.servers.items():
            try:
                server.update_policy(policy)
            except Exception as e:
                print(f"[Dispatch Module] Failed to update policy for server {server_id}: {e}")
    
    def broadcast_policy(self, policy):
        """外部调用的策略广播方法"""
        self._broadcast_policy(policy)
    
    def set_learner(self, learner):
        """设置RL Learner引用"""
        self.learner = learner
    
    def get_status(self):
        """获取状态信息"""
        return {
            'num_servers': len(self.servers),
            'buffer_size': len(self.data_buffer),
            'total_data_received': self.total_data_received,
            'total_data_sent': self.total_data_sent,
            'compression_ratio': f"{self.compression_ratio:.2f}%"
        }
    
    def stop(self):
        """停止Dispatch Module"""
        self.is_running = False
        if self.process_thread.is_alive():
            self.process_thread.join()
        if self.policy_thread.is_alive():
            self.policy_thread.join()
        print("[Dispatch Module] Stopped")
