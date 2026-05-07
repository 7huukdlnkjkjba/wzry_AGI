import argparse
import os
import threading
import time

import torch

from ai_server import AIServer
from android_tool import AndroidTool
from argparses import args, device
from dispatch_module import DispatchModule
from feather_cnn import FeatherCNNManager
from getReword import GetRewordUtil
from globalInfo import GlobalInfo
from onnxRunner import OnnxRunner
from ppoAgent import PPOAgent
from rl_learner import RLLearner
from wzry_env import Environment


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Distributed RL Training for wzry_ai')
    
    # 分布式相关参数
    parser.add_argument('--num_servers', type=int, default=1, help='Number of AI Servers')
    parser.add_argument('--min_gradients', type=int, default=1, help='Minimum gradients for update')
    parser.add_argument('--sync_interval', type=int, default=10, help='Sync interval in steps')
    
    # 原有参数
    parser.add_argument('--device_id', type=str, default='cuda:0', help='Device ID')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--clip_param', type=float, default=0.2, help='PPO clip parameter')
    parser.add_argument('--dual_ppo_c', type=float, default=0.2, help='dual-PPO c parameter')
    parser.add_argument('--ppo_epochs', type=int, default=10, help='PPO epochs')
    
    # FeatherCNN相关参数
    parser.add_argument('--use_feather_cnn', action='store_true', help='Use FeatherCNN for inference')
    parser.add_argument('--feather_threads', type=int, default=1, help='Number of threads for FeatherCNN')
    parser.add_argument('--onnx_model_path', type=str, default='models/ppo_policy.onnx', help='ONNX model path')
    
    return parser.parse_args()


def main():
    """主训练函数"""
    # 解析参数
    dist_args = parse_args()
    
    # 初始化全局状态
    globalInfo = GlobalInfo()
    
    # 初始化FeatherCNN管理器
    feather_manager = FeatherCNNManager.get_instance()
    
    # 创建共享的PPO Agent（用于策略推断）
    shared_agent = PPOAgent(use_feather_cnn=dist_args.use_feather_cnn)
    
    # 如果使用FeatherCNN，导出ONNX模型
    if dist_args.use_feather_cnn:
        print("[Main] Using FeatherCNN for inference acceleration")
        
        # 导出ONNX模型
        if not os.path.exists(dist_args.onnx_model_path):
            print(f"[Main] Exporting model to ONNX: {dist_args.onnx_model_path}")
            shared_agent.export_to_onnx(dist_args.onnx_model_path)
        
        # 加载模型到FeatherCNN
        feather_manager.load_model('ppo_policy', dist_args.onnx_model_path)
        feather_manager.set_default_threads(dist_args.feather_threads)
        
        # 性能基准测试
        feather_manager.benchmark_all()
    
    # 初始化RL Learner
    learner = RLLearner()
    
    # 初始化Dispatch Module
    dispatch = DispatchModule.get_instance()
    dispatch.set_learner(learner)
    
    # 初始化游戏工具
    rewordUtil = GetRewordUtil()
    start_check = OnnxRunner('models/start.onnx', classes=['started'])
    
    # 创建多个AI Server
    servers = []
    for i in range(dist_args.num_servers):
        server = AIServer(server_id=f'SRV-{i:03d}')
        
        # 创建独立的游戏环境
        tool = AndroidTool()
        if i == 0:
            tool.show_scrcpy()  # 只在第一个服务器显示画面
        
        env = Environment(tool, rewordUtil)
        
        # 初始化服务器
        server.initialize(tool, env, rewordUtil, shared_agent, start_check)
        
        # 注册到Dispatch Module
        dispatch.register_server(server)
        
        servers.append(server)
    
    print(f"[Main] Created {len(servers)} AI Servers")
    
    # 启动所有AI Server
    for server in servers:
        server.start()
        time.sleep(0.1)  # 避免同时启动导致资源竞争
    
    print("[Main] All AI Servers started")
    
    # 启动监控线程
    monitor_thread = threading.Thread(target=monitor_loop, args=(learner, dispatch, servers), daemon=True)
    monitor_thread.start()
    
    # 主循环
    try:
        while True:
            # 定期保存模型
            if learner.update_count > 0 and learner.update_count % 100 == 0:
                learner.save_model()
            
            # 广播策略
            learner.broadcast_policy()
            
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("[Main] Received shutdown signal")
    
    # 停止所有组件
    print("[Main] Stopping components...")
    
    for server in servers:
        server.stop()
    
    dispatch.stop()
    learner.stop()
    
    print("[Main] All components stopped")


def monitor_loop(learner, dispatch, servers):
    """监控循环"""
    while True:
        # 获取状态信息
        learner_stats = learner.get_stats()
        dispatch_stats = dispatch.get_status()
        
        print("\n" + "="*60)
        print(f"[Monitor] Policy Version: {learner_stats['policy_version']}")
        print(f"[Monitor] Updates: {learner_stats['update_count']}")
        print(f"[Monitor] Total Samples: {learner_stats['total_samples']}")
        print(f"[Monitor] Active Servers: {dispatch_stats['num_servers']}")
        print(f"[Monitor] Data Buffer: {dispatch_stats['buffer_size']}")
        print(f"[Monitor] Compression Ratio: {dispatch_stats['compression_ratio']}")
        
        # 打印各服务器状态
        for server in servers:
            status = server.get_status()
            print(f"  Server {status['server_id']}: "
                  f"Running={status['is_running']}, "
                  f"Samples={status['samples_collected']}, "
                  f"Episodes={status['episodes_completed']}")
        
        print("="*60)
        
        time.sleep(5)


if __name__ == '__main__':
    main()