<p align="center">
    <a href="https://github.com/myBoris/wzry_ai">
        <img src="https://socialify.git.ci/myBoris/wzry_ai/image?description=1&font=Rokkitt&language=1&name=1&owner=1&theme=Auto" alt="wzry_ai"/>    
    </a>
</p>

<p align="center">
    <a href="https://github.com/myBoris/wzry_ai/stargazers">
        <img src="https://img.shields.io/github/stars/myBoris/wzry_ai?style=flat-square&label=STARS&color=%23dfb317" alt="stars">
    </a>
    <a href="https://github.com/myBoris/wzry_ai/network/members">
        <img src="https://img.shields.io/github/forks/myBoris/wzry_ai?style=flat-square&label=FORKS&color=%2397ca00" alt="forks">
    </a>
    <a href="https://github.com/myBoris/wzry_ai/issues">
        <img src="https://img.shields.io/github/issues/myBoris/wzry_ai?style=flat-square&label=ISSUES&color=%23007ec6" alt="issues">
    </a>
    <a href="https://github.com/myBoris/wzry_ai/pulls">
        <img src="https://img.shields.io/github/issues-pr/myBoris/wzry_ai?style=flat-square&label=PULLS&color=%23fe7d37" alt="pulls">
    </a>
</p>

---

>声明:本人二次微调，已与wzry_ai项目无关，但某人用该项目AI外挂的话，本人只是微调，请找wzry_ai项目作者喝茶。

## 一、项目简介

这是一个开源的人工智能模型玩王者荣耀的项目，采用**分布式强化学习架构**，目标是将AI实力从"青铜"提升到"王者百星"级别。

**进化路线图：**

| 组件 | 原始版本 | 改进方案 | 状态 |
|------|----------|----------|------|
| 训练方式 | 单机单环境 | **分布式（RL Learner + AI Server）** | ✅ |
| 状态输入 | 单帧图像 | **图像 + LSTM时序 + 游戏状态解析** | ✅ |
| 推理速度 | PyTorch CPU/GPU | **FeatherCNN加速** | ✅ |
| 算法 | DQN | **dual-PPO** | ✅ |
| 动作处理 | 无约束 | **action mask** | ✅ |
| 奖励 | 简单攻击指示器 | **击杀/补刀/推塔等精细奖励** | ✅ |

```
文章网址: https://stack-traceable.top/

环境安装详细教程: doc/说明文档.md
环境安装视频: https://www.bilibili.com/video/BV1ZXYuePEUG/
```

## 二、架构设计

### 2.1 分布式架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        RL Learner                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Policy Network  │  dual-PPO Loss  │  Gradient Sync    │    │
│  └─────────────────────────────────────────────────────────┘    │
└───────────────────────────┬─────────────────────────────────────┘
                            │ 发送策略
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Dispatch Module                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Data Compression  │  Data Distribution  │  Policy Broadcast│  │
│  └─────────────────────────────────────────────────────────┘    │
└───────────────────────────┬─────────────────────────────────────┘
     ▲                      ▲                      ▲
     │ 发送数据             │ 发送数据             │ 发送数据
┌────┴────┐          ┌──────┴──────┐          ┌──────┴──────┐
│ AI Server│          │ AI Server  │          │ AI Server  │
│   SRV-001│          │   SRV-002  │          │   SRV-NNN  │
└────┬────┘          └──────┬──────┘          └──────┬──────┘
     │                      │                      │
     ▼                      ▼                      ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Game Env    │    │  Game Env    │    │  Game Env    │
│  + FeatherCNN│    │  + FeatherCNN│    │  + FeatherCNN│
└───────────────┘    └───────────────┘    └───────────────┘
```

### 2.2 核心组件

| 组件 | 说明 | 文件 |
|------|------|------|
| **RL Learner** | 分布式训练环境，并行梯度累积，同步更新策略 | `rl_learner.py` |
| **AI Server** | 游戏环境交互，收集数据，执行动作 | `ai_server.py` |
| **Dispatch Module** | 数据收集分发，策略广播 | `dispatch_module.py` |
| **Memory Pool** | 内存高效循环队列，支持时间戳索引 | `memory.py` |
| **FeatherCNN** | 腾讯开源快速推断库接口 | `feather_cnn.py` |
| **Game State Parser** | 游戏状态解析（血量、蓝量、冷却等） | `game_state_parser.py` |
| **Action Mask** | 专家规则过滤不合理动作 | `action_mask.py` |
| **Reward System** | 可扩展多维度奖励系统 | `reward_system.py` |

### 2.3 dual-PPO算法

改进的PPO损失函数，在优势函数为负时添加下界约束：

```
L_dual(θ) = E_t[max(min(r_t(θ)Â_t, clip(r_t(θ),1−ε,1+ε)Â_t), cÂ_t)]
```

**效果**：对于坏动作（优势为负），防止策略更新幅度过大，训练更稳定。

## 三、环境配置教程

### 3.1 基础环境

```bash
# 1. 创建conda环境
conda create --name wzry_ai python=3.10

# 2. 激活环境
conda activate wzry_ai

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装PyTorch (CUDA 11.8)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. 安装ONNX Runtime (CUDA 11)
pip install onnxruntime-gpu

# CUDA 12 用户
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

### 3.2 zlibwapi.dll 问题解决

如果出现 `Could not locate zlibwapi.dll` 错误：

```bash
# 复制文件
cp "C:\Program Files\NVIDIA Corporation\Nsight Systems 2022.4.2\host-windows-x64\zlib.dll" ^
   "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\zlibwapi.dll"
```

## 四、训练教程

### 4.1 单机训练（保留原有方式）

```bash
# 运行单机训练
python train.py
```

### 4.2 分布式训练（推荐）

```bash
# 基本用法
python train_distributed.py --num_servers 4

# 使用FeatherCNN加速推理
python train_distributed.py --num_servers 4 --use_feather_cnn --feather_threads 2

# 完整参数
python train_distributed.py \
    --num_servers 4 \
    --min_gradients 1 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --gamma 0.99 \
    --clip_param 0.2 \
    --dual_ppo_c 0.2 \
    --ppo_epochs 10 \
    --use_feather_cnn \
    --feather_threads 2
```

### 4.3 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num_servers` | AI Server数量 | 1 |
| `--min_gradients` | 最小梯度数更新 | 1 |
| `--batch_size` | 批次大小 | 64 |
| `--learning_rate` | 学习率 | 0.001 |
| `--gamma` | 折扣因子 | 0.99 |
| `--clip_param` | PPO裁剪参数 | 0.2 |
| `--dual_ppo_c` | dual-PPO下界参数 | 0.2 |
| `--ppo_epochs` | PPO迭代次数 | 10 |
| `--use_feather_cnn` | 使用FeatherCNN加速 | False |
| `--feather_threads` | FeatherCNN线程数 | 1 |

## 五、模型下载

```bash
# 下载预训练模型
# 网址: https://stack-traceable.top/archives/wzry-ai-model

# 将ONNX模型放在models目录下
# 生成的模型会保存在src目录下
```

## 六、按键映射配置

### 6.1 修改按键位置

编辑 `argparses.py` 文件，修改以下配置：

```python
# 移动坐标
move_actions_detail = {
    1: {'action_name': '移动', 'position': (0.164, 0.798), 'radius': 200}
}

# 点击坐标（购买装备、信号、升级技能等）
info_actions_detail = {...}

# 攻击动作（攻击、技能、召唤师技能等）
attack_actions_detail = {...}
```

### 6.2 可视化位置工具

```bash
# 运行位置显示工具
python showposition.py

# 点击图片上的位置，获取百分比坐标
# 将结果填入argparses.py中
```

## 七、项目结构

```
wzry_ai-main/
├── src/                    # 模型输出目录
├── models/                 # ONNX模型目录
├── scrcpy-win64-v2.0/      # 投屏工具
├── images/                 # 文档图片
├── doc/                    # 文档
│   ├── command.txt
│   ├── requirements.txt
│   └── 说明文档.md
├── action_mask.py          # Action Mask机制
├── reward_system.py        # 奖励系统
├── net_ppo.py              # PPO Actor-Critic网络
├── ppoAgent.py             # PPO代理（含dual-PPO）
├── rl_learner.py           # 分布式RL Learner
├── ai_server.py            # AI Server组件
├── dispatch_module.py      # 数据分发模块
├── feather_cnn.py          # FeatherCNN推理接口
├── game_state_parser.py    # 游戏状态解析器
├── memory.py               # 内存池
├── train.py                # 单机训练脚本
├── train_distributed.py    # 分布式训练脚本
├── android_tool.py         # Android工具
├── wzry_env.py             # 游戏环境
├── argparses.py            # 参数配置
├── globalInfo.py           # 全局状态
├── getReword.py            # 奖励计算
├── onnxRunner.py           # ONNX运行器
└── README.md               # 项目说明
```

## 八、未来计划

- [ ] 集成FeatherCNN原生库加速
- [ ] 实现完整的游戏状态解析（小地图、Buff等）
- [ ] 添加更多英雄支持
- [ ] 实现多智能体协作
- [ ] 可视化训练监控

## 九、贡献

欢迎提交Issue和PR！

---

**注：开源不易，共同努力。**