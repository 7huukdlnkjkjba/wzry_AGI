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

**进化路线图（已完成）：**

| 组件 | 原始版本 | 改进方案 | 状态 |
|------|----------|----------|------|
| 训练方式 | 单机单环境 | **分布式（RL Learner + AI Server）** | ✅ |
| 状态输入 | 单帧图像 | **图像 + LSTM时序 + 游戏状态解析** | ✅ |
| 推理速度 | PyTorch CPU/GPU | **FeatherCNN加速（真实库支持）** | ✅ |
| 算法 | DQN | **dual-PPO** | ✅ |
| 动作处理 | 无约束 | **action mask** | ✅ |
| 奖励 | 简单攻击指示器 | **击杀/补刀/推塔等精细奖励** | ✅ |
| 游戏状态解析 | 模拟默认值 | **PaddleOCR真实解析** | ✅ |

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
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│  Game Env          │  │  Game Env          │  │  Game Env          │
│  + PaddleOCR       │  │  + PaddleOCR       │  │  + PaddleOCR       │
│  + FeatherCNN      │  │  + FeatherCNN      │  │  + FeatherCNN      │
│  + RewardSystem    │  │  + RewardSystem    │  │  + RewardSystem    │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
```

### 2.2 核心组件

| 组件 | 说明 | 文件 |
|------|------|------|
| **RL Learner** | 分布式训练环境，并行梯度累积，同步更新策略 | `rl_learner.py` |
| **AI Server** | 游戏环境交互，收集数据，执行动作 | `ai_server.py` |
| **Dispatch Module** | 数据收集分发，策略广播 | `dispatch_module.py` |
| **Memory Pool** | 内存高效循环队列，支持时间戳索引 | `memory.py` |
| **FeatherCNN** | 腾讯开源快速推断库接口（真实库支持） | `feather_cnn.py` |
| **PaddleOCR Parser** | PaddleOCR游戏状态解析（血量/冷却/小地图） | `paddleocr_parser.py` |
| **Game State Parser** | 游戏状态解析（备用） | `game_state_parser.py` |
| **Action Mask** | 专家规则过滤不合理动作 | `action_mask.py` |
| **Reward System** | 可扩展多维度奖励系统（完全集成） | `reward_system.py` |

### 2.3 dual-PPO算法

改进的PPO损失函数，在优势函数为负时添加下界约束：

```
L_dual(θ) = E_t[max(min(r_t(θ)Â_t, clip(r_t(θ),1−ε,1+ε)Â_t), cÂ_t)]
```

**效果**：对于坏动作（优势为负），防止策略更新幅度过大，训练更稳定。

### 2.4 PaddleOCR游戏状态解析

使用PaddleOCR识别游戏画面中的关键信息：

- **血量/蓝量**：颜色分析 + OCR数值识别
- **技能冷却**：灰色遮罩检测 + 数字识别
- **召唤师技能**：闪现、治疗等冷却时间
- **小地图**：友方/敌方位置检测
- **金币/等级**：OCR文字识别

### 2.5 Reward System奖励系统

多维度奖励计算：

| 奖励类型 | 权重 | 说明 |
|----------|------|------|
| 击杀 | 10.0 | 击杀敌方英雄 |
| 助攻 | 3.0 | 协助击杀 |
| 死亡 | -5.0 | 死亡惩罚 |
| 补刀 | 1.0 | 击杀小兵 |
| 推塔 | 20.0 | 摧毁防御塔 |
| 伤害 | 0.01 | 造成伤害 |
| 生存 | 0.01 | 每帧生存奖励 |

## 三、环境配置教程

### 3.1 基础环境

```bash
# 1. 创建conda环境
conda create --name wzry_ai python=3.10

# 2. 激活环境
conda activate wzry_ai

# 3. 安装基础依赖
pip install -r doc/requirements.txt

# 4. 安装PyTorch (CUDA 11.8)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. 安装ONNX Runtime (CUDA 11)
pip install onnxruntime-gpu

# CUDA 12 用户
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# 6. 安装PaddleOCR（用于游戏状态解析）
pip install paddleocr>=2.7.0
pip install paddlepaddle>=2.5.0
```

### 3.2 zlibwapi.dll 问题解决

如果出现 `Could not locate zlibwapi.dll` 错误：

```bash
# 复制文件
cp "C:\Program Files\NVIDIA Corporation\Nsight Systems 2022.4.2\host-windows-x64\zlib.dll" ^
   "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\zlibwapi.dll"
```

### 3.3 FeatherCNN编译（可选，用于推理加速）

```bash
# 克隆仓库
git clone https://github.com/Tencent/FeatherCNN.git

# 编译
cd FeatherCNN
mkdir build && cd build
cmake .. -DUSE_OPENMP=ON
make -j4

# 复制库文件到项目根目录
cp lib/FeatherCNN.dll ../  # Windows
# 或 cp lib/libfeathercnn.so ../  # Linux
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

# 使用PaddleOCR解析游戏状态
python train_distributed.py --num_servers 4 --use_paddleocr --paddleocr_gpu

# 使用FeatherCNN加速推理
python train_distributed.py --num_servers 4 --use_feather_cnn --feather_threads 2

# 完整配置（推荐）
python train_distributed.py \
    --num_servers 4 \
    --min_gradients 1 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --gamma 0.99 \
    --clip_param 0.2 \
    --dual_ppo_c 0.2 \
    --ppo_epochs 10 \
    --use_paddleocr \
    --paddleocr_gpu \
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
| `--use_paddleocr` | 使用PaddleOCR解析游戏状态 | False |
| `--paddleocr_gpu` | 使用GPU加速PaddleOCR | False |
| `--paddleocr_model_dir` | PaddleOCR模型目录 | models/ocr |

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
│   └── ocr/                # PaddleOCR模型目录（可选）
├── scrcpy-win64-v2.0/      # 投屏工具
├── images/                 # 文档图片
├── doc/                    # 文档
│   ├── command.txt
│   ├── requirements.txt
│   └── 说明文档.md
├── action_mask.py          # Action Mask机制
├── reward_system.py        # 奖励系统（完全集成）
├── net_ppo.py              # PPO Actor-Critic网络
├── ppoAgent.py             # PPO代理（含dual-PPO）
├── rl_learner.py           # 分布式RL Learner
├── ai_server.py            # AI Server组件（集成OCR+奖励）
├── dispatch_module.py      # 数据分发模块
├── feather_cnn.py          # FeatherCNN推理接口（真实库支持）
├── paddleocr_parser.py     # PaddleOCR游戏状态解析器
├── game_state_parser.py    # 游戏状态解析器（备用）
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

## 八、功能特性

### 8.1 已实现

- ✅ 分布式训练架构（RL Learner + AI Server）
- ✅ dual-PPO算法（稳定训练）
- ✅ Action Mask机制
- ✅ 可扩展奖励系统（击杀/助攻/推塔/伤害等）
- ✅ PaddleOCR游戏状态解析（血量/冷却/小地图）
- ✅ FeatherCNN推理加速（真实库支持 + ONNX Runtime回退）
- ✅ ONNX模型导出和加载
- ✅ 详细的训练监控和统计

### 8.2 未来计划

- [ ] 可视化训练监控面板
- [ ] 多智能体协作模式
- [ ] 更多英雄支持
- [ ] 自适应奖励权重调整
- [ ] 模型量化压缩

## 九、贡献

欢迎提交Issue和PR！

---

**注：开源不易，共同努力。**