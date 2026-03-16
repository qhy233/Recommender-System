# RecoGym 中文详细说明文档

## 一、项目简介

### 1.1 什么是RecoGym？

RecoGym是一个基于OpenAI Gym框架的推荐系统强化学习实验环境，由Criteo研究团队开发，发表于RecSys 2018 REVEAL workshop。

**论文信息：**

- 标题：RecoGym: A Reinforcement Learning Environment for the problem of Product Recommendation in Online Advertising
- 作者：David Rohde, Stephen Bonner, Travis Dunlop, Flavian Vasile, Alexandros Karatzoglou
- 论文链接：https://arxiv.org/abs/1808.00720

### 1.2 项目目标

RecoGym旨在解决推荐系统研究中的以下问题：

1. **离线与在线评估的差距**：传统推荐系统研究多采用离线评估，但离线指标与在线效果往往不一致
2. **强化学习在推荐系统中的应用**：为推荐系统和强化学习社区提供合作平台
3. **算法验证环境**：提供一个快速验证推荐算法的模拟环境

### 1.3 核心特点

| 特点           | 说明                             |
| -------------- | -------------------------------- |
| OpenAI Gym兼容 | 遵循标准强化学习环境接口         |
| 用户行为模拟   | 模拟真实电商用户浏览和点击行为   |
| 离线+在线学习  | 支持离线预训练和在线学习两种模式 |
| 多种基线Agent  | 内置多种经典推荐算法实现         |
| 可扩展性强     | 支持自定义Agent开发              |

## 二、项目结构

```
reco-gym/
├── recogym/                      # 核心库代码
│   ├── envs/                     # 环境实现
│   │   ├── reco_env_v0.py       # 环境版本0（简化版）
│   │   ├── reco_env_v1.py       # 环境版本1（完整版）
│   │   ├── abstract.py          # 环境基类
│   │   ├── observation.py       # 观察数据结构
│   │   ├── configuration.py     # 配置管理
│   │   ├── context.py           # 上下文信息
│   │   ├── session.py           # 会话数据结构
│   │   └── features/            # 特征生成
│   │       └── time/            # 时间特征
│   ├── agents/                   # Agent实现
│   │   ├── abstract.py          # Agent基类
│   │   ├── organic_count.py     # 基于有机浏览统计
│   │   ├── bandit_count.py      # 基于Bandit反馈统计
│   │   ├── epsilon_greedy.py    # ε-greedy探索
│   │   ├── bandit_mf.py         # 矩阵分解
│   │   ├── organic_mf.py        # 有机矩阵分解
│   │   ├── logreg_ips.py       # 逻辑回归+IPS
│   │   ├── logreg_poly.py      # 多项式逻辑回归
│   │   ├── nn_ips.py           # 神经网络+IPS
│   │   ├── bayesian_poly.py    # 贝叶斯多项式
│   │   ├── pytorch_mlr.py      # PyTorch混合逻辑回归
│   │   └── random_agent.py     # 随机Agent
│   ├── evaluate_agent.py         # 评估工具
│   ├── bench_agents.py          # 基准测试
│   └── competition.py           # 竞赛评分
│
├── leaderboard_entries/          # 排行榜Agent示例
│   ├── organic_count.py         # 有机统计Agent
│   ├── bandit_count.py          # Bandit统计Agent
│   ├── likelihood.py            # 似然模型Agent
│   ├── pytorch_likelihood.py    # PyTorch似然模型
│   ├── context_bandit.py        # 上下文Bandit
│   └── ...
│
├── my_entries/                   # 自定义实验目录
│   ├── experiments.py           # 实验代码
│   ├── results.png              # 实验结果图
│   └── 实验说明.md              # 实验说明文档
│
├── course_slides/                # 教程幻灯片
│   ├── images/                  # 图片资源
│   └── *.tex                    # LaTeX源文件
│
├── images/                       # 文档图片
│
├── *.ipynb                       # Jupyter教程笔记本
│   ├── Getting Started.ipynb    # 入门教程
│   ├── Compare Agents.ipynb     # Agent对比
│   ├── Bandit Feedback *.ipynb  # Bandit反馈系列教程
│   └── ...
│
├── sim_test.py                   # 模拟测试脚本
├── quality_test.py               # 质量测试
├── deterministic_test.py         # 确定性测试
├── setup.py                      # 安装脚本
└── environment.yml               # Conda环境配置
```

## 三、核心概念

### 3.1 Organic与Bandit会话

RecoGym引入了两个核心概念：

#### Organic Session（有机会话）

用户自然浏览行为，不涉及推荐干预。例如：

- 电商网站上用户浏览商品
- 音乐应用中用户听歌
- 新闻网站上用户阅读文章

#### Bandit Session（Bandit会话）

系统有机会向用户推荐物品并观察反馈。例如：

- 展示广告推荐
- 商品推荐位
- 内容推荐

```
┌─────────────────────────────────────────────────────────┐
│                    用户会话流程                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   Organic ──→ 用户浏览物品 ──→ Bandit ──→ 系统推荐       │
│      ↑                              │                   │
│      └────────── 点击后继续浏览 ←────┘                   │
│                                                         │
│   状态转移概率：                                        │
│   - prob_organic_to_bandit: 有机→推荐的概率             │
│   - prob_bandit_to_organic: 推荐→有机的概率             │
│   - prob_leave_organic: 有机状态下离开的概率            │
│   - prob_leave_bandit: Bandit状态下离开的概率           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 3.2 强化学习框架

RecoGym遵循标准强化学习设置：

```
┌──────────────┐         ┌──────────────┐
│              │  Action │              │
│    Agent     │ ──────→ │ Environment  │
│  (推荐算法)   │         │   (用户行为) │
│              │ ←────── │              │
└──────────────┘ Reward  └──────────────┘
                       Observation
```

| 组件        | 对应概念                   |
| ----------- | -------------------------- |
| Agent       | 推荐算法                   |
| Environment | 用户行为模拟器             |
| State       | 用户当前状态（浏览历史等） |
| Action      | 推荐的物品ID               |
| Reward      | 用户是否点击（0或1）       |
| Observation | 用户浏览会话数据           |

### 3.3 离线训练与在线评估

```
┌─────────────────────────────────────────────────────────┐
│                    学习流程                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  第一阶段：离线训练                                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │  使用固定策略生成的历史数据训练Agent              │   │
│  │  - 大量历史交互数据                               │   │
│  │  - 无需在线探索风险                               │   │
│  │  - 快速迭代验证                                   │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  第二阶段：在线评估                                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │  使用训练好的Agent进行在线推荐                    │   │
│  │  - 真实用户交互                                   │   │
│  │  - 评估实际效果                                   │   │
│  │  - 支持在线学习                                   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 四、环境配置

### 4.1 依赖要求

**核心依赖：**

```
Python >= 3.6
numpy >= 1.15
scipy >= 1.1
gym >= 0.14
pandas >= 0.25
matplotlib >= 3.1
tqdm >= 4.36
numba >= 0.45
```

**可选依赖：**

```
torch >= 1.2    # 用于神经网络Agent
scikit-learn    # 用于某些特征工程
```

### 4.2 安装方法

**方法一：使用pip安装（推荐）**

```bash
# 创建虚拟环境
conda create -n reco-gym python=3.6
conda activate reco-gym

# 安装recogym
pip install recogym==0.1.2.3
```

**方法二：从源码安装**

```bash
# 克隆仓库
git clone https://github.com/criteo-research/reco-gym.git
cd reco-gym

# 安装
pip install -e .
```

**MacOS用户额外步骤：**

```bash
brew install libomp
```

### 4.3 验证安装

```python
import recogym
print(recogym.__version__)  # 应输出: 0.1.2.3
```

## 五、快速入门

### 5.1 环境初始化

```python
import gym
import recogym
from recogym import env_1_args, Configuration

# 设置随机种子
env_1_args['random_seed'] = 42

# 创建环境
env = gym.make('reco-gym-v1')
env.init_gym(env_1_args)

# 查看配置
print(f"物品数量: {env.config.num_products}")
print(f"用户数量: {env.config.num_users}")
```

### 5.2 环境配置参数

```python
env_1_args = {
    # 基础参数
    'num_products': 10,           # 物品数量
    'num_users': 100,             # 用户数量
    'random_seed': 42,            # 随机种子
  
    # 隐因子维度
    'K': 5,                       # 用户兴趣隐因子维度
  
    # 状态转移概率
    'prob_organic_to_bandit': 0.25,  # 有机→推荐概率
    'prob_bandit_to_organic': 0.05,  # 推荐→有机概率
    'prob_leave_organic': 0.01,      # 有机状态离开概率
    'prob_leave_bandit': 0.01,       # Bandit状态离开概率
  
    # 其他参数
    'sigma_omega_initial': 1,     # 用户兴趣初始化方差
    'sigma_omega': 0.1,           # 用户兴趣更新方差
    'number_of_flips': 0,         # 有机与Bandit偏好的翻转数
}
```

### 5.3 离线模式交互

```python
# 重置环境
env.reset()

observation = None
reward = 0
done = False

while not done:
    # 环境自动生成动作（使用内置随机策略）
    action, observation, reward, done, info = env.step_offline(observation, reward, done)
  
    print(f"动作: {action}")
    print(f"观察: {observation.sessions()}")
    print(f"奖励: {reward}")
    print(f"结束: {done}")
```

### 5.4 在线模式交互

```python
# 重置环境
env.reset()

observation, _, done, _ = env.step(None)

while not done:
    # Agent自己选择动作
    action_id = 0  # 推荐物品0
    observation, reward, done, info = env.step(action_id)
  
    print(f"推荐物品: {action_id}")
    print(f"用户点击: {reward}")
```

## 六、开发自定义Agent

### 6.1 Agent基类

所有Agent需要继承 `recogym.agents.Agent`类：

```python
from recogym.agents import Agent

class MyAgent(Agent):
    def __init__(self, config):
        super(MyAgent, self).__init__(config)
        # 初始化模型参数
        self.model_params = np.zeros(config.num_products)
  
    def train(self, observation, action, reward, done=False):
        """
        训练方法：根据观察、动作、奖励更新模型
      
        参数:
            observation: 观察数据，包含用户浏览会话
            action: Agent执行的动作
            reward: 获得的奖励（0或1）
            done: 是否结束当前用户会话
        """
        if observation is not None:
            # 处理观察数据
            for session in observation.sessions():
                product_id = session['v']
                # 更新模型...
  
    def act(self, observation, reward, done):
        """
        决策方法：根据观察选择推荐动作
      
        返回:
            dict: 包含动作信息的字典
                - 't': 时间戳
                - 'u': 用户ID
                - 'a': 推荐的物品ID
                - 'ps': 选择该动作的概率
                - 'ps-a': 所有动作的概率分布（可选）
        """
        # 决策逻辑
        action_id = self._choose_action(observation)
      
        return {
            't': observation.context().time(),
            'u': observation.context().user(),
            'a': action_id,
            'ps': 1.0,  # 选择概率
            'ps-a': np.zeros(self.config.num_products)
        }
  
    def reset(self):
        """重置Agent状态"""
        pass
```

### 6.2 完整Agent示例

```python
import numpy as np
from recogym.agents import Agent

class PopularityAgent(Agent):
    """基于流行度的推荐Agent"""
  
    def __init__(self, config):
        super(PopularityAgent, self).__init__(config)
        self.product_views = np.zeros(config.num_products)
  
    def train(self, observation, action, reward, done=False):
        # 统计物品浏览次数
        if observation is not None:
            for session in observation.sessions():
                self.product_views[session['v']] += 1
  
    def act(self, observation, reward, done):
        # 推荐浏览次数最多的物品
        if np.sum(self.product_views) == 0:
            action = np.random.randint(self.config.num_products)
        else:
            action = np.argmax(self.product_views)
      
        return {
            't': observation.context().time(),
            'u': observation.context().user(),
            'a': action,
            'ps': 1.0,
            'ps-a': np.zeros(self.config.num_products)
        }
```

### 6.3 使用Model-Based架构

RecoGym提供了更高级的Model-Based Agent架构：

```python
from recogym.agents import (
    AbstractFeatureProvider,
    Model,
    ModelBasedAgent
)

class MyFeatureProvider(AbstractFeatureProvider):
    """特征提取器"""
  
    def __init__(self, config):
        super(MyFeatureProvider, self).__init__(config)
  
    def build(self):
        # 构建特征和模型
        features, actions, deltas, pss = self.train_data()
        # 返回 (feature_provider, model)
        return MyFeatureProvider(self.config), MyModel(self.config)

class MyModel(Model):
    """模型"""
  
    def __init__(self, config):
        super(MyModel, self).__init__(config)
  
    def act(self, observation, features):
        # 根据特征决策
        action_id = np.argmax(features)
        return {
            'a': action_id,
            'ps': 1.0,
            'ps-a': np.zeros(self.config.num_products)
        }

class MyModelBasedAgent(ModelBasedAgent):
    """基于模型的Agent"""
  
    def __init__(self, config):
        super(MyModelBasedAgent, self).__init__(
            config,
            MyFeatureProvider(config)
        )
```

## 七、内置Agent介绍

### 7.1 Agent列表

| Agent                    | 类型     | 说明                   |
| ------------------------ | -------- | ---------------------- |
| RandomAgent              | 基线     | 随机推荐               |
| OrganicCount             | 统计     | 基于有机浏览统计       |
| OrganicUserCount         | 统计     | 基于用户级有机浏览统计 |
| BanditCount              | 统计     | 基于Bandit反馈统计     |
| EpsilonGreedy            | 探索     | ε-greedy探索策略      |
| BanditMFSquare           | 矩阵分解 | Bandit数据矩阵分解     |
| OrganicMFSquare          | 矩阵分解 | 有机数据矩阵分解       |
| LogregPolyAgent          | 机器学习 | 多项式逻辑回归         |
| LogregMulticlassIpsAgent | 机器学习 | IPS校正逻辑回归        |
| NnIpsAgent               | 深度学习 | 神经网络+IPS           |
| BayesianAgentVB          | 贝叶斯   | 贝叶斯多项式回归       |
| PyTorchMLRAgent          | 深度学习 | PyTorch混合逻辑回归    |

### 7.2 Agent使用示例

```python
from recogym.agents import (
    OrganicCount, organic_count_args,
    BanditCount, bandit_count_args,
    EpsilonGreedy, epsilon_greedy_args,
    BanditMFSquare, bandit_mf_square_args,
)
from recogym import Configuration

# 创建Agent
agent_organic = OrganicCount(Configuration({
    **organic_count_args,
    **env_1_args,
}))

agent_bandit = BanditCount(Configuration({
    **bandit_count_args,
    **env_1_args,
}))

agent_eps_greedy = EpsilonGreedy(
    Configuration({**epsilon_greedy_args, **env_1_args}),
    agent_organic  # 包装基础Agent
)

agent_mf = BanditMFSquare(Configuration({
    **bandit_mf_square_args,
    **env_1_args,
}))
```

## 八、评估与测试

### 8.1 使用test_agent函数

```python
from recogym import test_agent
from copy import deepcopy

# 测试Agent
result = test_agent(
    env=deepcopy(env),
    agent=agent,
    num_offline_users=1000,      # 离线训练用户数
    num_online_users=100,        # 在线测试用户数
    num_organic_offline_users=0, # 仅有机数据用户数
    num_epochs=1,                # 实验轮数
)

# 返回值：(中位数CTR, 2.5%分位数, 97.5%分位数)
print(f"CTR中位数: {result[0]:.4f}")
print(f"95%置信区间: [{result[1]:.4f}, {result[2]:.4f}]")
```

### 8.2 使用evaluate_agent函数

```python
from recogym import evaluate_agent

# 评估Agent演化
stats = evaluate_agent(
    env=env,
    agent=agent,
    num_initial_train_users=100,
    num_step_users=1000,
    num_steps=10,
)

# stats包含各步骤的成功/失败统计
```

### 8.3 批量对比Agent

```python
from recogym import gather_agent_stats, plot_agent_stats

# 定义要测试的Agent
agents_init = {
    'Random': {
        AgentInit.CTOR: RandomAgent,
        AgentInit.DEF_ARGS: random_args,
    },
    'OrganicCount': {
        AgentInit.CTOR: OrganicCount,
        AgentInit.DEF_ARGS: organic_count_args,
    },
    'BanditMF': {
        AgentInit.CTOR: BanditMFSquare,
        AgentInit.DEF_ARGS: bandit_mf_square_args,
    },
}

# 收集统计信息
stats = gather_agent_stats(
    env=env,
    env_args=env_1_args,
    agents_init_data=agents_init,
    user_samples=[100, 500, 1000, 5000],
)

# 绘制对比图
plot_agent_stats(stats)
```

### 8.4 运行排行榜测试

```bash
# 运行sim_test.py测试my_entries目录下的所有Agent
python sim_test.py --entries_dir my_entries --P 10 --U 100 --Utest 100
```

## 九、Jupyter教程笔记本

### 9.1 教程列表

| 笔记本                                        | 内容                          |
| --------------------------------------------- | ----------------------------- |
| Getting Started.ipynb                         | 入门教程，介绍环境和基本Agent |
| Compare Agents.ipynb                          | Agent性能对比                 |
| Bandit Feedback - *.ipynb                     | Bandit反馈系列教程            |
| Organic vs Likelihood.ipynb                   | 有机数据与似然模型对比        |
| IPS vs Non-IPS.ipynb                          | IPS校正方法对比               |
| Offline Evaluation with Bandit Feedback.ipynb | 离线评估方法                  |
| ContextualBandits.ipynb                       | 上下文Bandit                  |
| Complex Time Behaviour.ipynb                  | 复杂时间行为                  |
| Likelihood Agents.ipynb                       | 似然模型Agent                 |

### 9.2 推荐学习路径

```
1. Getting Started.ipynb
   ↓ 理解基本概念和环境交互
   
2. Compare Agents.ipynb
   ↓ 了解不同Agent的性能差异
   
3. Bandit Feedback - Organic best of vs Bandit best of.ipynb
   ↓ 理解有机数据和Bandit数据的差异
   
4. Bandit Feedback - Likelihood based feature engineering logistic regression.ipynb
   ↓ 学习特征工程和似然模型
   
5. IPS vs Non-IPS.ipynb
   ↓ 理解IPS校正的重要性
   
6. Offline Evaluation with Bandit Feedback.ipynb
   ↓ 掌握离线评估方法
```

## 十、高级功能

### 10.1 自定义环境参数

```python
# 创建自定义环境
custom_args = {
    **env_1_args,
    'num_products': 100,           # 100个物品
    'K': 20,                       # 20维隐因子
    'prob_organic_to_bandit': 0.5, # 更高的推荐机会
    'number_of_flips': 10,         # 有机与Bandit偏好差异更大
}

env = gym.make('reco-gym-v1')
env.init_gym(custom_args)
```

### 10.2 时间特征生成

```python
from recogym.envs.features.time import NormalTimeGenerator

# 自定义时间生成器
time_gen = NormalTimeGenerator(Configuration({
    'mu': 100,    # 平均时间间隔
    'sigma': 20,  # 时间间隔标准差
}))

env_args['time_generator'] = time_gen
```

### 10.3 使用缓存加速训练

```python
# 启用缓存
result = test_agent(
    env=env,
    agent=agent,
    num_offline_users=1000,
    num_online_users=100,
    with_cache=True,  # 启用缓存
)
```

### 10.4 多进程评估

```python
# 多进程并行评估
stats = gather_agent_stats(
    env=env,
    env_args=env_1_args,
    agents_init_data=agents_init,
    user_samples=[100, 500, 1000],
    num_epochs=10,  # 多轮并行
)
```

## 十一、常见问题

### 11.1 环境相关问题

**Q: 如何设置不同的随机种子？**

```python
env_1_args['random_seed'] = 123
env.reset_random_seed(456)  # 重置随机种子
```

**Q: 如何获取用户浏览历史？**

```python
observation, _, _, _ = env.step(None)
for session in observation.sessions():
    print(f"用户浏览了物品 {session['v']}")
```

**Q: 如何判断用户会话结束？**

```python
done = False
while not done:
    observation, reward, done, info = env.step(action)
    if done:
        print("用户会话结束")
```

### 11.2 Agent开发问题

**Q: 如何处理冷启动问题？**

```python
def act(self, observation, reward, done):
    if np.sum(self.product_views) == 0:
        # 冷启动：随机推荐
        action = np.random.randint(self.config.num_products)
    else:
        # 正常推荐
        action = np.argmax(self.product_views)
    return {...}
```

**Q: 如何实现在线学习？**

```python
def train(self, observation, action, reward, done=False):
    # 每次交互都更新模型
    if action is not None:
        self.update_model(action['a'], reward)
```

**Q: 如何实现探索策略？**

```python
def act(self, observation, reward, done):
    if np.random.random() < self.epsilon:
        # 探索
        action = np.random.randint(self.config.num_products)
    else:
        # 利用
        action = np.argmax(self.Q)
    return {...}
```

### 11.3 性能优化问题

**Q: 如何加速训练？**

- 使用 `with_cache=True`缓存训练数据
- 减少离线用户数量
- 使用更简单的模型

**Q: 如何处理大规模物品？**

- 使用矩阵分解方法
- 使用近似最近邻搜索
- 采用负采样策略

## 十二、扩展资源

### 12.1 相关论文

1. **RecoGym论文**

   - Rohde et al., "RecoGym: A Reinforcement Learning Environment for the problem of Product Recommendation in Online Advertising" (RecSys 2018)
2. **IPS校正**

   - Ionides et al., "Importance Sampling" (2008)
   - Swaminathan & Joachims, "Counterfactual Risk Minimization" (WSDM 2015)
3. **强化学习推荐**

   - Chen et al., "Top-K Off-Policy Correction for a REINFORCE Recommender System" (WSDM 2019)
   - Zheng et al., "DRN: A Deep Reinforcement Learning Framework for News Recommendation" (WWW 2018)

### 12.2 相关项目

- [RecSim](https://github.com/google-research/recsim): Google的可配置推荐模拟器
- [Recommenders](https://github.com/microsoft/recommenders): Microsoft推荐系统最佳实践
- [TensorFlow Recommenders](https://www.tensorflow.org/recommenders): TensorFlow推荐系统库

### 12.3 社区资源

- [RecoGym GitHub](https://github.com/criteo-research/reco-gym)
- [OpenAI Gym文档](https://gym.openai.com/)
- [推荐系统学习资源](../README.md)

## 十三、引用

如果你在研究中使用了RecoGym，请引用：

```bibtex
@article{rohde2018recogym,
  title={RecoGym: A Reinforcement Learning Environment for the problem of Product Recommendation in Online Advertising},
  author={Rohde, David and Bonner, Stephen and Dunlop, Travis and Vasile, Flavian and Karatzoglou, Alexandros},
  journal={arXiv preprint arXiv:1808.00720},
  year={2018}
}
```

## 十四、许可证

Copyright CRITEO

Licensed under the Apache License, Version 2.0

---

**文档版本**：1.0
**更新时间**：2026年3月
**适用版本**：RecoGym 0.1.2.3
