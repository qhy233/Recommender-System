# 推荐系统学习笔记

本仓库包含推荐系统的系统化学习资料，涵盖从基础理论到前沿研究的完整知识体系，特别聚焦于强化学习在推荐系统中的应用。

## 仓库结构

### 核心学习资料

#### 1. 推荐系统基础知识笔记 ([note.md](note.md))

- **内容概述**：系统化的推荐系统基础知识笔记，涵盖召回、排序、重排三大核心模块
- **主要章节**：
  - 推荐系统概述（微观、工业、宏观三个视角）
  - 召回模型（协同过滤、向量召回、序列召回）
  - 排序模型（Wide & Deep、特征交叉、序列建模、多目标优化、多场景建模）
  - 重排模型（MMR、DPP、PRM、PRS）
  - 多场景建模（多塔结构、动态权重建模）
- **特点**：详细的数学公式、算法原理、技术细节，适合深入学习和参考

#### 2. 推荐系统工作原理 ([论文/推荐系统工作原理.md](论文/推荐系统工作原理.md))

- **内容概述**：推荐系统的全局性概览，帮助建立整体认知框架
- **主要章节**：
  - 为什么需要推荐系统
  - 推荐系统的整体架构（召回-排序-重排）
  - 各阶段的核心任务与技术演进
  - 推荐系统的技术演进脉络
  - 面临的挑战与未来方向
  - 如何构建一个推荐系统
- **特点**：系统性、逻辑连贯，适合快速建立对推荐系统的整体理解

#### 3. 强化学习推荐系统论文研读 ([论文/](论文/))

##### 3.1 综合总结 ([论文/00_强化学习推荐系统论文综合总结.md](论文/00_强化学习推荐系统论文综合总结.md))

- **内容概述**：8篇强化学习推荐系统论文的综合分析与总结
- **主要章节**：
  - 论文概览与技术分类
  - 核心问题与动机
  - 技术框架深度解析（Value-based、Policy-based、Slate推荐、增强技术）
  - 关键技术主题（探索-利用、Off-policy学习、状态表示、奖励设计）
  - 工程实践挑战
  - 论文间技术联系
  - 未来研究方向
  - 个人见解与实践建议
- **特点**：深度技术分析、论文间联系、实践指导

##### 3.2 单篇论文研读

| 论文                                                       | 标题                                                                                         | 会议   | 年份 | 核心贡献                                 |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ------ | ---- | ---------------------------------------- |
| [01](论文/01_Sur_2019_SIGIR_DRL_in_Search_Rec_Ads.md)         | Deep Reinforcement Learning in Search, Recommendation, and Ads                               | SIGIR  | 2019 | 强化学习在推荐系统中的综述，建立理论基础 |
| [02](论文/02_Val_2018_WWW_DRN_News_Recommendation.md)         | DRN: A Deep Reinforcement Learning Framework for News Recommendation                         | WWW    | 2018 | Value-based方法，用户活跃度建模          |
| [03](论文/03_PolicyBased_2019_WSDM_TopK_OffPolicy_YouTube.md) | Top-K Off-Policy Correction for a REINFORCE Recommender System                               | WSDM   | 2019 | Policy-based方法，Top-K Off-policy修正   |
| [04](论文/04_Simulator_2019_AAAI_Virtual_Taobao.md)           | Virtual-Taobao: Virtualizing Real-world Online Retail Environment for Reinforcement Learning | AAAI   | 2019 | 用户行为模拟器，支持离线RL训练           |
| [05](论文/05_ListRec_2019_IJCAI_SlateQ_Decomposition.md)      | SlateQ: A Tractable Decomposition for Reinforcement Learning with Recommender Systems        | IJCAI  | 2019 | Slate推荐，Q值分解方法                   |
| [06](论文/06_PageRec_2018_RecSys_Page_wise_DDPG.md)           | Page-wise Recommender Systems for E-commerce                                                 | RecSys | 2018 | 页面级推荐，Actor-Critic + CNN           |
| [07](论文/07_Explainable_2020_SIGIR_KnowledgeGraph_RL.md)     | Knowledge Graph Enhanced Interactive Explainable Recommendation                              | SIGIR  | 2020 | 知识图谱增强，可解释性推荐               |
| [08](论文/08_NegFeedback_2018_KDD_Pairwise_DRL.md)            | Deep Reinforcement Learning for List-wise Recommendation                                     | KDD    | 2018 | 负反馈建模，Pairwise正则化               |

**每篇论文研读包含**：

- 核心痛点与创新点
- 技术细节与算法原理
- 实验结果与分析
- 与其他论文的联系
- 核心段落翻译

#### 4. 强化学习基础与实现

##### 4.1 强化学习基本概念 ([论文/强化学习基本概念.md](论文/强化学习基本概念.md))

- **内容概述**：强化学习核心概念的系统梳理
- **主要章节**：
  - 强化学习概述与核心要素
  - 马尔可夫决策过程（MDP）
  - 策略与价值函数
  - 探索与利用
  - 算法分类（Value-based、Policy-based、Actor-Critic）
  - 关键技术（经验回放、目标网络）
  - 应用场景与挑战
- **特点**：理论与实践结合，为理解RL推荐系统奠定基础

##### 4.2 强化学习在推荐系统中的实现方式 ([论文/强化学习在推荐系统中的实现方式.md](论文/强化学习在推荐系统中的实现方式.md))

- **内容概述**：强化学习算法在推荐系统中的具体实现方法
- **主要章节**：
  - 推荐系统的MDP建模（状态、动作、奖励设计）
  - 算法实现（DQN、REINFORCE、Actor-Critic、SlateQ）
  - 工程实践挑战（大规模动作空间、样本效率、探索-利用平衡）
  - 实现案例（新闻推荐、视频推荐、电商推荐）
  - 技术选型指南
  - 实现流程详解
- **特点**：理论到实践的桥梁，涵盖工程落地的关键考量

#### 5. 用户行为数据处理 ([论文/推荐系统用户行为数据的获取与处理.md](论文/推荐系统用户行为数据的获取与处理.md))

- **内容概述**：推荐系统用户行为数据的完整处理流程
- **主要章节**：
  - 用户行为数据概述与分类
  - 数据采集（埋点技术、采集架构）
  - 数据处理流程（清洗、转换、集成）
  - 特征工程（用户特征、物品特征、交叉特征）
  - 正负样本构建
  - 数据质量保障
  - 实时数据处理
  - 强化学习场景的特殊要求
- **特点**：覆盖数据处理的完整链路，结合工业实践

#### 6. 调研报告 ([调研报告.md](调研报告.md))

- **内容概述**：强化学习在推荐系统中应用的系统性调研报告
- **主要章节**：
  - 引言：推荐系统的挑战与强化学习范式的价值
  - 学习过程与成果：理论基础→知识体系→论文研读→技术实现→数据处理
  - 核心发现与见解：范式价值、技术演进、工程考量、未来方向
  - 总结与展望
  - 附录：学习产出清单
- **特点**：连贯叙述，融入个人见解，适合作为项目交付文档

### 实操项目

#### RecoGym：强化学习推荐系统实验环境 ([reco-gym/](reco-gym/))

- **项目来源**：RecSys 2018 REVEAL workshop论文配套代码
- **论文**：RecoGym: A Reinforcement Learning Environment for the problem of Product Recommendation in Online Advertising
- **内容概述**：OpenAI Gym风格的推荐系统强化学习环境

##### 核心功能

| 功能 | 说明 |
|------|------|
| 用户行为模拟 | 模拟真实电商用户浏览和点击行为 |
| 离线训练 | 使用历史数据预训练Agent |
| 在线学习 | 支持实时交互和增量学习 |
| 多种基线Agent | 内置10+种经典推荐算法 |
| 自定义Agent | 支持开发自己的推荐算法 |

##### 内置Agent列表

| Agent | 类型 | 说明 |
|-------|------|------|
| RandomAgent | 基线 | 随机推荐 |
| OrganicCount | 统计 | 基于有机浏览统计 |
| BanditCount | 统计 | 基于Bandit反馈统计 |
| EpsilonGreedy | 探索 | ε-greedy探索策略 |
| BanditMFSquare | 矩阵分解 | Bandit数据矩阵分解 |
| LogregPolyAgent | 机器学习 | 多项式逻辑回归 |
| NnIpsAgent | 深度学习 | 神经网络+IPS校正 |

##### 自定义实验 ([reco-gym/my_entries/](reco-gym/my_entries/))

实现了四种经典推荐算法：

| 算法 | 理论基础 | 学习目标 |
|------|----------|----------|
| Popularity Agent | 协同过滤、群体智慧 | 理解基于统计的推荐方法 |
| ε-greedy Agent | 探索-利用困境 | 掌握探索策略的设计 |
| Policy Gradient Agent | REINFORCE算法 | 理解策略梯度方法 |
| DQN Agent | Q-learning | 掌握价值函数方法 |

##### 实验结果

| Agent | CTR | 说明 |
|-------|-----|------|
| ε-greedy(ε=0.3) | 0.0156 | 适度探索效果最好 |
| Popularity | 0.0148 | 简单方法表现优异 |
| Policy Gradient | 0.0094 | 需要更多数据 |
| DQN | 0.0089 | 需要更多数据 |

##### 学习资源

- [中文详细说明文档](reco-gym/README_CN.md)：完整的项目介绍、使用指南、API文档
- [实验说明文档](reco-gym/my_entries/实验说明.md)：实验流程、代码解析、结果分析
- [Getting Started.ipynb](reco-gym/Getting%20Started.ipynb)：入门教程
- [Compare Agents.ipynb](reco-gym/Compare%20Agents.ipynb)：Agent性能对比
- [Bandit Feedback系列教程](reco-gym/)：深入理解Bandit反馈

##### 特点

- 理论与实践结合，适合算法验证和实验
- 支持离线预训练和在线学习两种模式
- 提供完整的评估工具和可视化
- 无需深度学习框架即可运行基础实验

### 系统化教程

#### Chapter 0: 推荐系统概述 ([chapter_0_introduction/](chapter_0_introduction/))

- 推荐系统是什么
- 技术地图

#### Chapter 1: 召回模型 ([chapter_1_retrieval/](chapter_1_retrieval/))

- 协同过滤（ItemCF、UserCF、Swing）
- 向量召回（i2i、u2i）
- 序列召回（MIND、SDM、SASRec、HSTU、TIGER）

#### Chapter 2: 排序模型 ([chapter_2_ranking/](chapter_2_ranking/))

- Wide & Deep
- 特征交叉（FM、DeepFM、xDeepFM、AutoInt）
- 序列建模（DIN、DIEN、DSIN）
- 多目标优化（Shared-Bottom、MMoE、PLE）
- 多场景建模（多塔结构、动态权重）

#### Chapter 3: 重排模型 ([chapter_3_rerank/](chapter_3_rerank/))

- 贪心方法（MMR）
- 个性化重排（PRM、PRS）

#### Chapter 4: 难点及热点研究 ([chapter_4_trends/](chapter_4_trends/))

- 偏差消除
- 冷启动
- 生成式推荐

#### Chapter 5: 项目实践 ([chapter_5_projects/](chapter_5_projects/))

- 需求理解与数据分析
- 基线建立
- 召回优化
- 特征工程
- 排序模型训练

#### Chapter 6: 面试经验 ([chapter_6_interview/](chapter_6_interview/))

- 机器学习基础
- 推荐系统核心
- 前沿趋势
- 产品思维

### 辅助资源

#### 图片资源 ([img/](img/))

- 各类模型架构图
- 算法流程图
- 技术对比图

#### 参考资料 ([chapter_references/](chapter_references/))

- 参考文献
- 相关论文链接

## 学习路径建议

### 初学者路径

1. 阅读 [推荐系统工作原理](论文/推荐系统工作原理.md)，建立整体认知
2. 学习 [Chapter 0-3](chapter_0_introduction/) 的基础教程
3. 参考详细笔记 [note.md](note.md) 深入理解技术细节

### 进阶路径

1. 系统学习 [Chapter 0-6](chapter_0_introduction/) 的完整教程
2. 深入研读 [note.md](note.md) 的数学推导和算法实现
3. 学习 [Chapter 5](chapter_5_projects/) 的项目实践

### 强化学习方向路径

1. 先掌握推荐系统基础知识（初学者路径）
2. 学习 [强化学习基本概念](论文/强化学习基本概念.md)，建立RL理论基础
3. 阅读 [00_强化学习推荐系统论文综合总结](论文/00_强化学习推荐系统论文综合总结.md)
4. 逐篇研读 [8篇论文](论文/)，理解不同技术路线
5. 学习 [强化学习在推荐系统中的实现方式](论文/强化学习在推荐系统中的实现方式.md)
6. 学习 [用户行为数据处理](论文/推荐系统用户行为数据的获取与处理.md)
7. 在 [RecoGym](reco-gym/) 环境中进行实操练习
8. 阅读 [调研报告](调研报告.md) 总结学习成果

## 核心技术体系

### 推荐系统三阶段架构

```
召回 → 排序 → 重排
  │       │       │
  │       │       └── 多样性优化、业务规则
  │       │
  │       └── 特征交叉、序列建模、多目标、多场景
  │
  └── 协同过滤、向量召回、序列召回
```

### 强化学习推荐系统技术分类

- **Value-based方法**：DQN、DRN、DEERS
- **Policy-based方法**：REINFORCE、Top-K Off-policy
- **Slate推荐**：SlateQ、Q值分解
- **增强技术**：模拟器、知识图谱、负反馈建模

## 贡献者

本仓库为个人学习笔记整理，欢迎交流讨论。

## 许可证

本仓库内容仅供学习交流使用。

## 更新日志

- 2026-03-01: 完成推荐系统基础知识笔记整理
- 2026-03-10: 完成8篇强化学习推荐系统论文研读
- 2026-03-12: 创建推荐系统工作原理概览文档
- 2026-03-12: 完成强化学习推荐系统论文综合总结
- 2026-03-13: 完成强化学习基本概念与实现方式文档
- 2026-03-13: 完成用户行为数据处理文档
- 2026-03-14: 完成调研报告
- 2026-03-14: 添加RecoGym实操项目
