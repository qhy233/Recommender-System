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
2. 阅读 [00_强化学习推荐系统论文综合总结](论文/00_强化学习推荐系统论文综合总结.md)
3. 逐篇研读 [8篇论文](论文/)，理解不同技术路线
4. 结合 [note.md](note.md) 中的推荐系统知识，理解RL与推荐系统的结合点

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

- 2026-03-1: 完成推荐系统基础知识笔记整理
- 2026-03-10: 完成8篇强化学习推荐系统论文研读
- 2026-03-12: 创建推荐系统工作原理概览文档
- 2026-03-12: 完成强化学习推荐系统论文综合总结
