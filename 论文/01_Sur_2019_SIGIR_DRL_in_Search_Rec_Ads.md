# 01_Survey_2019_SIGIR_DRL_in_Search_Rec_Ads

**论文信息**
- 标题：Deep Reinforcement Learning for Search, Recommendation, and Online Advertising: A Survey
- 作者：Xiangyu Zhao (MSU), Long Xia (JD.com), Jiliang Tang (MSU), Dawei Yin (JD.com)
- 发表：SIGWEB Newsletter Spring 2019

---

## 一、核心痛点与动机 (The "Why")

### 1.1 传统方法的局限性

> "The majority of existing methods consider information seeking as a static task and generate objects following a fixed greedy strategy. This may fail to capture the dynamic nature of users' preferences (or environment)."
> 
> **翻译**：大多数现有方法将信息检索视为静态任务，采用固定的贪心策略生成对象。这可能无法捕捉用户偏好（或环境）的动态变化本质。

> "Most traditional methods are developed to maximize the short-term reward, while completely neglecting whether the suggested objects will contribute more in long-term reward."
> 
> **翻译**：大多数传统方法致力于最大化短期奖励，却完全忽略了推荐对象是否能为长期奖励做出更大贡献。

**传统方法的核心问题**：
| 问题 | 描述 | 后果 |
|------|------|------|
| 静态贪心策略 | 将任务视为静态过程，采用固定策略 | 无法捕捉用户偏好的动态变化 |
| 目光短浅 | 仅优化短期奖励（如即时点击率） | 忽略用户生命周期价值等长期收益 |
| 缺乏交互性 | 无法根据实时反馈调整策略 | 难以适应用户兴趣漂移 |

### 1.2 强化学习的破局优势

> "Employing RL for information seeking can naturally resolve the aforementioned challenges. First, considering the information seeking tasks as sequential interactions between an RL agent (system) and users (environment), the agent can continuously update its strategies according to users' real-time feedback during the interactions. Second, the RL frameworks are designed to maximize the long-term cumulative reward from users."

**RL 的两大核心优势**：
1. **持续更新策略**：将信息检索视为智能体与用户间的顺序交互过程，根据实时反馈持续更新策略
2. **最大化长期累积奖励**：识别那些即时奖励小但长期贡献大的物品

---

## 二、技术基础 (Technical Foundations)

### 2.1 问题建模 (Problem Formulation)

#### 2.1.1 多臂老虎机 (Multi-Armed Bandits, MAB)

> "The Multi-Armed Bandits (MABs) problem is a simple model for the exploration/exploitation trade-off."

**定义**：一个 K-MAB 是一个三元组 $\langle A, R, \pi \rangle$
- $A$：动作（臂）集合，$|A| = K$
- $r = R(a)$：执行动作 $a$ 时的奖励分布
- $\pi$：描述可能动作的概率分布策略

**上下文老虎机 (Contextual Bandit)**：
- MAB 的扩展，考虑额外的上下文信息
- 适用于推荐系统中利用用户/物品特征进行个性化决策

#### 2.1.2 马尔可夫决策过程 (Markov Decision Process, MDP)

> "A Markov decision process (MDP) is a classical formalization of sequential decision making, which is a mathematically idealized form of reinforcement learning problem."

**定义**：MDP 是一个五元组 $\langle S, A, T, R, \pi \rangle$
- $S$：状态集合
- $A$：离散动作集合
- $T$：状态转移函数 $s_{t+1} = T(s_t, a_t)$
- $r = R(s, a)$：在状态 $s$ 执行动作 $a$ 的奖励分布
- $\pi(a|s)$：策略，描述智能体行为的概率分布

**目标**：最大化期望折扣回报
$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$
其中 $\gamma \in [0, 1]$ 为折扣因子。

**部分可观测马尔可夫决策过程 (POMDP)**：
- MDP 的扩展，适用于系统状态不一定可观测的情况
- 在会话搜索、对话推荐等场景中广泛应用

#### 2.1.3 多智能体设置 (Multi-Agent Setting)

**定义**：多智能体博弈是一个元组 $\langle S, A^1, ..., A^n, T, R^1, ..., R^n, \pi^1, ..., \pi^n \rangle$

**类型**：
- **完全合作**：$\pi^1 = ... = \pi^n$，所有智能体采用相同策略最大化相同期望回报
- **完全竞争**：$n=2$ 且 $\pi^1 = -\pi^2$，两个智能体策略相反
- **混合博弈**：既非完全合作也非完全竞争

### 2.2 策略学习 (Policy Learning)

#### 2.2.1 基于模型 vs 无模型 (Model-based vs Model-free)

| 类型 | 特点 | 代表算法 |
|------|------|----------|
| **Model-based** | 显式学习系统模型，利用模型求解 MDP | Dyna, Prioritized Sweeping, Q-iteration, Policy Gradient |
| **Model-free** | 忽略模型，直接从环境交互中学习值函数 | Q-learning, SARSA, LSPI, Actor-Critic |

#### 2.2.2 值函数 vs 策略搜索 (Value function vs Policy search)

| 类型 | 思路 | 代表算法 |
|------|------|----------|
| **值函数方法** | 先找到最优值函数，再提取最优策略 | Dyna, Q-learning, SARSA, DQN |
| **策略搜索方法** | 直接在策略空间中搜索最优策略 | Policy Gradient (PG), REINFORCE |
| **Actor-Critic** | 结合两者：Critic 估计值函数，Actor 更新策略 | A2C, A3C, DDPG |

---

## 三、强化学习在搜索中的应用 (RL for Search)

### 3.1 查询理解 (Query Understanding)

> "Query understanding is the primary task for the search engine to understand users' information needs."

**核心痛点**：用户原始查询往往词不达意，需要查询重写 (Query Reformulation)

**RL 方案**：
- 将搜索引擎视为黑盒，智能体学习重写查询以最大化返回的相关文档数量
- **多智能体架构**：子智能体处理不同查询子集，元智能体 (Meta-agent) 聚合答案
- 优势：每个子智能体只需学习针对子集的策略，训练更高效；支持并行学习

### 3.2 排序 (Ranking)

> "Relevance Ranking is the core problem of information retrieval and learning to rank (LTR) is the key technology in relevance ranking."

**核心痛点**：传统 LTR 仅优化预定义的前 $K$ 个位置指标，丢弃了其余信息

**MDPRank 方案**：
- 将排序构建为马尔可夫决策过程 (MDP)
- 奖励函数基于 IR 评估指标定义
- 通过最大化所有决策的累积奖励，利用全量排名位置信息

**多样性排序 (MDP-DIV)**：
> "To explicitly model the utility perceived by the users, the construction of a diverse ranking is formalized as a process of sequential decision making."

- 将多样化排序构建为连续状态 MDP
- 捕捉用户从前面文档中感知到的实用性
- 使用 REINFORCE 策略梯度算法训练

### 3.3 整页优化 (Whole-Page Optimization)

> "Page presentation is broadly defined as the strategy to present a set of items on search result page (SERP), which is much more expressive than a ranked list."

**核心痛点**：异构结果（网页、新闻、图像、视频、购物）在搜索结果页上的排版面临组合爆炸问题

**RL 方案**：
- **Bandit + 爬山算法**：仅考虑组件间的成对交互，实现实时高效探索
- **Policy-based 方法**：从高维排版空间中快速选择动作，解决效率问题

### 3.4 会话搜索 (Session Search)

> "Markov chain in session search is observed: users' judgment of search results in the prior iteration will influence users' behaviors in the next search iteration."

**核心洞察**：用户会话内的多次查询重写具有马尔可夫链特性

**POMDP 建模**：
- 将会话搜索建模为用户与搜索引擎之间的合作博弈
- 用户和搜索引擎共同工作以最大化长期累积奖励

**多场景协同 (MA-RDPG)**：
- 在电商搜索中引入 Multi-Agent Actor-Critic 思想
- **私有智能体 (Actor)**：执行各场景排序
- **中心化评论家 (Centralized Critic)**：评估整体表现
- 通过共享动作价值函数和传递历史信息消息实现跨场景协作

---

## 四、强化学习在推荐中的应用 (RL for Recommendation)

> "Recommender systems target to capture users' preferences according to their feedback (or behaviors, e.g. rating and review) and suggest items that match their preferences."

### 4.1 探索与利用困境 (Exploitation/Exploration Dilemma)

> "Traditional recommender systems suffer from the exploitation-exploration dilemma, where exploitation is to recommend items that are predicted to best match users' preferences, while exploration is to recommend items randomly to collect more users' feedback."

**核心问题**：
- **利用**：推荐预测最匹配用户偏好的物品
- **探索**：随机推荐物品以收集更多用户反馈
- 一直利用 $\Rightarrow$ 信息茧房；随机探索 $\Rightarrow$ 伤害用户体验

**LinUCB 方案**：
- 将个性化新闻推荐建模为上下文老虎机 (Contextual Bandit)
- 根据用户和文章的上下文信息顺序选择文章
- 在个性化利用和发掘新物品之间取得平衡，最大化总点击量

**传统策略**：$\epsilon$-greedy, EXP3, UCB1

### 4.2 时序动态性 (Temporal Dynamics)

> "Most existing recommender systems such as collaborative filtering, content-based and learning-to-rank have been extensively studied with the stationary environment (reward) assumption, where user's preference is assumed to be static. However, this assumption is usually not true in reality since users' preferences are dynamic."

**核心痛点**：传统模型假设用户偏好是静态的，但现实中用户偏好是动态变化的

**MDP 建模方案**：
- **状态**：表征用户偏好
- **状态转移**：捕捉偏好随时间的演化
- 用户对推荐物品的每次反馈（跳过、点击、购买）都会触发状态的实时更新

**Bandit 扩展**：
- 引入可变奖励函数来描述环境的动态特性
- 粒子学习的动态上下文漂移模型
- 基于变化检测的框架，主动检测变化点并重启 UCB 索引

### 4.3 长期用户参与度 (Long Term User Engagement)

> "User engagement in recommendation is the assessment of user's desirable (even essential) responses to the items suggested by the recommender systems. User engagement can be measured not only in terms of immediate response, but more importantly in terms of long-term response."

**核心目标**：评估推荐行为对用户留存的长期影响

**关键指标**：
- 即时响应：点击、评分
- 长期响应：用户重复购买、回访频率

**DeepChain 方案**：
- 多智能体 RL 捕捉用户在入口页、详情页等多场景间的序列相关性
- 联合优化多个推荐策略
- 使用 DQN 框架将长期响应作为补充目标优化
- 引入基于模型的 RL 技术减少训练样本需求

### 4.4 页面级推荐 (Page-Wise Recommendation)

**核心痛点**：同时推荐一整页多样的物品并安排 2D 布局，导致动作空间极大（Action Space 爆炸）

**DDPG 方案**：
- 引入深度确定性策略梯度
- **Actor**：直接生成确定性的最优动作
- **Critic**：输出该状态-动作对的 Q 值
- 极大降低传统 Value-based 方法的计算成本
- 结合 CNN 捕捉页面展示模式和用户反馈

---

## 五、强化学习在在线广告中的应用 (RL for Online Advertising)

### 5.1 担保交付 (Guaranteed Delivery, GD)

**RL 方案**：
- 在不稳定的环境中，利用多智能体强化学习 (MARL) 为发布商推导合作策略
- 将曝光分配制定为拍卖问题
- 通过求解最优出价函数来分配

### 5.2 实时竞价 (Real-Time Bidding, RTB)

**RL 方案**：
- 将广告选择建模为具有预算约束和可变成本的 MAB 问题
- 由于广告活动生命周期的多次竞价特性，将其转化为 MDP
- **基于模型的 RL**：逼近状态价值以应对大规模拍卖和预算限制
- **RewardNet**：解决直接使用即时奖励带来的奖励设计陷阱
- **聚类方法**：解决海量广告主竞争的问题

---

## 六、核心挑战与未来方向 (Future Directions)

### 6.1 跨场景协同

> 打破单场景限制，建立同时考虑搜索、推荐和广告场景的协作 RL 框架

**关键问题**：
- 如何在不同场景间共享知识
- 如何处理场景间的冲突与协同

### 6.2 奖励与状态丰富化

**奖励函数设计**：
- 设计更复杂的奖励函数
- 将加入购物车、停留时间、AI 对话交互等纳入 RL 框架

**状态表示增强**：
- 更丰富的用户状态表示
- 融合多模态信息（文本、图像、行为序列）

### 6.3 最核心工程痛点：仿真环境

> "Due to the high cost and risk of directly testing new algorithms online, developing online environment simulators or history log based offline evaluation methods is a necessary prerequisite for pre-training and evaluating algorithms before they go online."

**核心挑战**：
- 直接在线测试新算法成本高昂且风险极大
- 需要开发在线环境仿真器 或基于历史日志的离线评估方法

**解决方案方向**：
- **Virtual-Taobao**：虚拟化真实在线零售环境
- **RecSim**：可配置的推荐系统仿真平台
- **RecoGym/PyRecGym**：推荐系统 RL 环境

---

## 七、关键技术总结

### 7.1 RL 方法分类

| 方法类型 | 适用场景 | 优势 | 代表工作 |
|----------|----------|------|----------|
| **Bandit** | 探索-利用平衡、无状态转移 | 简单高效、理论保证 | LinUCB, Thompson Sampling |
| **DQN** | 离散动作空间、大规模状态 | 样本效率高、稳定 | DRN, DEERS |
| **Policy Gradient** | 连续/高维动作空间 | 直接优化策略 | REINFORCE, Top-K Off-Policy |
| **Actor-Critic** | 复杂动作空间、需要稳定性 | 结合两者优势 | DDPG, MA-RDPG |
| **Multi-Agent** | 多场景协同 | 场景间协作 | DeepChain, MA-RDPG |

### 7.2 MDP 建模要素

| 要素 | 推荐系统中的含义 |
|------|------------------|
| **State $S$** | 用户偏好、历史行为、上下文 |
| **Action $A$** | 推荐物品、排序策略、页面布局 |
| **Reward $R$** | 点击率、转化率、用户参与度、长期价值 |
| **Transition $T$** | 用户偏好演化、行为模式变化 |
| **Policy $\pi$** | 推荐策略、排序策略 |

### 7.3 核心创新点总结

1. **从静态到动态**：将推荐视为顺序决策过程，捕捉用户偏好的动态演化
2. **从短期到长期**：最大化长期累积奖励而非即时点击
3. **从单点到系统**：多智能体协同优化整个推荐链路
4. **从规则到学习**：端到端学习最优策略，减少人工设计

---

## 八、论文引用

```
Xiangyu Zhao, Long Xia, Jiliang Tang, Dawei Yin. 
Deep Reinforcement Learning for Search, Recommendation, and Online Advertising: A Survey. 
SIGWEB Newsletter, Spring 2019.
```
