# 05_ListRec_2019_IJCAI_SlateQ_Decomposition

**论文信息**
- 标题：Reinforcement Learning for Slate-based Recommender Systems: A Tractable Decomposition and Practical Methodology
- 作者：Eugene Ie, Vihan Jain, Jing Wang, Sanmit Narvekar, Ritesh Agarwal, Rui Wu, Heng-Tze Cheng, Morgane Lustman, Vince Gatto, Paul Covington, Jim McFadden, Tushar Chandra, Craig Boutilier
- 单位：Google Research, University of Texas at Austin, YouTube
- 发表：IJCAI 2019

---

## 一、核心痛点与动机 (The "Why")

### 1.1 现实推荐系统的Slate问题

> "Most practical recommender systems focus on estimating immediate user engagement without considering the long-term effects of recommendations on user behavior."

**翻译**：大多数实用推荐系统专注于估计即时用户参与度，而不考虑推荐对用户行为的长期影响。

**核心挑战**：
| 挑战 | 描述 | 影响 |
|------|------|------|
| Slate推荐 | 一次推荐多个物品 | 动作空间组合爆炸 |
| 物品交互效应 | 物品间可能存在相互影响 | 选择概率建模复杂 |
| 即时vs长期 | 传统方法只优化即时奖励 | 忽略长期用户价值 |

### 1.2 RL在推荐系统中的复杂性

> "However, the application of RL has largely been confined to restricted domains due to the complexities of putting such models into practice at scale."

**翻译**：然而，由于在实际规模部署这些模型的复杂性，RL的应用在很大程度上仅限于受限领域。

**传统方法的问题**：
- 关注即时响应（点击率、评分）
- 忽略推荐对用户未来行为的影响
- Slate推荐面临组合动作空间问题

### 1.3 现有方法的局限

> "Recent approaches to RL with such combinatorial actions make inroads into this problem, but are unable to scale to problems of the size encountered in large, real-world recommender systems, in part because of their generality."

**翻译**：最近处理这种组合动作的RL方法开始解决这个问题，但无法扩展到大型现实推荐系统中遇到的问题规模，部分原因是它们的通用性。

---

## 二、核心创新点

### 2.1 SLATE Q：可分解的Slate价值函数

> "We develop SLATE Q, a decomposition of value-based temporal-difference and Q-learning that renders RL tractable with slates. Under mild assumptions on user choice behavior, we show that the long-term value (LTV) of a slate can be decomposed into a tractable function of its component item-wise LTVs."

**翻译**：我们开发了SLATE Q，这是一种基于价值的时序差分和Q学习的分解方法，使得使用slate的RL变得可处理。在对用户选择行为的温和假设下，我们表明slate的长期价值(LTV)可以分解为其组成物品级LTV的易处理函数。

**核心思想**：
- Slate的LTV分解为物品级LTV的函数
- 物品级LTV可以独立学习
- 大大降低计算复杂度

### 2.2 Slate优化：多项式时间求解

> "Despite the combinatorial (and fractional) nature of the underlying optimization problem, we show that it can be solved in polynomial-time by a two-step reduction to a linear program (LP)."

**翻译**：尽管底层优化问题是组合的（和分数的），我们表明可以通过两步约简到线性规划(LP)在多项式时间内求解。

**求解方法**：
1. 分数混合整数规划 formulation
2. 重新 formulation 和松弛
3. 精确求解为线性规划

**启发式方法**：
- Top-k 策略
- Greedy 策略
- 虽然无理论保证，但实践效果好

### 2.3 实用方法论：复用现有基础设施

> "We outline a methodology that leverages existing myopic learning-based recommenders to quickly develop a recommender that handles LTV."

**翻译**：我们概述了一种利用现有短视基于学习推荐器的方法，以快速开发处理LTV的推荐器。

**核心优势**：
- 加速RL模型开发
- 复用现有训练基础设施
- 复用相同的服务基础设施进行LTV评分

---

## 三、问题建模

### 3.1 Slate推荐MDP

**MDP要素**：
| 要素 | 定义 | 推荐系统含义 |
|------|------|--------------|
| $S$ | 状态空间 | 用户状态 |
| $A$ | 动作空间 | Slate（物品集合） |
| $T$ | 转移函数 | 用户状态演化 |
| $R$ | 奖励函数 | 用户参与度 |
| $\gamma$ | 折扣因子 | 未来奖励折扣 |

### 3.2 用户选择模型

#### 3.2.1 条件Logit模型 (Conditional Logit)

> "The probability of the user selecting j from a slate A of items is $P(j|A) = \frac{e^{u(x_{ij})}}{\sum_{l \in A} e^{u(x_{il})}}$"

**翻译**：用户从物品集合A中选择物品j的概率为 $P(j|A) = \frac{e^{u(x_{ij})}}{\sum_{l \in A} e^{u(x_{il})}}$

**一般形式**：
$$P(j|A) = \frac{v(x_{ij})}{\sum_{l \in A} v(x_{il})}$$

其中 $v(\cdot)$ 是任意函数。

#### 3.2.2 级联模型 (Cascade Model)

> "The standard cascade model assumes that a user i has some affinity (e.g., perceived utility) $u_{ijk}$ for any item $j_k$; sequentially scans a list of items $A = (j_1, j_2, ..., j_K)$ in order; and will select an item with probability $\phi(u_{ijk})$ for some non-decreasing function $\phi$."

**翻译**：标准级联模型假设用户i对任何物品$j_k$有一定的亲和力（例如感知效用）$u_{ijk}$；按顺序浏览物品列表$A = (j_1, j_2, ..., j_K)$；将以某个非递减函数$\phi$的概率$\phi(u_{ijk})$选择物品。

**选择概率**：
$$P(j_k|A) = \prod_{l < k} (1 - \phi(u_{ij_l})) \cdot \phi(u_{ijk})$$

### 3.3 长期价值定义

**Q值定义**：
$$Q(s, a) = \mathbb{E}\left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \right]$$

**目标**：
$$\max_\pi \mathbb{E}_{s_0 \sim \rho_0, a_t \sim \pi(\cdot|s_t)} \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \right]$$

---

## 四、SLATE Q分解详解

### 4.1 分解原理

**核心洞察**：
在温和的用户选择行为假设下，Slate的长期价值可以分解为物品级长期价值的函数。

**分解公式**：
假设用户从Slate A中选择物品j的概率为 $P(j|A)$，则Slate的价值为：
$$Q(s, A) = \sum_{j \in A} P(j|A) \cdot Q_{item}(s, j)$$

其中 $Q_{item}(s, j)$ 是物品j在状态s下的长期价值。

### 4.2 TD学习中的分解

**SARSA和Q学习**：
- 物品级LTV可以在物品级别学习
- 尽管物品总是以Slate形式呈现给用户
- 这对泛化和探索效率至关重要

**优势**：
- 无需构建显式的Slate级Q函数
- 物品级探索和泛化
- 大大降低状态空间复杂度

### 4.3 物品级Q值学习

**TD更新**：
$$Q_{item}(s, j) \leftarrow Q_{item}(s, j) + \alpha \left[ r + \gamma \sum_{j' \in A'} P(j'|A') \cdot Q_{item}(s', j') - Q_{item}(s, j) \right]$$

---

## 五、Slate优化

### 5.1 优化问题定义

**目标**：给定物品级Q值，找到最大化LTV的Slate：
$$\max_A \sum_{j \in A} P(j|A) \cdot Q_{item}(s, j)$$

**约束**：
- $|A| = K$（Slate大小固定）
- $A \subseteq I$（物品来自候选集）

### 5.2 分数混合整数规划

**Formulation**：
$$\max \sum_{j \in I} x_j \cdot v_j$$
$$\text{s.t.} \sum_{j \in I} x_j = K$$
$$x_j \in \{0, 1\}$$

其中 $v_j = P(j|A) \cdot Q_{item}(s, j)$

### 5.3 线性规划求解

**两步约简**：
1. 分数混合整数规划 formulation
2. 重新 formulation 和松弛
3. 精确求解为线性规划

**复杂度**：
- 多项式时间可解
- 适用于大规模问题

### 5.4 启发式方法

#### 5.4.1 Top-k策略
- 选择Q值最高的k个物品
- 简单但忽略物品间交互

#### 5.4.2 Greedy策略
- 贪心选择每次增加价值最大的物品
- 考虑物品间部分交互

---

## 六、方法论：复用现有基础设施

### 6.1 动机

> "Despite the recent successes of RL afforded by deep Q-networks (DQNs), the deployment of RL in practical recommenders is hampered by the need to construct relevant state and action features for DQN models, and to train models that serve millions-to-billions of users."

**翻译**：尽管深度Q网络(DQN)带来了RL的最新成功，但RL在实用推荐器中的部署受到阻碍，因为需要为DQN模型构建相关的状态和动作特征，并训练为数十亿用户服务的模型。

### 6.2 方法论框架

**核心思想**：
- 基于现有的短视推荐管道构建TD学习
- 复用现有基础设施
- 逐步引入LTV优化

**实现步骤**：
1. **复用现有推荐器**：使用已有的myopic模型
2. **构建TD学习**：在现有管道上添加TD学习
3. **复用评分服务**：使用相同的服务基础设施进行LTV评分

### 6.3 在线实验

> "We demonstrate our methods in simulation, and validate the scalability of decomposed TD-learning using SLATE Q in live experiments on YouTube."

**翻译**：我们在模拟中展示了我们方法，并验证了SLATE Q在YouTube在线实验中分解的TD学习的可扩展性。

---

## 七、实验设置

### 7.1 仿真环境：RecSim

> "To evaluate these methods systematically, we introduce a recommender simulation environment, RecSim, that allows the straightforward configuration of an item collection (or vocabulary), a user (latent) state model and a user choice model."

**翻译**：为了系统评估这些方法，我们引入了一个推荐仿真环境RecSim，允许直接配置物品集合（或词汇表）、用户（潜在）状态模型和用户选择模型。

**RecSim特性**：
- 可配置的物品集合
- 用户状态模型
- 用户选择模型

### 7.2 在线实验

**平台**：YouTube首页视频推荐

**验证内容**：
- SLATE Q的可扩展性
- 用户参与度提升
- 与短视方法对比

---

## 八、与Survey论文的联系

### 8.1 作为Slate推荐的实践案例

本论文是Survey论文中提到的"Slate推荐"的具体实践案例，详细解决了组合动作空间的RL问题。

### 8.2 关键技术对应关系

| Survey概念 | 本论文实现 |
|-------------|--------------|
| Slate推荐 | SLATE Q分解 |
| 组合动作空间 | LP多项式时间求解 |
| 用户选择模型 | Conditional Logit, Cascade Model |
| 实用方法论 | 复用现有基础设施 |

### 8.3 与YouTube REINFORCE论文的对比

| 维度 | YouTube REINFORCE | SLATE Q |
|------|-------------------|---------|
| 算法 | Policy Gradient | Q-learning |
| 优化 | Top-K Off-Policy Correction | Slate分解 + LP |
| 选择模型 | 隐式学习 | 显式建模 |
| 可扩展性 | 极大规模 | 多项式时间 |

---

## 九、工程实践启示

### 9.1 选择模型的重要性

**关键洞察**：
- 显式建模用户选择行为
- 利用选择模型的结构进行分解
- 在温和假设下实现可处理性

### 9.2 渐进式部署策略

**步骤**：
1. 使用现有短视推荐器
2. 添加TD学习组件
3. 逐步引入LTV优化
4. 在线A/B测试验证

### 9.3 优化策略选择

| 策略 | 适用场景 | 复杂度 |
|------|----------|--------|
| LP求解 | 大规模精确求解 | 多项式 |
| Top-k | 快速原型 | O(n log n) |
| Greedy | 中等规模 | O(k·n) |

---

## 十、论文引用

```
Eugene Ie, Vihan Jain, Jing Wang, Sanmit Narvekar, Ritesh Agarwal, Rui Wu, Heng-Tze Cheng, 
Morgane Lustman, Vince Gatto, Paul Covington, Jim McFadden, Tushar Chandra, Craig Boutilier
Reinforcement Learning for Slate-based Recommender Systems: A Tractable Decomposition and Practical Methodology
IJCAI 2019
```
