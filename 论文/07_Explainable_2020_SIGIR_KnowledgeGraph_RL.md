# 07_Explainable_2020_SIGIR_KnowledgeGraph_RL

**论文信息**
- 标题：Interactive Recommender System via Knowledge Graph-enhanced Reinforcement Learning
- 作者：Sijin Zhou, Xinyi Dai, Haokun Chen, Weinan Zhang, Kan Ren, Ruiming Tang, Xiuqiang He, Yong Yu
- 单位：Shanghai Jiao Tong University, Huawei Noah's Ark Lab
- 发表：SIGIR 2020

---

## 一、核心痛点与动机 (The "Why")

### 1.1 交互式推荐系统的挑战

> "Interactive recommender system (IRS) has drawn huge attention because of its flexible recommendation strategy and the consideration of optimal long-term user experiences."

**翻译**：交互式推荐系统(IRS)因其灵活的推荐策略和对最优长期用户体验的考虑而受到广泛关注。

**核心挑战**：
| 挑战 | 描述 | 影响 |
|------|------|------|
| 样本效率低 | RL方法需要大量交互数据 | 在线探索损害用户体验 |
| 用户响应稀疏 | 大量候选物品，稀疏反馈 | 难以学习有效策略 |
| 动作空间大 | 大量候选物品 | 探索和泛化困难 |

### 1.2 DRL在IRS中的困境

> "DRL methods normally face sample efficiency problem, i.e., learning such a policy requires a huge amount of data through interacting with real users before achieving the best policy, which may degrade user experience and damage system profit."

**翻译**：DRL方法通常面临样本效率问题，即学习这样的策略需要通过与真实用户交互获得大量数据才能达到最佳策略，这可能会降低用户体验并损害系统利润。

**传统方法的局限**：
- **MAB方法**：假设用户偏好不变，无法建模动态偏好转移
- **DRL方法**：样本效率低，需要大量交互数据
- **离线训练**：存在估计偏差问题

### 1.3 知识图谱的潜力

> "Knowledge graph (KG), a well-known structured knowledge base, represents various relations as the attributes of items and links items if they have common attributes, which has shown great effectiveness for representing the correlation between items."

**翻译**：知识图谱(KG)是一种著名的结构化知识库，将各种关系表示为物品属性，并在物品具有共同属性时链接物品，已显示出表示物品相关性的巨大有效性。

**关键洞察**：
- 用户喜欢电影《盗梦空间》→ 可能喜欢导演诺兰的其他作品
- 一次用户-物品交互可以揭示用户对多个相关物品的偏好
- KG的语义空间有助于提取用户兴趣

---

## 二、核心创新点

### 2.1 KGQR：知识图谱增强的Q学习框架

> "We propose KGQR (Knowledge Graph enhanced Q-learning framework for interactive Recommendation), a novel architecture that extends DQN."

**翻译**：我们提出了KGQR（知识图谱增强的交互式推荐Q学习框架），一种扩展DQN的新颖架构。

**核心思想**：
- 将图学习和顺序决策整合为一体
- 利用KG中的知识促进IRS中的模式挖掘
- 提高样本效率

### 2.2 三大创新应用

#### 2.2.1 候选选择引导

> "Each step the candidate set for recommendation is dynamically generated from the local graph of KG, by considering the neighborhood of the items in user's high-scored interacted items."

**翻译**：每一步推荐候选集从KG的局部图动态生成，考虑用户高分交互物品的邻域。

**优势**：
- 避免枚举整个物品集
- 聚焦KG认为更有用的样本
- 更好利用有限学习样本

#### 2.2.2 物品和用户状态表示增强

> "By aggregating the semantic correlations among items in KG, the item embedding and the user's preference are effectively represented, which leads to more accurate Q-value approximation."

**翻译**：通过聚合KG中物品间的语义相关性，物品嵌入和用户偏好得到有效表示，从而实现更准确的Q值近似。

#### 2.2.3 用户偏好传播

> "The user feedback is modeled to propagate via structure information of KG, so that the user's preference can be transited among correlated items. In this way, one interactive record can affect multiple connected items, thus the sample efficiency is improved."

**翻译**：用户反馈被建模为通过KG的结构信息传播，使用户偏好可以在相关物品之间转移。这样，一条交互记录可以影响多个相关物品，从而提高样本效率。

---

## 三、问题建模

### 3.1 MDP定义

**目标**：学习推荐策略 $\pi: S \rightarrow I$，最大化累积效用：
$$\pi^* = \arg\max_{\pi \in \Pi} \mathbb{E}\left[ \sum_{t=0}^{T} r(s_t, i_t) \right]$$

### 3.2 Q值函数

**Q值定义**：
$$Q^\pi(s_t, i_t) = \mathbb{E}\left[ \sum_{j=0}^{T-t} \gamma^j r_{t+j} \right]$$

**最优Bellman方程**：
$$Q^*(s_t, i_t) = \mathbb{E}_{s_{t+1}}[r_t + \gamma \max_{i_{t+1}} Q^*(s_{t+1}, i_{t+1}) | s_t, i_t]$$

### 3.3 KG增强的Q函数

**关键创新**：
$$Q^*_{\theta_Q}(s_t, i_t; G) = Q^*_{\theta_Q}(s_t(G), i_t(G))$$

其中 $G$ 是知识图谱，提供环境和动作的先验知识。

---

## 四、KGQR框架详解

### 4.1 整体架构

**四大组件**：
1. **图卷积模块**：从KG学习实体嵌入
2. **状态表示模块**：建模用户偏好
3. **候选选择模块**：动态生成候选集
4. **Q学习网络模块**：学习推荐策略

### 4.2 KG增强的状态表示

#### 4.2.1 图卷积嵌入层

**输入**：用户点击物品 $o_t = \{i_1, i_2, ..., i_n\}$

**图卷积**：
$$e_h^{(l+1)} = \sigma\left( W^{(l)} \sum_{r \in R} \sum_{e_t \in N_h^r} \frac{1}{|N_h^r|} e_t^{(l)} + b^{(l)} \right)$$

其中：
- $e_h^{(l)}$：实体 $h$ 在第 $l$ 层的嵌入
- $N_h^r$：通过关系 $r$ 与实体 $h$ 相连的实体集合
- $W^{(l)}, b^{(l)}$：可学习参数

**优势**：
- 捕捉KG中的高阶语义关系
- 聚合邻域信息
- 学习物品间的相关性

#### 4.2.2 用户状态表示

**方法**：
- 使用RNN（如GRU）建模用户顺序偏好
- 输入：点击物品的嵌入序列
- 输出：用户状态 $s_t$

**状态更新**：
$$s_t = \text{GRU}(s_{t-1}, e_{i_{t-1}})$$

### 4.3 候选选择模块

**核心思想**：
- 从KG局部图动态生成候选集
- 考虑用户高分交互物品的邻域

**算法流程**：
1. 识别用户高分交互物品
2. 在KG中找到这些物品的邻域
3. 构建候选物品集 $I_t(G)$

**优势**：
- 避免枚举整个物品集
- 聚焦相关物品
- 提高计算效率

### 4.4 Q学习网络

**网络结构**：
- 输入：状态 $s_t$ 和候选物品嵌入
- 输出：每个候选物品的Q值

**损失函数**：
$$L(\theta_Q) = \mathbb{E}\left[ \left( r_t + \gamma \max_{i_{t+1}} Q_{\theta'_Q}(s_{t+1}, i_{t+1}) - Q_{\theta_Q}(s_t, i_t) \right)^2 \right]$$

**训练策略**：
- 经验回放
- 目标网络
- $\epsilon$-greedy探索

---

## 五、关键技术细节

### 5.1 知识图谱构建

**三元组形式**：(head, relation, tail)

**示例**：
- (Nolan, DirectorOf, Inception)
- (Inception, HasGenre, SciFi)

**数据来源**：
- DBpedia
- NELL
- Microsoft Satori

### 5.2 图神经网络选择

**GCN优势**：
- 捕捉高阶关系
- 端到端学习
- 可扩展性好

**其他选择**：
- GAT（图注意力网络）
- R-GCN（关系GCN）

### 5.3 用户偏好传播

**核心机制**：
- 用户对物品 $i$ 的偏好传播到KG中与 $i$ 相连的物品
- 一次交互影响多个相关物品
- 缓解数据稀疏问题

**数学表达**：
$$\text{Preference}(i) \rightarrow \text{Preference}(N_i)$$

其中 $N_i$ 是物品 $i$ 在KG中的邻域。

---

## 六、实验结果

### 6.1 数据集

- 两个真实世界数据集
- 包含用户交互历史和知识图谱

### 6.2 评估指标

| 指标 | 说明 |
|------|------|
| 点击率 (CTR) | 用户点击推荐物品的比例 |
| 累积奖励 | 整个交互过程的累积效用 |
| 样本效率 | 达到目标性能所需交互次数 |

### 6.3 性能对比

**对比方法**：
- 传统DQN
- DDPG
- MAB方法

**关键发现**：
> "KGQR is able to achieve better performance than state-of-the-arts with much fewer user-item interactions, which indicates high sample efficiency."

**翻译**：KGQR能够以更少的用户-物品交互实现比最先进方法更好的性能，这表明了高样本效率。

---

## 七、与Survey论文的联系

### 7.1 作为知识图谱增强推荐的实践案例

本论文是Survey论文中提到的"知识图谱增强推荐"的具体实践案例，展示了KG如何提升RL推荐系统的性能。

### 7.2 关键技术对应关系

| Survey概念 | 本论文实现 |
|-------------|--------------|
| 状态表示增强 | KG增强的实体嵌入 |
| 样本效率 | 用户偏好传播机制 |
| 候选生成 | KG局部图候选选择 |
| 先验知识 | KG结构化知识 |

### 7.3 与其他论文的对比

| 维度 | DRN | YouTube REINFORCE | KGQR |
|------|-----|-------------------|------|
| 算法 | DQN | REINFORCE | DQN + KG |
| 样本效率 | 低 | 中 | 高 |
| 先验知识 | 无 | 无 | KG |
| 可解释性 | 低 | 低 | 高 |

---

## 八、工程实践启示

### 8.1 KG构建与维护

**关键步骤**：
1. 选择合适的知识图谱源
2. 对齐物品与KG实体
3. 定期更新KG

### 8.2 候选选择策略

**动态候选集**：
- 基于用户历史交互
- 考虑KG邻域
- 平衡相关性和多样性

### 8.3 样本效率优化

**用户偏好传播**：
- 一次交互影响多个物品
- 缓解冷启动问题
- 加速策略学习

---

## 九、论文引用

```
Sijin Zhou, Xinyi Dai, Haokun Chen, Weinan Zhang, Kan Ren, Ruiming Tang, Xiuqiang He, Yong Yu
Interactive Recommender System via Knowledge Graph-enhanced Reinforcement Learning
SIGIR '20, July 25–30, 2020, Virtual Event, China
```
