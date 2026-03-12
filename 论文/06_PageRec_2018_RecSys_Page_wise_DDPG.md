# 06_PageRec_2018_RecSys_Page_wise_DDPG

**论文信息**
- 标题：Deep Reinforcement Learning for Page-wise Recommendations
- 作者：Xiangyu Zhao, Long Xia, Liang Zhang, Zhuoye Ding, Dawei Yin, Jiliang Tang
- 单位：Michigan State University, JD.com
- 发表：RecSys 2018

---

## 一、核心痛点与动机 (The "Why")

### 1.1 现实推荐系统的交互模式

> "In real-world recommendations such as e-commerce, a typical interaction between the system and its users is – users are recommended a page of items and provide feedback; and then the system recommends a new page of items."

**翻译**：在现实推荐系统（如电商）中，系统与用户之间的典型交互是——用户被推荐一页物品并提供反馈；然后系统推荐新的一页物品。

**核心挑战**：
| 挑战 | 描述 | 影响 |
|------|------|------|
| 实时反馈 | 如何根据用户实时反馈更新推荐策略 | 传统方法无法捕捉动态偏好 |
| 页面展示 | 如何生成一页物品并正确展示 | 需要同时优化物品选择和展示策略 |
| 2D布局 | 推荐页面是2D网格而非1D列表 | 传统排序方法不适用 |

### 1.2 传统方法的局限

> "Most existing recommender systems consider the recommendation procedure as a static process and make recommendations following a fixed greedy strategy. However, these approaches may fail in capturing the dynamic nature of the users' preferences."

**翻译**：大多数现有推荐系统将推荐过程视为静态过程，采用固定的贪心策略进行推荐。然而，这些方法可能无法捕捉用户偏好的动态特性。

**问题1：静态策略**
- 无法捕捉用户偏好的动态变化
- 无法根据实时反馈更新策略

**问题2：物品相似性**
> "These approaches recommend items based on the same state, which leads to the recommended items to be similar. In practice, a bundling of complementary items may receive higher rewards than recommending all similar items."

**翻译**：这些方法基于相同状态推荐物品，导致推荐的物品相似。在实践中，互补物品的组合可能比推荐所有相似物品获得更高的奖励。

**问题3：展示策略分离**
> "The set of items and the display strategy are generated separately; hence they may be not optimal to each other."

**翻译**：物品集合和展示策略是分开生成的，因此它们可能对彼此不是最优的。

### 1.3 RL的优势

**两大优势**：
1. **持续更新策略**：根据用户实时反馈持续更新推荐策略
2. **最大化长期奖励**：识别即时奖励小但对未来贡献大的物品

---

## 二、核心创新点

### 2.1 Page-wise推荐：同时优化物品和展示

> "We introduce a principled approach to generate a set of complementary items and properly display them in one 2-D recommendation page simultaneously."

**翻译**：我们引入了一种原则性方法，同时生成一组互补物品并在一个2D推荐页面中正确展示它们。

**核心思想**：
- 联合生成互补物品集合
- 联合优化2D页面展示策略
- 同时考虑物品选择和展示布局

### 2.2 DeepPage框架：基于Actor-Critic的深度RL

> "We propose a page-wise recommendation framework DeepPage, which can jointly optimize a page of items by incorporating real-time feedback from users."

**翻译**：我们提出了一个页面级推荐框架DeepPage，可以通过结合用户的实时反馈来联合优化一页物品。

**框架优势**：
- 适用于大且动态的动作空间
- 减少冗余计算
- 支持实时更新

### 2.3 为什么选择Actor-Critic而非DQN

> "The Actor-Critic architecture is preferred from the studied problem since it is suitable for large and dynamic action space, and can also reduce redundant computation simultaneously compared to alternative architectures."

**翻译**：Actor-Critic架构更适合所研究的问题，因为它适用于大且动态的动作空间，并且与替代架构相比可以同时减少冗余计算。

**架构对比**：
| 架构 | 输入 | 输出 | 适用场景 | 问题 |
|------|------|------|----------|------|
| DQN (a) | 状态 | 所有动作Q值 | 小/固定动作空间 | 无法处理大动态动作空间 |
| DQN (b) | 状态-动作对 | 单个Q值 | 需要遍历所有动作 | 计算代价高 |
| Actor-Critic | 状态 | 确定性动作 | 大动态动作空间 | 适合推荐系统 |

---

## 三、问题建模

### 3.1 MDP定义

**MDP五元组** $\langle S, A, P, R, \gamma \rangle$：

| 要素 | 定义 | 推荐系统含义 |
|------|------|--------------|
| $S$ | 状态空间 | 用户当前偏好（基于浏览历史） |
| $A$ | 动作空间 | 推荐一页M个物品 $a = \{a_1, ..., a_M\}$ |
| $P$ | 转移函数 | 状态转移 $p(s'|s, a)$ |
| $R$ | 奖励函数 | 用户反馈（跳过/点击/购买） |
| $\gamma$ | 折扣因子 | 未来奖励折扣 |

### 3.2 目标

**寻找最优策略** $\pi: S \rightarrow A$，最大化累积奖励：
$$\max_\pi \mathbb{E}\left[ \sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \right]$$

### 3.3 Q值函数

**Bellman方程**：
$$Q^*(s, a) = \mathbb{E}_{s'}\left[ r + \gamma \max_{a'} Q^*(s', a') | s, a \right]$$

**Actor-Critic中的Q值**：
$$Q(s, a) = \mathbb{E}_{s'}\left[ r + \gamma Q(s', a') | s, a \right]$$

---

## 四、DeepPage框架详解

### 4.1 整体架构

**Actor-Critic框架**：
- **Actor**：输入状态 $s$，输出确定性动作 $a = \{a_1, ..., a_M\}$
- **Critic**：输入状态-动作对 $(s, a)$，输出Q值判断

### 4.2 Actor架构

**三大挑战**：
1. 设置新会话的初始偏好
2. 学习当前会话的实时偏好
3. 联合生成推荐物品和2D页面展示

**Encoder-Decoder架构**：

#### 4.2.1 初始状态生成（Encoder）

**输入**：用户在当前会话前最后点击/购买的物品 $\{e_1, ..., e_N\}$

**模型**：GRU捕捉用户顺序行为

**GRU更新门**：
$$z_t = \sigma(W_z E_t + U_z h_{t-1})$$

**GRU重置门**：
$$r_t = \sigma(W_r E_t + U_r h_{t-1})$$

**GRU激活**：
$$h_t = (1 - z_t) h_{t-1} + z_t \hat{h}_t$$

**候选激活**：
$$\hat{h}_t = \tanh[W E_t + U(r_t \cdot h_{t-1})]$$

**输出**：初始状态 $s_{ini} = h_t$

#### 4.2.2 实时状态生成（Encoder）

**输入**：每个推荐页面的物品表示和用户反馈 $\{x_1, ..., x_M\}$

**物品表示**：
$$x_i = (e_i, c_i, f_i)$$

其中：
- $e_i$：物品嵌入
- $c_i$：物品类别（one-hot）
- $f_i$：用户反馈（跳过/点击/购买）

**嵌入变换**：
$$X_i = \text{concat}(E_i, C_i, F_i) = \tanh[\text{concat}(W_E e_i + b_E, W_C c_i + b_C, W_F f_i + b_F)]$$

**2D页面表示**：
- 将物品表示重塑为2D网格（类似图像）
- 使用CNN学习空间展示策略

**CNN处理**：
$$p_t = \text{conv2d}(P_t)$$

**GRU捕捉实时偏好**：
- 输入：$\{p_1, ..., p_T\}$
- 初始状态：$s_{ini}$
- 输出：实时状态 $s_{cur}$

#### 4.2.3 注意力机制

**目的**：捕捉用户在不同页面的注意力分布

**注意力权重**：
$$\alpha_t = \frac{\exp(f(s_{cur}, p_t))}{\sum_{t'=1}^{T} \exp(f(s_{cur}, p_{t'}))}$$

**加权状态**：
$$s = \sum_{t=1}^{T} \alpha_t p_t$$

#### 4.2.4 Decoder：生成推荐页面

**输出**：一页M个物品及其展示策略

**方法**：
- 使用全连接层生成物品嵌入
- 使用CNN学习2D展示策略

### 4.3 Critic架构

**输入**：状态-动作对 $(s, a)$

**输出**：Q值 $Q(s, a)$

**作用**：判断推荐是否匹配用户偏好

**更新**：根据Actor的表现更新Q值估计

---

## 五、关键技术细节

### 5.1 物品嵌入

**预训练方法**：
- 将物品视为单词
- 将一个推荐会话中的点击物品视为句子
- 使用Word2Vec训练物品嵌入

**应用**：
- 搜索、排序、竞价、推荐

### 5.2 类别嵌入

**目的**：捕捉不同类别物品的用户偏好

**方法**：
- 物品类别one-hot编码
- 嵌入层转换为低维稠密向量

### 5.3 反馈嵌入

**目的**：捕捉用户对物品的兴趣/反馈

**方法**：
- 用户反馈one-hot编码（跳过/点击/购买）
- 嵌入层转换为低维稠密向量

### 5.4 CNN学习展示策略

**动机**：
> "Eye-tracking studies show that rather than scanning a page in a linear fashion, users do page chunking, i.e., they partition the 2-D page into chunks, and browse the chunk they prefer more."

**翻译**：眼动研究表明，用户不是以线性方式扫描页面，而是进行页面分块，即将2D页面划分为块，并浏览他们更喜欢的块。

**方法**：
- 将推荐页面视为2D网格（类似图像）
- 使用CNN学习空间展示策略
- 发现复杂的空间相关性

---

## 六、实验结果

### 6.1 数据集

- 真实电商数据集
- 用户浏览历史和反馈

### 6.2 评估指标

| 指标 | 说明 |
|------|------|
| 点击率 (CTR) | 用户点击推荐物品的比例 |
| 转化率 | 用户购买推荐物品的比例 |
| 长期奖励 | 累积用户参与度 |

### 6.3 性能对比

**对比方法**：
- 传统监督学习方法
- 标准DQN
- 不带注意力机制的DeepPage

**关键发现**：
- DeepPage显著优于传统方法
- 注意力机制提升性能
- 联合优化物品和展示策略有效

---

## 七、与Survey论文的联系

### 7.1 作为Page-wise推荐的实践案例

本论文是Survey论文中提到的"Page-wise推荐"的具体实践案例，详细解决了2D页面推荐的RL问题。

### 7.2 关键技术对应关系

| Survey概念 | 本论文实现 |
|-------------|--------------|
| Page-wise推荐 | DeepPage框架 |
| Actor-Critic | DDPG架构 |
| 2D展示策略 | CNN学习空间布局 |
| 实时反馈 | GRU捕捉动态偏好 |

### 7.3 与其他论文的对比

| 维度 | DRN | SLATE Q | DeepPage |
|------|-----|---------|----------|
| 算法 | DQN | Q-learning | Actor-Critic |
| 推荐形式 | 单物品 | Slate | 2D页面 |
| 展示策略 | 无 | 无 | CNN学习 |
| 动作空间 | 中等 | 大 | 大且连续 |

---

## 八、工程实践启示

### 8.1 架构选择

**Actor-Critic vs DQN**：
- 大且动态的动作空间选择Actor-Critic
- 避免遍历所有动作的计算代价

### 8.2 特征工程

**三类嵌入**：
- 物品嵌入：预训练Word2Vec
- 类别嵌入：捕捉类别偏好
- 反馈嵌入：捕捉用户兴趣

### 8.3 2D展示优化

**CNN优势**：
- 学习空间相关性
- 捕捉用户分块浏览行为
- 优化页面布局

---

## 九、论文引用

```
Xiangyu Zhao, Long Xia, Liang Zhang, Zhuoye Ding, Dawei Yin, Jiliang Tang
Deep Reinforcement Learning for Page-wise Recommendations
RecSys '18, October 2–7, 2018, Vancouver, BC, Canada
```
