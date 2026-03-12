# 02_Val_2018_WWW_DRN_News_Recommendation

**论文信息**
- 标题：DRN: A Deep Reinforcement Learning Framework for News Recommendation
- 作者：Guanjie Zheng, Fuzheng Zhang, Zihan Zheng, Nicholas Jing Yuan, Xing Xie, Zhenhui Li
- 单位：Pennsylvania State University, Microsoft Research Asia
- 发表：WWW 2018

---

## 一、核心痛点与动机 (The "Why")

### 1.1 在线新闻推荐的三大挑战

> "Online personalized news recommendation is a highly challenging problem due to dynamic nature of news features and user preferences."

**挑战1：动态性**
> "news become outdated very fast. In our dataset, average time between time that one piece of news is published and time of its last click is 4.1 hours. Therefore, news features and news candidate set are changing rapidly."

**翻译**：新闻过时非常快。在我们的数据集中，一篇新闻发布时间和其最后一次点击时间平均仅4.1小时。因此，新闻特征和新闻候选集变化迅速。

**挑战2：用户兴趣漂移**
> "users' interest on different news might evolve during time. For instance, Figure 1 displays that categories of news that one user has read in 10 weeks. During first few weeks, this user prefers to read about 'Politics', but his interest gradually moves to 'Entertainment' and 'Technology' over time."

**翻译**：用户对不同新闻的兴趣可能会随时间演化。例如，图1显示了一个用户在10周内阅读的新闻类别。在前几周，该用户更喜欢阅读"政治"，但他的兴趣逐渐转向"娱乐"和"科技"。

**挑战3：反馈不完整**
> "current recommendation methods [23,35,36,43] usually only consider user click / no click labels or ratings as users' feedback. However, how soon one user will return to this service will also indicate how satisfied this user is with recommendation."

**翻译**：当前推荐方法通常只考虑用户点击/不点击标签或评分作为用户反馈。然而，用户多久后会返回该服务也将表明该用户对推荐的满意程度。

### 1.2 现有方法的三大问题

> "Although there are some online recommendation methods [11,24] that can capture dynamic change of news features and user preference through online model updates, they only try to optimize current reward (e.g., Click Through Rate), and hence ignore what effect that current recommendation might bring to future."

**问题1：仅优化当前奖励**
- 传统方法只优化即时奖励（如点击率），忽略推荐对未来奖励的影响

**问题2：反馈信息不完整**
- 很少有工作尝试利用用户返回模式来帮助改善推荐
- 只考虑最近返回间隔，无法在任意时间估计用户活跃度

**问题3：探索策略损害准确性**
> "State-of-art reinforcement learning methods usually apply simple-greedy strategy or Upper Confidence Bound (UCB). However, both strategies could harm recommendation performance to some extent in a short period. ϵ-greedy strategy may recommend totally unrelated items, while UCB can not get a relatively accurate reward estimation for an item until this item has been tried several times."

**翻译**：最先进的强化学习方法通常应用简单的贪心策略或上置信界(UCB)。然而，这两种策略在短期内都可能对推荐性能造成一定程度的损害。$\epsilon$-贪心策略可能会推荐完全无关的物品，而UCB在某个物品被尝试多次之前无法获得相对准确的奖励估计。

---

## 二、核心创新点

### 2.1 DQN框架：同时考虑即时和长期奖励

> "we propose a Deep Reinforcement Learning framework that can help to address these three challenges in online personalized news recommendation. First, in order to better model dynamic nature of news characteristics and user preference, we propose to use Deep Q-Learning (DQN) framework. This framework can consider current reward and future reward simultaneously."

**翻译**：我们提出了一个深度强化学习框架，可以帮助解决在线个性化新闻推荐中的这三个挑战。首先，为了更好地建模新闻特征和用户偏好的动态特性，我们提出使用深度Q学习(DQN)框架。该框架可以同时考虑当前奖励和未来奖励。

**关键优势**：
| 特性 | 描述 |
|------|------|
| 在线更新 | DQN结构能够处理新闻特征和用户偏好的快速变化 |
| 可扩展性 | 使用连续状态和动作表示，易于扩展到大规模系统 |
| 未来奖励建模 | 显式建模未来奖励，而非仅优化即时点击率 |

### 2.2 用户活跃度：捕捉更多用户反馈

> "we consider user return as another form of user feedback information, by maintaining an activeness score for each user. Different from existing work [48] that can only consider most recent return interval, we consider multiple historical return interval information to better measure user feedback."

**翻译**：我们将用户返回视为用户反馈信息的另一种形式，为每个用户维护一个活跃度分数。与现有工作[48]不同，后者只能考虑最近的返回间隔，我们考虑多个历史返回间隔信息以更好地衡量用户反馈。

**关键公式**：
$$Activeness(u) = f(\text{ReturnInterval}_1, \text{ReturnInterval}_2, ..., \text{ReturnInterval}_n)$$

**优势**：
- 可以在任意时间估计用户活跃度（不仅当用户返回时）
- 提供比点击/不点击标签更丰富的用户反馈信息
- 支持经验回放更新

### 2.3 Dueling Bandit Gradient Descent：新颖的探索策略

> "we propose to apply Dueling Bandit Gradient Descent (DBGD) method for exploration, by choosing random item candidates in the neighborhood of current recommender."

**翻译**：我们提出应用Dueling Bandit梯度下降(DBGD)方法进行探索，通过在当前推荐器的邻域中选择随机物品候选项。

**核心思想**：
- 在当前推荐物品的邻域内选择随机候选
- 避免推荐完全无关的物品（如$\epsilon$-贪心策略可能的问题）
- 保持推荐多样性

**与传统方法对比**：
| 方法 | 优点 | 缺点 |
|------|------|------|
| $\epsilon$-greedy | 简单高效 | 可能推荐完全无关物品 |
| UCB | 理论保证 | 需要多次尝试才能准确估计 |
| DBGD | 平衡多样性与准确性 | 计算复杂度稍高 |

---

## 三、模型架构

### 3.1 整体框架

> "Our deep reinforcement recommender system can be shown as Figure 2."

**系统组成**：
- **Agent (智能体)**：推荐算法
- **Environment (环境)**：用户池 + 新闻池
- **State (状态)**：用户特征 + 新闻候选特征
- **Action (动作)**：推荐新闻列表
- **Reward (奖励)**：点击标签 + 用户活跃度估计

### 3.2 离线训练与在线学习

#### 3.2.1 离线阶段

**特征提取**（4类共417维）：
- **News Features**（417维one-hot特征）：
  - Headline（标题）
  - Provider（提供者）
  - Ranking（排名）
  - Entity Name（实体名称）
  - Category（类别）
  - Topic Category（主题类别）
  - Click counts（点击次数）：last 1 hour, 6 hours, 24 hours, 1 week, 1 year
- **User Features**：
  - 用户点击过的新闻特征（headline, provider, ranking, entity name, category, topic category）

**DQN训练**：
- 使用多层深度Q网络
- 输入：用户特征 + 新闻特征
- 输出：预测奖励（点击概率 + 活跃度）
- 训练数据：离线用户新闻点击日志

#### 3.2.2 在线学习阶段

**PUSH（推送）**：
> "In each timestamp (t1, t2, t3, t4, t5, ...), when a user sends a news request to system, recommendation agent G will take feature representation of current user and news candidates as input, and generate a top-k list of news to recommend L. L is generated by combining the exploitation of current model and exploration of novel items."

**翻译**：在每个时间戳(t1, t2, t3, t4, t5, ...)，当用户向系统发送新闻请求时，推荐智能体G将获取当前用户和新闻候选的特征表示作为输入，生成一个top-k新闻推荐列表L。L是通过结合当前模型的利用和新物品的探索生成的。

**FEEDBACK（反馈）**：
> "User u who has received recommended news L will give their feedback B by his clicks on this set of news."

**翻译**：收到推荐新闻L的用户u将通过在该组新闻上的点击给出他们的反馈B。

**MINOR UPDATE（次要更新）**：
> "After each timestamp (e.g., after timestamp t1), with the feature representation of previous user u and news list L, and feedback B, agent G will update the model by comparing recommendation performance of exploitation network Q and exploration network ˜Q (will be discussed in Section 4.5). If ˜Q gives better recommendation result, current network will be updated towards ˜Q. Otherwise, Q will be kept unchanged."

**翻译**：在每个时间戳(例如，在时间戳t1之后)，使用前一个用户u的特征表示和新闻列表L以及反馈B，智能体G将通过比较利用网络Q和探索网络˜Q的推荐性能来更新模型。如果˜Q给出更好的推荐结果，当前网络将向˜Q更新。否则，Q将保持不变。

**MAJOR UPDATE（主要更新）**：
> "After certain period of time TR (e.g., after timestamp t3), agent G will use user feedback and user activeness stored in memory to update the network Q."

**翻译**：在经过一定时间TR后（例如，在时间戳t3之后），智能体G将使用存储在记忆中的用户反馈和用户活跃度来更新网络Q。

**经验回放**：
> "Here, we use experience replay technique to update the network. Specifically, agent G maintains a memory with recent historical click and user activeness records. When each update happens, agent G will sample a batch of records to update the model."

**翻译**：这里我们使用经验回放技术来更新网络。具体来说，智能体G维护一个包含最近历史点击和用户活跃度记录的记忆。每次更新发生时，智能体G将采样一批记录来更新模型。

---

## 四、特征工程

### 4.1 新闻特征（417维）

| 特征类别 | 维度 | 说明 |
|----------|------|------|
| Headline | one-hot | 新闻标题 |
| Provider | one-hot | 内容提供者 |
| Ranking | one-hot | 排名位置 |
| Entity Name | one-hot | 实体名称 |
| Category | one-hot | 新闻类别 |
| Topic Category | one-hot | 主题类别 |
| Click Counts | 6维 | 点击次数：1h, 6h, 24h, 1w, 1y |

### 4.2 用户特征

- **历史点击新闻特征**：用户在1小时、6小时、24小时、1周、1年内点击过的新闻的headline, provider, ranking, entity name, category, topic category

**特征设计原则**：
- 捕捉用户近期兴趣偏好
- 区分短期兴趣和长期兴趣
- 支持不同时间粒度的特征

---

## 五、奖励设计

### 5.1 奖励组成

> "Specifically, reward is composed of click labels and estimation of user activeness."

**翻译**：具体来说，奖励由点击标签和用户活跃度估计组成。

$$Reward = \alpha \cdot \text{ClickLabel} + \beta \cdot \text{Activeness}$$

### 5.2 用户活跃度估计

**关键洞察**：
- 用户返回App的频率反映了用户对推荐系统的满意程度
- 活跃度高的用户更可能持续使用推荐服务
- 活跃度低的用户可能已经流失

**计算方法**：
- 维护每个用户的历史返回间隔记录
- 使用多个时间窗口（1h, 6h, 24h, 1w, 1y）
- 综合多个时间粒度的信息

---

## 六、探索策略

### 6.1 Dueling Bandit Gradient Descent (DBGD)

> "This exploration strategy can avoid recommending totally unrelated items and hence maintain better recommendation accuracy."

**核心思想**：
1. **邻域选择**：在当前推荐物品的邻域内选择随机候选项
2. **梯度下降**：通过优化探索方向，逐步发现更好的物品

**与传统方法对比**：

| 方法 | 策略 | 优点 | 缺点 |
|------|------|------|------|
| $\epsilon$-greedy | 随机选择 | 可能推荐完全无关物品，伤害用户体验 |
| UCB | 基于置信界选择 | 需要多次尝试才能准确估计 |
| DBGD | 邻域内梯度下降 | 平衡多样性与准确性，避免极端探索 |

**算法流程**：
1. 计算当前推荐物品的邻域
2. 在邻域内选择随机物品作为探索候选项
3. 评估探索候选项的质量
4. 通过梯度下降调整探索策略

---

## 七、实验结果

### 7.1 数据集

- **离线数据集**：商业新闻推荐应用的历史日志
- **在线环境**：实际部署的商业新闻推荐应用

### 7.2 评估指标

| 指标 | 说明 |
|------|------|
| CTR (Click-Through Rate) | 点击率 |
| 用户活跃度 | 用户返回频率 |
| 长期收益 | 累积奖励 |

### 7.3 性能对比

> "Extensive experiments are conducted on offline dataset and online production environment of a commercial news recommendation application and have shown superior performance of our methods."

**翻译**：在离线数据集和商业新闻推荐应用的在线生产环境中进行了大量实验，证明了我们方法的优越性能。

**优势总结**：
- 在线学习机制适应新闻快速变化
- 用户活跃度提供更丰富的反馈信息
- DBGD探索策略平衡多样性与准确性

---

## 八、与Survey论文的联系

### 8.1 作为Value-based方法的实践案例

本论文是Survey论文中提到的"Value-based方法"和"DQN方法"的具体实践案例，验证了DQN在推荐系统中的可行性。

### 8.2 关键技术对应关系

| Survey概念 | 本论文实现 |
|-------------|--------------|
| DQN框架 | 多层深度Q网络，预测即时+长期奖励 |
| Model-free方法 | 经验回放，在线更新 |
| 探索策略 | Dueling Bandit Gradient Descent |
| 状态表示 | 连续特征表示（417维新闻特征 + 用户特征） |
| 奖励设计 | 点击标签 + 用户活跃度 |

### 8.3 创新点验证

| Survey提出的挑战 | 本论文解决方案 |
|----------------|--------------|
| 动态性 | DQN在线更新机制 |
| 反馈不完整 | 用户活跃度作为额外反馈 |
| 探索代价 | DBGD邻域探索策略 |

---

## 九、论文引用

```
Guanjie Zheng, Fuzheng Zhang, Zihan Zheng, Nicholas Jing Yuan, Xing Xie, Zhenhui Li
DRN: A Deep Reinforcement Learning Framework for News Recommendation
WWW 2018, April 23–27, 2018, Lyon, France
```
