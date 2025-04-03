# bayes-mails-classify

# 项目名称
## 核心功能说明

### 算法基础：多项式朴素贝叶斯分类器

本仓库采用**多项式朴素贝叶斯分类器**实现文本分类任务，其核心假设是：
1. ​**特征条件独立性**：假设各特征（如单词）在给定类别下相互独立，即特征联合概率可分解为独立概率的乘积：
   $$P(x_1,x_2,...,x_n|y) = \prod_{i=1}^n P(x_i|y)$$
2. ​**贝叶斯定理应用**：通过先验概率和条件概率计算后验概率：
   $$P(y|x) = \frac{P(x|y)P(y)}{P(x)} \propto P(y)\prod_{i=1}^n P(x_i|y)$$
3. ​**邮件分类实现**：
   - 训练阶段：统计每个单词在不同类别邮件中的出现频率（+平滑处理避免零概率）
   - 预测阶段：计算新邮件属于各类别的对数概率，取最大值作为预测结果

### 特征构建过程
#### 高频词特征选择
- ​**数学表达**：选取文档频率(DF)最高的N个词作为特征
  $$DF(t) = \text{包含词t的文档数}$$
- ​**实现差异**：直接统计词频排序，计算复杂度低但忽略词区分能力

#### TF-IDF特征加权
- ​**数学形式**：
  $$\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \log\left( \frac{N}{\text{DF}(t) + 1} \right)$$
  其中TF(t,d)为词t在文档d中的出现次数
- ​**优势**：同时考虑词频(TF)和逆文档频率(IDF)，更好体现词重要性

## 数据处理流程

### 1. 文本清洗

系统首先对原始邮件文本进行深度清洗：
```python
# 使用正则表达式去除标点、数字等干扰字符
line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
```

### 2. 中文分词处理

采用jieba分词工具进行精准的中文分词：
```python
# 执行分词并过滤无效词汇
line = cut(line)  # jieba分词
line = filter(lambda word: len(word) > 1, line)  # 过滤单字词
```

### 3. 样本平衡处理

针对数据不平衡问题，系统集成了SMOTE过采样技术：
```python
# 使用SMOTE算法平衡样本分布
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
vector_resampled, labels_resampled = smote.fit_resample(vector, labels)
```

#### 特征模式切换方法

1. ​**参数化设计**：通过配置参数选择特征模式：
   ```python
   if mode == "high_freq":
       features = top_k_words(corpus, k=1000)
   else:  # tfidf
       vectorizer = TfidfVectorizer()
       features = vectorizer.fit_transform(corpus)
   
# 代码运行结果

## 默认分类模式 classify1.py
![屏幕截图 2025-03-31 154212](https://github.com/user-attachments/assets/5509674f-0b84-40ed-9ad0-a2d925bb8403)

## 局部切换 class_local.py
![屏幕截图 2025-04-03 114742](https://github.com/user-attachments/assets/1bab8026-10b5-45a5-b2ad-c8b5f085b920)

## 局部切换  class_global.py
![屏幕截图 2025-04-03 114904](https://github.com/user-attachments/assets/7b7769c8-89f3-4fd8-8e76-113eee205a84)

## 样本平衡处理  sample.py
![屏幕截图 2025-04-03 115120](https://github.com/user-attachments/assets/45d3e83e-dc36-4d27-ad0a-481a5bb3a419)

## 添加全局方法选择/样本平衡处理/模型评估标准 all.py
![屏幕截图 2025-04-03 115330](https://github.com/user-attachments/assets/4d1d6a5c-641f-4aec-9135-96bfca790174)









