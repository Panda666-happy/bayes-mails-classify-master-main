# bayes-mails-classify-master-main

# 项目名称
## 核心功能说明

### 算法基础：多项式朴素贝叶斯分类器
本仓库采用**多项式朴素贝叶斯分类器**实现文本分类任务，其核心假设是：
1. ​**特征条件独立性**：假设各特征（如单词）在给定类别下相互独立，即特征联合概率可分解为独立概率的乘积：
   $$P(x_1,x_2,...,x_n|y) = \prod_{i=1}^n P(x_i|y)$$
2. ​**贝叶斯定理应用**：通过先验概率和条件概率计算后验概率：
   $$P(y|x) = \frac{P(x|y)P(y)}{P(x)} \propto P(y)\prod_{i=1}^n P(x_i|y)$$
3. ​**邮件分类实现**：
   - 训练阶段：统计每个单词在不同类别邮件中的出现频率（+平滑处理避免零概率）[5,7](@ref)
   - 预测阶段：计算新邮件属于各类别的对数概率，取最大值作为预测结果[8,10](@ref)

### 数据处理流程
1. ​**分词处理**：
   - 使用正则表达式去除特殊字符，采用最大匹配法或统计分词（如Jieba）[11,13](@ref)
   - 示例代码：
     ```python
     import jieba
     words = jieba.lcut("原始文本")
     ```
2. ​**停用词过滤**：
   - 加载停用词表（如NLTK/自定义表），过滤无意义词汇[14,16](@ref)
   - 实现逻辑：
     ```python
     stop_words = set(["的", "是", "在"])
     filtered_words = [w for w in words if w not in stop_words]
     ```

### 特征构建过程
#### 高频词特征选择
- ​**数学表达**：选取文档频率(DF)最高的N个词作为特征
  $$DF(t) = \text{包含词t的文档数}$$
- ​**实现差异**：直接统计词频排序，计算复杂度低但忽略词区分能力[17](@ref)

#### TF-IDF特征加权
- ​**数学形式**：
  $$TFIDF(t,d) = TF(t,d) \times \log\left(\frac{N}{DF(t)+1}\right)$$
  其中$TF(t,d)$为词t在文档d中的出现次数[18,24](@ref)
- ​**优势**：同时考虑词频(TF)和逆文档频率(IDF)，更好体现词重要性

#### 特征模式切换方法
1. ​**参数化设计**：通过配置参数选择特征模式：
   ```python
   if mode == "high_freq":
       features = top_k_words(corpus, k=1000)
   else:  # tfidf
       vectorizer = TfidfVectorizer()
       features = vectorizer.fit_transform(corpus)
