# MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries （多跳查询的基准检索增强生成）
[yixuantt/MultiHop-RAG: Repository for "MultiHop-RAG: A Dataset for Evaluating Retrieval-Augmented Generation Across Documents" (COLM 2024)](https://github.com/yixuantt/MultiHop-RAG/)

[2401.15391](https://arxiv.org/pdf/2401.15391)

解决多条查询的问题，公布了一**个多跳查询的数据集**。

多条查询示例：

例如，查询 “在谷歌、苹果和英伟达中，哪家公司在 2023 年第三季度报告中报告的利润率最大？” 需要 1）从这三家公司的报告中检索与利润率相关的证据片段；

2）通过比较和推理多个检索到的证据片段来生成答案。

这与单跳查询不同，例如 “谷歌在 2023 年第三季度报告中的利润率是多少”，其答案可以直接从单个证据片段中得出。





**MultiHop-RAG多跳查询数据集**的构建：

- **数据收集：** 选择 2023年9 月 26 日至 2023 年 12 月 26 日期间发布的新闻文章，只保留标记长度大于或等于 1024 的文章，保存新闻文章，标题、发布日期、作者、类别、URL 和新闻来源。
- **要素提取：** 对于每篇文章，使用经过训练的语言模型提取事实或观点句子。这些事实句子稍后用作回答多跳查询的证据。
- **主张、桥接实体、桥接主题、生成：** 使用 GPT-4 基于证据集自动生成高质量的多跳查询。但是一些证据片段使用代词来指代主体，并且在文本中缺少实际的实体。为了解决这个问题，使用 GPT-4 对证据进行改写，我们将其称为主张，
- **桥实体和桥主题：** 证据之间的共享实体或主题被称为桥实体或桥主题。这些桥实体或桥主题可用于连接不同的证据，从中得出多跳查询的答案。



作者将多跳查询分为**四类**：

**推理查询(Inference query)**：需要从证据集合中推理出答案。比如：苹果公司的供应链风险在 2019 年年度报告还是 2020 年年度报告中有讨论？

**比较查询(Comparison query)**：需要比较证据集合中的事实。比如：Netflix 和 Google 在 2023 年哪个报告的收入更高？

**时序查询(Temporal query)**：需要分析检索到的证据块的时序信息。比如：苹果公司推出 AirTag 追踪设备是在第五代 iPad Pro 发布之前还是之后？

**空查询(Null query)**：答案无法从检索集合中得出。空查询用于评估语言模型在缺乏相关证据时是否会产生幻觉。无法查询到



针对这四种问题，分别写提示词生成问题。







