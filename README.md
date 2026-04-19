# NLP Basework DIY(updating)

## 中文简介
这是一个用于练习 NLP 基础结构的 DIY 项目。核心逻辑手写，用vibe coding辅助实现了画图、日志等辅助功能。

项目目标：
- 理解数据处理、特征构建与模型训练流程，加深对NLP的学习认识
- 通过可复现实验对比不同方法在文本分类任务上的效果

当前内容：
- 基础数据读取与预处理
- 传统机器学习文本分类（BoW、N-gram）
- 深度学习文本分类（CNN、RNN、Transformer）
- 多组超参数实验与结果表导出

状态：持续更新中。目前的代码结构算法都比较简单，所以效果一般，在数据集上进行5分类任务，大概是40%多准确率，但是还是起到了学习效果。

数据来源声明：
- 本项目实验数据集来源于复旦 NLP 相关页面（飞书）：https://fudan-nlp.feishu.cn/wiki/VhCHwZiXQicIYrkbd2ocFS6Kn9c
- 数据版权与使用规范以原页面说明为准。

## English Overview
This is a DIY project for practicing core NLP building blocks.Implement core logic by hand to understand data processing, feature engineering, and training.Use vibe coding to assist engineering tasks (experiment loops, logging, result export, etc.)

Goals:
- Understand the processes of data processing, feature construction, and model training, and deepen the understanding of NLP learning
- Run reproducible comparisons across different text classification approaches

Current content:
- Basic data loading and preprocessing
- Traditional ML text classification (BoW, N-gram)
- Deep learning text classification (CNN, RNN, Transformer)
- Hyperparameter experiments and result table export

Status: work in progress.

Dataset Source Declaration:
- The dataset used in this project comes from the Fudan NLP related Feishu page:
	https://fudan-nlp.feishu.cn/wiki/VhCHwZiXQicIYrkbd2ocFS6Kn9c
- Data copyright and usage terms follow the original source page.

## Quick Start
1. Prepare Python environment and install dependencies
2. Place dataset files in project root
3. Run ML experiments via Classification_basicML.py
4. Run DL experiments via Classification_basicDL.py

## Notes
This repository focuses on learning-oriented implementations, readability, and iterative improvements rather than production optimization.
