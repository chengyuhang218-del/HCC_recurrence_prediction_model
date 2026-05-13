# 🧬 一个深度学习预测复发/非复发项目（基于bulk基因表达的）
**A lightweight Python toolkit for differential gene expression analysis**
# HCC Recurrence Prediction Project

基于 **GSE76427** 的肝细胞癌（HCC）复发/非复发预测项目说明仓库。

---

## 项目简介

本项目围绕「肝癌术后是否复发」这一二分类任务，整理了从数据下载、清洗、标签构建到模型训练（Attention 思路）的完整流程文档，便于复现与二次开发。

核心目标：

- 从 GEO 数据集提取可用于复发预测的真实肿瘤样本。
- 构建可靠的 RFS（复发）标签。
- 形成可直接进入 Python 模型的数据输入。
- 提供可读性高、可直接复制使用的项目文档。

---

## 🚀 Overview
## 仓库主要文件

**wustDEG** is a simple yet extensible Python package designed for differential gene expression (DEG) analysis from count-based transcriptomic data.
- `hcc_recurrence_project_tutorial_html_v_2.html`：原始 HTML 教程文档。
- `PROJECT_DOCUMENTATION.md`：项目总文档（你可以按需追加内容，不需要覆盖）。

It provides a streamlined workflow to:
---
## 流程概览

1. 下载 GSE76427 表达矩阵与临床信息。  
2. 筛选肿瘤样本并构建复发标签。  
3. Probe ID 映射为 Gene Symbol。  
4. 清理空基因名、重复基因、NA/Inf。  
5. 导出清洗后的表达矩阵与标签。  
6. 在 Python 中读取并进行特征筛选与 Attention 建模。  

## 推荐使用方式

1. 先阅读 `hcc_recurrence_project_tutorial_html_v_2.md`。
2. 在你自己的数据路径下执行 R / Python 代码块。

---
## 📦 Installation（还未上传，项目还未完善）

Install from PyPI:
```bash
pip install wustDEG
```
---

## ✨ Features
## 说明

📅 03/29/2026，天气 🌧️🌧️ 武汉科技大学 📍
- 本仓库以**文档可复用性**为主，优先保证步骤清晰、结构完整。
