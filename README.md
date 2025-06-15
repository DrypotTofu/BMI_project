# BMI Project

本项目主要围绕脑机接口（Brain-Machine Interface, BMI）相关的数学建模、神经网络（尤其是脉冲神经网络 SNN）理论与实践展开，包含以下内容：

## 目录结构

- `Note/`  
  SNN 相关理论笔记与学习资料和论文阅读报告。
- `Beamer/`  
  浙江大学风格的 LaTeX Beamer 幻灯片模板及相关演示文档，适用于学术报告和展示。
- `Example/`  
  SNN 相关的代码示例（分别以snnTorch 和 PyTorch为基础），便于理解 SNN 的实际应用与训练流程。
- `Papers/`
  SNN相关论文电子版文档。

## 主要内容

- **SNN 理论与实践**：详细笔记梳理了脉冲神经网络的背景、基本原理、常用神经元模型、信息编码方式、主流学习策略及其优缺点，并结合代码实现了 SNN 在 MNIST 数据集上的分类实验。
- **学术演示模板**：提供了美观的浙江大学 Beamer 幻灯片模板，便于撰写和展示学术报告。
- **代码示例**：包含完整的 SNN 训练与推理代码，支持自定义神经元、替代梯度等功能，适合学习和二次开发。

## 依赖环境

- Python 3.11
- PyTorch
- snnTorch
- seaborn
- matplotlib
- LaTeX（XeLaTeX，推荐 TeXLive 2022 及以上）

## 致谢

部分代码来自https://github.com/jeshraghian/snntorch

