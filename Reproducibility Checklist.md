# Reproducibility Checklist for DAST-Net

---

## 1. Environment Setup

- Python 3.8, PyTorch 2.0.0, CUDA 11.8, NVIDIA RTX 4090

---

## 2. Data Preparation

| Dataset | Availability | Split Provided |
|---------|--------------|----------------|
| Human3.6M | Public | ✅ |
| HumanEva-I | Public | ✅ |
| Lab Assembly | Not Public | ✅ (splits provided) |

Expected data directory: `./data/` with `.npz` files for each dataset.

---

## 3. Training Commands

| Dataset | Command | Seed |
|---------|---------|------|
| Human3.6M | `python exp.py --cfg h36m_25_100 --mode train --seed 0` | 0 |
| HumanEva-I | `python exp.py --cfg humaneva_25_100 --mode train --seed 0` | 0 |
| Lab Assembly | `python exp.py --cfg asb_st_X --mode train --seed X` (X=0,1,2,3,4) | 0-4 |

**Early stopping:** 15 epochs (H3.6M), 20 epochs (HumanEva-I), 10 epochs (Lab Assembly).

---

## 4. Evaluation Commands

**MPJPE & FDE:**

| Dataset | Command |
|---------|---------|
| Human3.6M | `python eval_humaneva_h36m.py --cfg h36m_25_100 --mode stats --use_best_model True` |
| HumanEva-I | `python eval_humaneva_h36m.py --cfg humaneva_25_100 --mode stats --use_best_model True` |
| Lab Assembly | `python eval_asb.py --cfg asb_st_X --mode stats --use_best_model True --seed X` |
| Lab Assembly (multi-seed) | `python eval_asb_multi_seeds.py` |

**MBLE, Parameters, Latency, FLOPs:** Use `--mode extended` instead of `--mode stats`.

---

## 5. Visualization

| Dataset | Command |
|---------|---------|
| Human3.6M | `python visualizationh36m.py` |
| HumanEva-I | `python visualizationhumaneva.py` |
| Lab Assembly | `python visualizationasb.py` |

**Per-frame MPJPE curves:**

| Dataset | Command |
|---------|---------|
| Human3.6M | `python eval_perframe_allactions.py --dataset h36m --cfg h36m_25_100` |
| HumanEva-I | `python eval_perframe_allactions.py --dataset humaneva --cfg humaneva_25_100` |

---

## 6. Pre-trained Models

- **Download:** (https://drive.google.com/drive/folders/1Ai3AU6RqX_ANFonh6qG49ZB3sPekikXU?usp=sharing)
- Replace the `./results` folder with the downloaded checkpoints.

---

## 7. Expected Results

| Dataset | MPJPE | FDE | MBLE |
|---------|-------|-----|------|
| Human3.6M | 0.3999 | 0.4796 | 0.0375 |
| HumanEva-I | 0.3649 | 0.3237 | 0.0249 |
| Lab Assembly | 0.3112 ± 0.0394 | 0.3524 ± 0.0432 | 0.0410 ± 0.0024 |

---

## 8. Hardware & Timing

| Item | Specification |
|------|---------------|
| GPU | NVIDIA RTX 4090 (24GB) |
| Batch Size | 256 |
| Latency | ~5 ms per sequence |
| FLOPs | 2.43G (H3.6M) / 2.08G (HumanEva-I) / 1.91G (Lab Assembly) |

---

## 9. Code & Data Availability

| Item | Link |
|------|------|
| GitHub | https://github.com/BanliSu0606/DAST-Net |
| Release | v1.0.0 |
| Zenodo DOI | To be assigned upon publication |
| Lab Assembly Dataset | Not publicly available (ongoing study); requests to corresponding author |

---

## 10. Summary

All experiments can be reproduced by following the commands above. See README for detailed instructions.
# 欢迎使用 MarkdownPro

## 功能特点

- **实时预览**: 支持实时Markdown预览
- **语法高亮**: 完整的语法高亮支持
- **多种导出**: 支持HTML等格式导出
- **响应式设计**: 完美适配各种设备
- **多语言支持**: 中英文界面切换

## 快速开始

开始编写您的Markdown文档吧！

```javascript
console.log("Hello MarkdownPro!");
```

> 这是一个引用示例

### 列表示例

1. 有序列表项1
2. 有序列表项2
3. 有序列表项3

- 无序列表项A
- 无序列表项B
- 无序列表项C

### 表格示例

| 功能 | 状态 | 说明 |
|------|------|------|
| 实时预览 | ✅ | 已完成 |
| 语法高亮 | ✅ | 已完成 |
| 导出功能 | ✅ | 已完成 |

**祝您使用愉快！**
