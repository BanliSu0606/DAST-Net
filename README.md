# DAST-Net: Dual-stream Adaptive Spatio-Temporal Network

## Environment

The project is developed under the following environment:

| Package | Version |
|---------|---------|
| Python | 3.8 |
| PyTorch | 2.0.0 |
| TorchVision | 0.15.0 |
| CUDA Toolkit | 11.8 |
| NumPy | 1.24.3 |
| SciPy | 1.10.1 |
| Matplotlib | 3.7.1 |
| PyYAML | 6.0 |
| TensorBoard | 2.13.0 |

---

## Data Preparation

Download all the datasets:

| Dataset | Source |
|---------|--------|
| Human3.6M | [Official Website](http://vision.imar.ro/human3.6m/description.php) (Exponential map version available [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip)) |
| HumanEva-I | [Official Website](http://humaneva.is.tue.mpg.de/) |
| Laboratory Assembly Dataset | Not publicly available (part of an ongoing study). Requests should be directed to the corresponding author. Data splits and preprocessing steps are provided. |

### Final Directory Structure

After preparation, the `./data` directory should look like this:

```
data
├── assemble_data
│   ├── assemble_test_data.npz
│   └── assemble_train_data.npz
├── data_3d_humaneva15.npz
├── data_3d_humaneva15_test.npz
├── data_3d_h36m.npz
└── data_3d_h36m_test.npz
```

---

## Training

Train on different datasets:

### Human3.6M
```bash
python exp.py --cfg h36m_25_100 --mode train --seed 0
```

### HumanEva-I
```bash
python exp.py --cfg humaneva_25_100 --mode train --seed 0
```

### Laboratory Assembly Dataset
```bash
python exp.py --cfg asb_st_0 --mode train --seed 0
python exp.py --cfg asb_st_1 --mode train --seed 1
python exp.py --cfg asb_st_2 --mode train --seed 2
python exp.py --cfg asb_st_3 --mode train --seed 3
python exp.py --cfg asb_st_4 --mode train --seed 4
```

> After running the commands, a directory named `<cfg>` will be created under `./results`.

---

## Evaluation

### Human3.6M

**MPJPE, FDE**
```bash
python eval_humaneva_h36m.py --cfg h36m_25_100 --mode stats --use_best_model True
```

**MBLE, Parameters, Latency, FLOPs**
```bash
python eval_humaneva_h36m.py --cfg h36m_25_100 --mode extended --use_best_model True
```

### HumanEva-I

**MPJPE, FDE**
```bash
python eval_humaneva_h36m.py --cfg humaneva_25_100 --mode stats --use_best_model True
```

**MBLE, Parameters, Latency, FLOPs**
```bash
python eval_humaneva_h36m.py --cfg humaneva_25_100 --mode extended --use_best_model True
```

### Laboratory Assembly Dataset

**Single random seed result (MPJPE, FDE)**
```bash
python eval_asb.py --cfg asb_st_0 --mode stats --use_best_model True --seed 0
python eval_asb.py --cfg asb_st_1 --mode stats --use_best_model True --seed 1
python eval_asb.py --cfg asb_st_2 --mode stats --use_best_model True --seed 2
python eval_asb.py --cfg asb_st_3 --mode stats --use_best_model True --seed 3
python eval_asb.py --cfg asb_st_4 --mode stats --use_best_model True --seed 4
```

**MBLE, Parameters, Latency, FLOPs**
```bash
python eval_asb.py --cfg asb_st_0 --mode extended --use_best_model True --seed 0
python eval_asb.py --cfg asb_st_1 --mode extended --use_best_model True --seed 1
python eval_asb.py --cfg asb_st_2 --mode extended --use_best_model True --seed 2
python eval_asb.py --cfg asb_st_3 --mode extended --use_best_model True --seed 3
python eval_asb.py --cfg asb_st_4 --mode extended --use_best_model True --seed 4
```

**Five random seeds result (mean, standard deviation, and 95% CI of MPJPE, FDE, MBLE)**
```bash
python eval_asb_multi_seeds.py
```

---

## Visualization

Visualization scripts are located in:

```
utils/
├── visualizationasb.py
├── visualizationh36m.py
└── visualizationhumaneva.py
```

### Human3.6M
```bash
python visualizationh36m.py
```

### HumanEva-I
```bash
python visualizationhumaneva.py
```

### Laboratory Assembly Dataset
```bash
python visualizationasb.py
```

---

## Per-frame MPJPE Comparison Curves

### Human3.6M
```bash
python eval_perframe_allactions.py --dataset h36m --cfg h36m_25_100
```

### HumanEva-I
```bash
python eval_perframe_allactions.py --dataset humaneva --cfg humaneva_25_100
```

---

## Pre-trained Models

Due to GitHub file size limits, the pre-trained model checkpoints are hosted on Google Drive. Please download and replace the `results` folder in the repository.
(https://drive.google.com/drive/folders/1Ai3AU6RqX_ANFonh6qG49ZB3sPekikXU?usp=sharing)

---
