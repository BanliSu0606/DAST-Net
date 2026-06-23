# DAST-Net: Dual-stream Adaptive Spatio-Temporal Network

### Environment
------
The project is developed under the following environment:
- python=3.8
- pytorch=2.0.0
- torchvision=0.15.0
- cudatoolkit=11.8
- numpy=1.24.3
- scipy=1.10.1
- matplotlib=3.7.1
- pyyaml=6.0
- tensorboard=2.13.0 


### Data Preparation
------
Download all the data.

Datasets for Human3.6M:
[Human3.6m](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).

Datasets for HumanEva-I:
[HumanEva-I](http://humaneva.is.tue.mpg.de/)

Datasets for the Laboratory Assembly Dataset:
[The Laboratory Assembly Dataset]:The Laboratory Assembly Dataset is not publicly available because the data are part of an ongoing study, but we provide the exact data splits and preprocessing steps. Requests to access the dataset should be directed to the corresponding author.

Final './data' directory structure is shown below:

```
data
├── assemble_data
│   ├── assemble_test_data.npz
│   └── assemble_train_data.npz
├── data_3d_humaneva15.npz
└── data_3d_humaneva15_test.npz
├── data_3d_h36m.npz
└── data_3d_h36m_test.npz
```


### Training
------
Train on the dataset.

Train on H3.6M
```
python exp.py --cfg h36m_25_100 --mode train --seed 0
```

Train on HumanEva-I
```
python exp.py --cfg humaneva_25_100 --mode train --seed 0
```

Train on the Laboratory Assembly Dataset
```
python exp.py --cfg asb_st_0 --mode train --seed 0
python exp.py --cfg asb_st_1 --mode train --seed 1
python exp.py --cfg asb_st_2 --mode train --seed 2
python exp.py --cfg asb_st_3 --mode train --seed 3
python exp.py --cfg asb_st_4 --mode train --seed 4
```

After running the command, a directory named `<cfg> ` is created in the `./results` directory.


## Evaluation
------
Evaluate on the dataset.

Evaluate on H3.6M:
MPJPE, FDE
```
python eval_humaneva_h36m.py --cfg h36m_25_100 --mode stats --use_best_model True
```
MBLE, Parameters, Latency, FLOPs
```
python eval_humaneva_h36m.py --cfg h36m_25_100 --mode extended --use_best_model True
```

Evaluate on HumanEva-I:
MPJPE, FDE
```
python eval_humaneva_h36m.py --cfg humaneva_25_100 --mode stats --use_best_model True
```
MBLE, Parameters, Latency, FLOPs
```
python eval_humaneva_h36m.py --cfg humaneva_25_100 --mode extended --use_best_model True
```

Evaluate on the Laboratory Assembly Dataset:
Single random seed result
MPJPE, FDE
```
python eval_asb.py --cfg asb_st_0 --mode stats --use_best_model True  --seed 0
python eval_asb.py --cfg asb_st_1 --mode stats --use_best_model True  --seed 1
python eval_asb.py --cfg asb_st_2 --mode stats --use_best_model True  --seed 2
python eval_asb.py --cfg asb_st_3 --mode stats --use_best_model True  --seed 3
python eval_asb.py --cfg asb_st_4 --mode stats --use_best_model True  --seed 4
```

MBLE, Parameters, Latency, FLOPs
```
python eval_asb.py --cfg asb_st_0 --mode extended --use_best_model True  --seed 0
python eval_asb.py --cfg asb_st_1 --mode extended --use_best_model True  --seed 1
python eval_asb.py --cfg asb_st_2 --mode extended --use_best_model True  --seed 2
python eval_asb.py --cfg asb_st_3 --mode extended --use_best_model True  --seed 3
python eval_asb.py --cfg asb_st_4 --mode extended --use_best_model True  --seed 4
```

five random seeds result (the mean and standard deviation and 95% CI of MPJPE, FDE, MBLE)
```
python eval_asb_multi_seeds.py
``

## Visualization
motion_pred
├── utils
     ├── visualizationasb.py
     └── visualizationh36m.py
     └── visualizationhumaneva.py

H3.6M:
```
python visualizationh36m.py
```

HumanEva-I:
```
python visualizationhumaneva.py
```

Laboratory Assembly Dataset:
```
python visualizationasb.py
```

## Per-frame MPJPE comparison curves for actions
H3.6M:
```
python eval_perframe_allactions.py  --dataset  h36m   --cfg h36m_25_100
```

HumanEva-I:
```
python eval_perframe_allactions.py  --dataset  humaneva  --cfg humaneva_25_100
```

