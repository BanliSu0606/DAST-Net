# DAST-Net: Dual-stream Adaptive Spatio-Temporal Network

### Environment
------
The project is developed under the following environment:
-PyTorch  2.0.0
-Python  3.8
-CUDA  11.8 


### Data Preparation
------
Download all the data.

Datasets for HumanEva-I:
[HumanEva-I](http://humaneva.is.tue.mpg.de/)
[The Laboratory Assembly Dataset]:The Laboratory Assembly Dataset is not publicly available because the data are part of an ongoing study. Requests to access the dataset should be directed to the corresponding author.

Final './data' directory structure is shown below:

```
data
├── assemble_data
│   ├── assemble_test_data.npz
│   └── assemble_train_data.npz
├── data_3d_humaneva15.npz
└── data_3d_humaneva15_test.npz
```


### Training
------
Train on the dataset.

Train on HumanEva-I
```
python exp.py --cfg humaneva_25_100 --mode train
```

Train on the Laboratory Assembly Dataset
```
python exp.py --cfg asb_25_100 --mode train
```

After running the command, a directory named `<cfg> ` is created in the `./results` directory.


## Evaluation
------
Evaluate on the dataset.

Evaluate on HumanEva-I:
```
python eval_humaneva.py --cfg humaneva_25_100 --mode stats --use_best_model True
```

Evaluate on HumanEva-I:
```
python eval_asb.py --cfg asb_25_100 --mode stats --use_best_model True
```

**Note**: We parallelize the process of evaluating metrics MPJPE(Mean Per-Joint Position Error) and FDE(Final Displacement Error) to speed up the process, so this part is strictly require GPU.
