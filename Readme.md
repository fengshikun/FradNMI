## Introduction

This is the offcial implementation of Paper "Enhancing Molecular Property Prediction with
Chemical Priors by Fractional Denoising"


## Requirements

The environment is composed of the following packages and versions:
```
pytorch-lightning   1.8.6
torch               1.13.1+cu116
torch-cluster       1.6.0+pt113cu116
torch-geometric     2.3.0
torch-scatter       2.1.0+pt113cu116
torch-sparse        0.6.17+pt113cu116
torch-spline-conv   1.2.1+pt113cu116
torchmetrics        0.11.4
wandb               0.15.3
numpy               1.22.4
scikit-learn        1.2.2
scipy               1.8.1
deepchem            2.7.1
ogb                 1.3.6
omegaconf           2.3.0
tqdm                4.66.2
```


The basic software and environment include Python 3.8, CUDA 11.6, Ubuntu 20.04.2 with OS version 9.4.0-1ubuntu1~20.04.2, and Linux kernel version 5.4.0-177-generic.

We ran all experiments on a server equipped with 8 NVIDIA A100-PCIE-40GB GPUs.


Additionally, we have updated a Conda environment package available at [google drive](https://drive.google.com/file/d/1X9gUELR6UAifUT7VVtgur2ZWfCGl7kcF/view?usp=sharing). You can download the environment package and unzip it into the 'envs' directory of Conda.

## Quick Start


To leverage Frad's fine-tuned model for predicting molecular quantum properties, follow these steps:

1. Prepare the molecular SMILES in format like `smiles.lst`
```
smiles
CC(C)n(c1c2CN(Cc3cn(C)nc3C)CC1)nc2-c1ncc(C)o1
C=CCN1C(SCc2nc(cccc3)c3[nH]2)=Nc(cccc2)c2C1=O
COc1ccc(Cn2c(C(O)=O)c(CNC3CCCC3)c3c2cccc3)cc1
O=C(CCc1ccc(CC(CC2)CCN2C(c2cscc2)=O)cc1)NC1CC1
Cc1nc(CCC2)c2c(N2CC(CN(C(C=C3)=O)N=C3c3ccncc3)C2)n1
CC(C1)OC(C)CN1c1nc(Nc2cc(OC)ccc2)c(cnn2C)c2n1
```

1. Generate coordinates for Molecular SMILES

```
python convert_smiles_pos.py --smiles_file=smiles.lst --output_file smiles_coord.lst
```

The generated coordinates and atom types for the input SMILES will be stored in `smiles_coord.lst`


2. Utilize the fine-tuned model for prediction

Download the fine-tuned model for either the gap property from this [URL](https://drive.google.com/file/d/14yxjvgbkRodDr6wn3qh4tqIijPMTqXCl/view?usp=sharing) or the lumo property from this [URL](https://drive.google.com/file/d/1pa2daJQk-Xvh8Mj0_YcQahbymE1BKftb/view?usp=sharing).

Execute the following command for property prediction. The prediction results will be stored in `results.csv`:

```
CUDA_VISIBLE_DEVICES=0 python scripts/test.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --dataset TestData --dataset-root smiles_coord.lst --train-size 1 --val-size 1 --layernorm-on-vec whitened --job-id gap{or lumo}_inference --dataset-arg gap{or lumo} --pretrained-model $finetuned-model --output-file results.csv
```



## Reproduce

### Assets

| Dataset   | Reference                                                                                                    |
|-----------|--------------------------------------------------------------------------------------------------------------|
| PCQM4Mv2  | [OGB Stanford](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/), [Figshare](https://figshare.com/articles/dataset/MOL_LMDB/24961485) |
| QM9       | [Figshare](https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904) |
| MD17      | [SGDML](http://www.sgdml.org/#datasets)                                                                     |
| MD22      | [SGDML](http://www.sgdml.org/#datasets)                                                                     |
| ISO17     | [Quantum Machine](http://quantum-machine.org/datasets/)                                                     |
| LBA       | [Zenodo](https://zenodo.org/records/4914718)                                                                |

Additionally, we offer the download link for the processed finetuned data at the following URL: [google drive](https://drive.google.com/drive/folders/1qe8EwXSnZ-K8dFaa5HQwWBmFpYYFe2Gn?usp=sharing)


### Pre-trained models 

All pre-trained models are uploaded to **Zenodo**: [Zenodo Link](https://zenodo.org/records/12697467)

Alternatively, individual pre-trained models can be accessed via Google Drive:

- Pretrained model for QM9: [google drive](https://drive.google.com/drive/folders/1sFH7s_L3hqW4HhR7CC8TBUKjwUeslex1?usp=sharing)
- Pretrained model for Force Predictioin(MD17, MD22, ISO17): [google drive](https://drive.google.com/drive/folders/18O-XaubUg_XMImAwnqSidaL0-TLszF3F?usp=sharing)
- Pretrained model for LBA: [google drive](https://drive.google.com/drive/folders/1Z32LO0p1MkF4NTILPzdKH2rIoRmf0ZE6?usp=sharing)




### Finetuning



#### Finetune on QM9

Below is the script for fine-tuning the QM9 task. Ensure to replace `pretrain_model_path` with the actual model path. In this script, the subtask is set to 'homo', but it can be replaced with other subtasks as well.

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_homo --dataset-arg homo  --denoising-weight 0.1 --dataset-root $datapath --pretrained-model $pretrain_model_path
```


#### Finetune on MD17
Below is the script for fine-tuning the MD17 task. Replace pretrain_model_path with the actual model path. In this script, the subtask is set to 'aspirin', but it can be replaced with other subtasks such as {'benzene', 'ethanol', 'malonaldehyde', 'naphthalene', 'salicylic_acid', 'toluene', 'uracil'}.


```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-MD17_FT-angle_9500.yaml  --job-id frad_aspirin --dataset-arg aspirin --pretrained-model $pretrain_model_path --dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss
```


#### Finetuning the MD22
Below is the script for fine-tuning the MD22 task. Replace pretrain_model_path with the actual model path. In this script, the subtask is set to 'AT-AT-CG-CG' by `--dataset-arg`, but it can be replaced with other subtasks such in ("AT-AT-CG-CG" "AT-AT" "Ac-Ala3-NHMe" "DHA" "buckyball-catcher" "double-walled_nanotube" "stachyose").
```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-MD22.yaml --batch-size 32 --inference-batch-size 32 --num-epochs 100 --lr 1e-3 --log-dir md22-AT-AT-CG-CG --dataset-arg AT-AT-CG-CG --ngpus 1 --job-id md22-AT-AT-CG-CG --pretrained-model $$pretrain_model_path --lr-schedule cosine_warmup --save-top-k 1 --save-interval 1 --test-interval 1 --seed 666 --md17 true --train-loss-type smooth_l1_loss
```

#### Finetuning the ISO17
Below is the script for fine-tuning the ISO17 task.
```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-ISO17.yaml --batch-size 256 --job-id iso17 --inference-batch-size 256 --pretrained-model $pretrain_model_path --num-epochs 50 --lr 2e-4 --log-dir iso-energy --ngpus 1  --save-top-k 1 --save-interval 1 --test-interval 1 --seed 666 --lr-schedule cosine_warmup --md17 true --train-loss-type smooth_l1_loss
```


#### Finetuning the LBA
Below is the script for fine-tuning the LBA task.
```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-LBA-FT_long_f2d.yaml --layernorm-on-vec whitened --job-id LBA --dataset-root $LBA_DATA_PATH --pretrained-model $pretrain_model_path
```


### Pretraining

Model for the QM9

```
CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-PCQM4MV2_dih_var0.04_var2_com_re.yaml --layernorm-on-vec whitened --job-id frad_pretraining --num-epochs 8 
```
Model for Atomic Forces Tasks like md17, md22, iso17

```
CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-PCQM4MV2_var0.4_var2_com_re_md17.yaml --layernorm-on-vec whitened --job-id frad_pretraining_force --num-epochs 8 
```


- The above script is for pre-training the model using RN noise. To switch to VRN noise, add the option ```--bat-noise true```.

- For the LBA task, we incorporate angular information into the molecular geometry embedding to better model the complexity of the input protein-ligand complex. Add the option ```--model equivariant-transformerf2d``` to apply the custom model for LBA.