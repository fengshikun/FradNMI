### Pretraining

Rotation Noise (Model for the QM9)

```
python -u scripts/train.py --conf examples/ET-PCQM4MV2_dih_var0.04_var2_com_re.yaml --layernorm-on-vec whitened --job-id frad_pretraining --num-epochs 8 
```

Vibration and Rotation (VR) Noise (Model for Atomic Forces Tasks like md17, md22, iso17)

```
python -u scripts/train.py --conf examples/ET-PCQM4MV2_var0.4_var2_com_re_md17.yaml --layernorm-on-vec whitened --job-id frad_pretraining_force --num-epochs 8 --bat-noise true
```

We have provided the pretrained models at the following links:

Rotation Noise: [https://drive.google.com/file/d/1O6f6FzYogBS2Mp4XsdAAEN4arLtLH38G/view?usp=sharing]

Vibration and Rotation (VR) Noise: [https://drive.google.com/file/d/12ZNPNnugD3ZxQPTMUKVNkEl3-fLjHBgJ/view?usp=sharing]



### Finetuning

#### Finetune on QM9

Below is the script for fine-tuning the QM9 task. Ensure to replace `pretrain_model_path` with the actual model path. In this script, the subtask is set to 'homo', but it can be replaced with other subtasks as well.

```bash
python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_homo --dataset-arg homo  --denoising-weight 0.1 --dataset-root $datapath --pretrained-model $pretrain_model_path
```


#### Finetune on MD17
Below is the script for fine-tuning the MD17 task. Replace pretrain_model_path with the actual model path. In this script, the subtask is set to 'aspirin', but it can be replaced with other subtasks such as {'benzene', 'ethanol', 'malonaldehyde', 'naphthalene', 'salicylic_acid', 'toluene', 'uracil'}.


```bash
python -u scripts/train.py --conf examples/ET-MD17_FT-angle_9500.yaml  --job-id frad_aspirin --dataset-arg aspirin --pretrained-model $pretrain_model_path --dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss
```


#### Finetuning the MD22


#### Finetuning the ISO17


