### Bias sampling pretraining experiments


Options in config:


```
position_noise_scale: 0.06
sample_number: 100 # sampling number
violate: false # violate the rule or not
sdf_path: 'mol_iter_all.pickle' # rule file
```

run cmd:

`python scripts/train.py --conf examples/ET-PCQM4MV2_BIAS_1k_var0.04.yaml --layernorm-on-vec whitened --job-id pretraining_debug `