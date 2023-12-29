# Finetune guidance on ISO17 and MD22 (and ANI1, ANI1x, SPICE)

## env
这两个是确定额外需要的，应该没有漏别的，方便的话可以在MD22上试一下
```
pip install h5py
pip install ase
```

## data
- [ISO17 dataset](http://quantum-machine.org/datasets/): download and extract the file to `data/iso17`
- [MD22 dataset](http://www.sgdml.org/#datasets): download the `.xyz` format files and save to `data/MD22/md22_extendedxyz/`

## finetune scripts
- The *_bash.sh and *_angle.sh indicates `origin node` and `noisy node` finetuning method, respectively. 
  Note that in current implementation, `noisy node` does not use torsional noise, but only use random noise.
- Take finetune_md22_bash.sh as an example: simply run
```
bash finetune_md22_bash.sh 2e-4 cos 0
```
You can change the learning rate, the learning rate schedule and the dataset index by changing the command-line arguments.

