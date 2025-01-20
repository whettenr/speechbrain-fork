# BEST-RQ pretraining with SpeechBrain

This folder contains the scripts to train a BEST-RQ model using Libri-light. It can be adapted to any dataset as long as you provide the csv or json files. No other adaptation will be required apart from controlling the sequence length and Dynamic Batching arguments to avoid out of memory issues.

More information on the architecture can be found in [the original paper](https://arxiv.org/pdf/2202.01855.).

# Go !
Simply type:
```shell
python train.py hparams/BEST-RQ.yaml --find_unused_parameters
```

Do not forget to replace the `!PLACEHOLDER` variables in the yaml corresponding to your local configuration.

train=/lustre/fswork/projects/rech/nkp/uaj64gk/bestrq_sb/speechbrain/recipes/Libri-light/self-supervised-learning/BEST-RQ/train.py
hyparams=/lustre/fswork/projects/rech/nkp/uaj64gk/bestrq_sb/speechbrain/recipes/Libri-light/self-supervised-learning/BEST-RQ/hparams/BEST-RQ.yaml
data_folder=/gpfsdswork/dataset/Libri-light
train_splits: ["small"]

```bash
module load pytorch-gpu/py3/2.1.1
conda activate libri-light
cd /lustre/fswork/projects/rech/nkp/uaj64gk/bestrq_sb/librilight_prep/

train=/lustre/fswork/projects/rech/nkp/uaj64gk/bestrq_sb/speechbrain/recipes/Libri-light/self-supervised-learning/BEST-RQ/train.py
hyparams=/lustre/fswork/projects/rech/nkp/uaj64gk/bestrq_sb/speechbrain/recipes/Libri-light/self-supervised-learning/BEST-RQ/hparams/BEST-RQ.yaml
data_folder=/gpfsdswork/dataset/Libri-light

python $train $hyparams --find_unused_parameters --data_folder $data_folder --seconds_per_batch 100 --train_num_buckets 50
```