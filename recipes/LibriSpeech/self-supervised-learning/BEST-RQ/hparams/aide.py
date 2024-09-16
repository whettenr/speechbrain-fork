import torch;from speechbrain.lobes.models.transformer.Conformer import ConformerEncoder;

x = torch.rand((8, 60, 512)); 
pos_emb = torch.rand((1, 2*60-1, 512))
net = ConformerEncoder(12, 512, 512, 8)
output, _, = net(x, pos_embs=pos_emb)
output.shape

cd /users/rwhetten/attention_alt/speechbrain/recipes/LibriSpeech/self-supervised-learning/BEST-RQ
conda activate aa

# 1 GPU
python train_grow_in.py hparams/BEST-RQ_growth.yaml \
    --data_folder /corpus/LibriSpeech --grad_accumulation_factor 2 --skip_prep true \
    --optimizer_step_limit 100 --growth_stage 1 --mask_prob 0.02

python train_grow_in.py hparams/BEST-RQ_growth.yaml \
    --data_folder /corpus/LibriSpeech --grad_accumulation_factor 2 --skip_prep true \
    --optimizer_step_limit 200 --growth_stage 2 ----mask_prob 0.04

python train_grow_in.py hparams/BEST-RQ_growth.yaml \
    --data_folder /corpus/LibriSpeech --grad_accumulation_factor 2 --skip_prep true \
    --optimizer_step_limit 300 --growth_stage 3 ----mask_prob 0.06

# distributed

srun -c 8 -p gpu --gpus-per-node=2  --constraint='GPURAM_Min_24GB&GPURAM_Max_32GB' --mem=32G --time=02:00:00 --pty /bin/bash

python -m torch.distributed.run --nproc_per_node=2 --rdzv_backend c10d --rdzv-endpoint=localhost:0 train_grow_in.py hparams/BEST-RQ_growth.yaml --find_unused_parameters \
    --data_folder /corpus/LibriSpeech --grad_accumulation_factor 2 --skip_prep true \
    --optimizer_step_limit 100 --mask_prob 0.02
    
python -m torch.distributed.run --nproc_per_node=2 --rdzv_backend c10d --rdzv-endpoint=localhost:0 train_grow_in.py hparams/BEST-RQ_growth.yaml --find_unused_parameters \
    --data_folder /corpus/LibriSpeech --grad_accumulation_factor 2 --skip_prep true \
    --optimizer_step_limit 200 --growth_stage 2 --mask_prob 0.04

python -m torch.distributed.run --nproc_per_node=2 --rdzv_backend c10d --rdzv-endpoint=localhost:0 train_grow_in.py hparams/BEST-RQ_growth.yaml --find_unused_parameters \
    --data_folder /corpus/LibriSpeech --grad_accumulation_factor 2 --skip_prep true \
    --optimizer_step_limit 300 --growth_stage 3 --mask_prob 0.06

python -m torch.distributed.run --nproc_per_node=2 --rdzv_backend c10d --rdzv-endpoint=localhost:0 train_grow_in.py hparams/BEST-RQ_growth.yaml --find_unused_parameters \
    --data_folder /corpus/LibriSpeech --grad_accumulation_factor 2 --skip_prep true \
    --optimizer_step_limit 300 --growth_stage 4 --mask_prob 0.08

python -m torch.distributed.run --nproc_per_node=2 --rdzv_backend c10d --rdzv-endpoint=localhost:0 train_grow_in.py hparams/BEST-RQ_growth.yaml --find_unused_parameters \
    --data_folder /corpus/LibriSpeech --grad_accumulation_factor 2 --skip_prep true \
    --optimizer_step_limit 300 --growth_stage 4 ----mask_prob 0.08
