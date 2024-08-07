#!/usr/bin/bash
echo "Starting Job"
export PATH="/esat/biomeddata/kkontras/r0786880/anaconda3/bin:$PATH"
echo "Current PATH: $PATH"
source /esat/biomeddata/kkontras/r0786880/anaconda3/etc/profile.d/conda.sh
conda activate /esat/biomeddata/kkontras/r0786880/anaconda3/envs/junyanz_env_new

which python
python -V
echo "Current PATH: $PATH"
cd /esat/biomeddata/kkontras/r0786880/models/pytorch-CycleGAN-and-pix2pix
export PYTHONPATH='/esat/biomeddata/kkontras/r0786880/models/pytorch-CycleGAN-and-pix2pix/:$PYTHONPATH'
echo $PWD
num_gpus=$(nvidia-smi --list-gpus | wc -l)
if [ $num_gpus -eq 1 ]; then
    echo "We have 1 GPU"
else
    echo "We have $num_gpus GPUs"
fi
echo "Current PATH: $PATH"
python /esat/biomeddata/kkontras/r0786880/models/pytorch-CycleGAN-and-pix2pix/whole_training_loop.py --dataroot ./train_images/trainA --load_size 256 --num_threads 0 --use_wandb --display_id 0 --name default --netD n_layers --n_layers_D 3 --lr 0.0002 --lambda_identity 0.5 --batch_size 1 --max_patients 25 --save_latest_freq 5000 --print_freq 2500 --display_freq 2500 --save_by_iter --netG resnet_9blocks
echo "Job finished"