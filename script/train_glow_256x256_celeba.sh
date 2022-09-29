# CUDA_VISIBLE_DEVICES=4,5,6,7 python src/train.py --config config/train/glow_256x256_celeba.yml
CUDA_VISIBLE_DEVICES=3 python src/train.py --config config/resume/glow_256x256_celeba.yml --resume
