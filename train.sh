# train nerf
python main_train.py --data_cfg configs/data/mpiis/DSC_7157.yml --exp_cfg configs/exp/stage_0_nerf.yml

# train hybrid
python main_train.py --data_cfg configs/data/mpiis/DSC_7157.yml --exp_cfg configs/exp/stage_1_hybrid_perceptual.yml --clean
## Note: this one uses perceptual loss rather than mrf loss as described in the paper.
## The reason is: mrf loss works for v100. But for a100, the loss will be NAN. 
## if you want to use mrf loss, make sure you are using v100, then run: 
## python main_train.py --data_cfg configs/data/mpiis/DSC_7157.yml --exp_cfg configs/exp/stage_1_hybrid.yml --clean

#--- training male-3-casual example 
# python main_train.py --data_cfg configs/data/snapshot/male-3-casual.yml --exp_cfg configs/exp/stage_0_nerf.yml
