CUDA_LAUNCH_BLOCKING=5 \
python train/train.py \
-n eikonal \
-c NeRRF.conf \
-D data/eikonal \
--gpu_id=0 \
--visual_path tet_visual \
--stage 1 \
--tet_scale 0.7 \
--sphere_radius 2.40 \
--resume \
--enable_refr \
--ior 1.5
# --use_progressive_encoder
# --use_grid \
# --use_cone
