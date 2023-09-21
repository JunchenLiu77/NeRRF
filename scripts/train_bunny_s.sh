CUDA_LAUNCH_BLOCKING=5 \
python train/train.py \
-n bunny_specular \
-c NeRRF.conf \
-D data/blender/specular/bunny \
--gpu_id=0 \
--visual_path tet_visual \
--stage 1 \
--tet_scale 3.8 \
--sphere_radius 2.43 \
--resume \
--enable_refl \
--use_sdf
# --use_cone \
# --use_progressive_encoder \
# --use_grid \