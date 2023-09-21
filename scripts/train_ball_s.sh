CUDA_LAUNCH_BLOCKING=5 \
python train/train.py \
-n ball_specular \
-c NeRRF.conf \
-D data/blender/specular/ball \
--gpu_id=0 \
--visual_path tet_visual \
--stage 1 \
--tet_scale 4.2 \
--sphere_radius 2.30 \
--resume \
--enable_refl \
--use_sdf
# --use_cone
# --use_progressive_encoder
# --use_grid \
