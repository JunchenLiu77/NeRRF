CUDA_LAUNCH_BLOCKING=5 \
python train/train.py \
-n cow_specular \
-c NeRRF.conf \
-D data/blender/specular/cow \
--gpu_id=0 \
--visual_path tet_visual \
--stage 2 \
--tet_scale 4.2 \
--sphere_radius 2.30 \
--enable_refl \
--use_sdf \
--resume
# --use_cone \
# --use_progressive_encoder \
# --use_grid \