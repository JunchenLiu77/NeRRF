CUDA_LAUNCH_BLOCKING=5 \
python train/train.py \
-n cow_transparent \
-c NeRRF.conf \
-D data/blender/transparent/cow \
--gpu_id=0 \
--visual_path tet_visual \
--stage 1 \
--tet_scale 4.2 \
--sphere_radius 2.30 \
--ior 1.2 \
--resume \
--enable_refr
# --use_cone
# --use_sdf
# --use_progressive_encoder