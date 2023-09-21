CUDA_LAUNCH_BLOCKING=5 \
python train/train.py \
-n ball_transparent \
-c NeRRF.conf \
-D data/blender/transparent/ball \
--gpu_id=0 \
--visual_path tet_visual \
--stage 1 \
--tet_scale 4.2 \
--sphere_radius 2.30 \
--resume \
--enable_refr \
--ior 1.1
# --use_sdf \
# --use_progressive_encoder
# --use_cone
