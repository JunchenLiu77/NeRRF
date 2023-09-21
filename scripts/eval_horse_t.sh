CUDA_LAUNCH_BLOCKING=5 \
python eval/eval_approx.py \
-n horse_transparent \
-c NeRRF.conf \
-D data/blender/transparent/horse \
--gpu_id=0 \
--stage 2 \
--tet_scale 3.8 \
--sphere_radius 2.40 \
--enable_refr \
--ior 1.5
# --use_cone
# --use_progressive_encoder