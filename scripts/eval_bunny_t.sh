CUDA_LAUNCH_BLOCKING=5 \
python eval/eval_approx.py  \
-n bunny_transparent \
-c NeRRF.conf \
-D data/blender/transparent/bunny \
--gpu_id=0 \
--stage 2 \
--tet_scale 3.8 \
--sphere_radius 2.43 \
--enable_refr \
--ior 1.1
# --use_cone
# --use_sdf
# --use_progressive_encoder