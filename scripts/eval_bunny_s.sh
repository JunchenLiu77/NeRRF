CUDA_LAUNCH_BLOCKING=5 \
python eval/eval_approx.py  \
-n bunny_specular \
-c NeRRF.conf \
-D data/blender/specular/bunny \
--gpu_id=0 \
--stage 2 \
--tet_scale 3.8 \
--sphere_radius 2.43 \
--enable_refl \
--use_sdf
# --use_cone
# --use_progressive_encoder