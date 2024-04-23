# import torch
# from torchvision.transforms import ToTensor
# from PIL import Image
# from model_definition import YourModel
##form chatgpt

import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import torch
import numpy as np
import util
from render import NeRFRenderer
from model import make_model

sys.path.append("dataloader")
import math
# from skimage.metrics import structural_similarity as compare_ssim
# from skimage.metrics import peak_signal_noise_ratio as compare_psnr
# from lpips import LPIPS
import torch.nn.functional as F
from math import exp
from dataset.dataloader import Dataset
import imageio

def extra_args(parser):
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Split of data to use train | val | test",
    )
    parser.add_argument(
        "--source",
        "-P",
        type=str,
        default=[0, 1, 2, 3],
        help="Source view(s) in image, in increasing order. -1 to use random 1 view.",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size") #4
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for selecting target views of each object",
    )
    parser.add_argument("--coarse", action="store_true", help="Coarse network as fine")
    parser.add_argument(
        "--enable_refr",
        action="store_true",
        default=False,
        help="Whether to enable refraction",
    )
    parser.add_argument(
        "--enable_refl",
        action="store_true",
        default=False,
        help="Whether to enable reflection",
    )
    parser.add_argument(
        "--use_cone",
        action="store_true",
        default=False,
        help="Whether to use cone sampling",
    )
    parser.add_argument(
        "--use_grid",
        action="store_true",
        default=False,
        help="Use grid param or MLP to predict sdf and deform",
    )
    parser.add_argument(
        "--use_sdf",
        action="store_true",
        default=False,
        help="Use sdf based intersection, aka sphere tracing or ray marching",
    )
    parser.add_argument(
        "--use_progressive_encoder",
        action="store_true",
        default=False,
        help="Whether to use progressive encoder",
    )
    parser.add_argument(
        "--stage",
        "-S",
        type=int,
        default=1,
        help="Stage of training, 1: optimize geometry, 2: optimize envmap, 3: optimize out direction",
    )
    parser.add_argument("--tet_scale", type=float, default=1.0, help="Scale of the tet")
    parser.add_argument(
        "--sphere_radius", type=float, default=1.0, help="Radius of the bounding sphere"
    )
    parser.add_argument("--ior", type=float, default=1.5, help="index of refraction")
    return parser


args, conf = util.args.parse_args(extra_args)
args.resume = True

device = util.get_cuda(args.gpu_id[0])
net = make_model(conf["model"]).to(device=device)
# net.load_weights(args)
default_net_state_path = "%s/%s/%d/net" % (
    args.checkpoints_path,
    args.name,
    args.stage,
)
model_path = "%s/%s/pixel_nerf_latest" % (
    args.checkpoints_path,
    args.name,
)
if hasattr(net, "load_weights") and os.path.exists(model_path):
    net.load_state_dict(torch.load(model_path, map_location=device))
if os.path.exists(default_net_state_path):
    net.load_state_dict(torch.load(default_net_state_path, map_location=device))

dset = Dataset(args.datadir, stage="test")

print(args.datadir)



data_loader = torch.utils.data.DataLoader(
    dset, batch_size=16, shuffle=False, pin_memory=False #1
)

renderer = NeRFRenderer.from_conf(
    conf["renderer"],
    eval_batch_size=args.ray_batch_size,
    enable_refr=args.enable_refr,
    enable_refl=args.enable_refl,
    stage=args.stage,
    tet_scale=args.tet_scale,
    sphere_radius=args.sphere_radius,
    ior=args.ior,
    use_cone=args.use_cone,
    use_grid=args.use_grid,
    use_sdf=args.use_sdf,
    use_progressive_encoder=args.use_progressive_encoder,
).to(device=device)



render_par = renderer.bind_parallel(net, args.gpu_id, simple_output=True).eval()

z_near = dset.z_near
z_far = dset.z_far

# load renderer paramters
renderer_state_path = "%s/%s/_renderer" % (
    args.checkpoints_path,
    args.name,
)
if os.path.exists(renderer_state_path):
    renderer.load_state_dict(
        torch.load(renderer_state_path, map_location=device), False
    )

# load mesh rendered in stage 1
if args.stage == 2 or args.stage == 3:
    if not args.use_sdf:
        renderer.init_tet(
            # mesh_path="dataloader/learned_geo/" + args.name.split("_")[0] + ".obj"
            mesh_path="data/learned_geo/" + args.name.split("_")[0] + ".obj"
        )
elif args.stage != 1:
    raise NotImplementedError()

torch.random.manual_seed(args.seed)

source = torch.tensor(args.source, dtype=torch.long)
NS = len(source)
random_source = NS == 1 and source[0] == -1


with torch.no_grad():
    for data in data_loader:
        image = data["images"][0].to(device=device)  # (3, H, W)
        pose = data["poses"][0].to(device=device)  # (4, 4)
        focal = data["focal"][0].to(device=device)  # [2]
        mask = data["mask"][0].to(device=device).cpu().float()  # todo
        _, H, W = image.shape  # (3, H, W)
        cam_rays = util.gen_rays(pose, W, H, focal, z_near, z_far)  # (H, W, 8)
        rgbs_gt = image * 0.5 + 0.5  # (3, H, W)

        rgbs, depth = render_par(
            cam_rays.view(-1, cam_rays.shape[-1]).unsqueeze(0), want_weights=True
        )
        rgbs = rgbs.permute(0, 2, 1).view(-1, 3, H, W).contiguous().cpu()
        rgbs_gt = rgbs_gt.unsqueeze(0).cpu()



        vis_u8 = (rgbs * 255).astype(np.uint8)
        imageio.imwrite("test_eval.png", vis_u8)




