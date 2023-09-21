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
from PIL import Image


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
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
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

net = net.to(device)


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

z_near = 0
z_far = 80

torch.random.manual_seed(args.seed)

with torch.no_grad():
    H, W = 512, 1024
    pose = torch.zeros((3,), dtype=torch.float32).to(device=device)
    center = pose.expand(H, W, -1)
    # phi, theta = torch.meshgrid(
    #     [
    #         torch.linspace(-0 * np.pi, *np.pi, H),
    #         torch.linspace(0 * np.pi, (2) * np.pi, W),
    #     ]
    # )
    phi, theta = torch.meshgrid(
        [torch.linspace(0.0, np.pi, H), torch.linspace(0 * np.pi, 2 * np.pi, W)]
    )
    viewdirs = torch.stack(
        [
            -torch.cos(theta) * torch.sin(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(phi),
        ],
        dim=-1,
    ).to(device=device)
    cam_nears = torch.tensor(0, device=device).view(1, 1, 1).expand(H, W, -1)
    cam_fars = torch.tensor(80, device=device).view(1, 1, 1).expand(H, W, -1)
    cam_rays = torch.cat((center, viewdirs, cam_nears, cam_fars), dim=-1)

    rgbs, depth = render_par(
        cam_rays.view(-1, cam_rays.shape[-1]).unsqueeze(0), want_weights=True
    )
    envmap = rgbs.reshape((H, W, 3))
    envmap = envmap.cpu().numpy()
    envmap = (envmap.clip(0, 1) * 255).astype("uint8")
    image = Image.fromarray(envmap)
    image.save("envmap1.png")

    # pyexr.write("envmap.exr", envmap)
