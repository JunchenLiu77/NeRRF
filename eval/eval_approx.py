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
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from lpips import LPIPS
import torch.nn.functional as F
from math import exp
from dataset.dataloader import Dataset

compare_lpips = LPIPS(net="squeeze").cpu()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    ).type(torch.FloatTensor)
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def calculate_psnr(img1, img2, mask):
    # img1 and img2 have range [0, 1]
    # img [H, W, 3], mask [H, W, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = ((((img1 - img2) ** 2) * mask).sum() / mask.sum()) / 3.0
    if mse == 0:
        return float("inf")
    return 20 * math.log10(1.0 / math.sqrt(mse))


def calculate_ssim(
    img1,
    img2,
    mask,
    window_size=11,
    channel=3,
    compute_average=False,
    sum_normalized=False,
):
    # img: [H, W, 3]
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0)
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0)

    window = create_window(window_size, channel)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )
    C1 = 0.01**2
    C2 = 0.03**2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    loss = ssim_map

    mask = torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(0)

    loss = torch.sum(loss * mask.float()) / (3 * mask.sum().float() + 1e-7)
    return loss


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

# if renderer.n_coarse < 64:
#     # Ensure decent sampling resolution
#     renderer.n_coarse = 64
# if args.coarse:
#     renderer.n_coarse = 64
#     renderer.n_fine = 128
#     renderer.using_fine = True

# renderer.n_coarse = 32
# renderer.n_fine = 0
# renderer.using_fine = False

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

total_psnr = 0.0
total_ssim = 0.0
total_lpips = 0.0
cnt = 0

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

        apply_mask = True
        if apply_mask:
            import cv2

            cv2.imwrite("mask.png", mask.squeeze(0).unsqueeze(-1).cpu().numpy() * 255)

            rgb_file_name = f"rgbs_{cnt}.png"
            cv2.imwrite(
                rgb_file_name,
                rgbs.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255,
            )
            # cv2.imwrite(
            #     "rgbs.png",
            #     (rgbs.squeeze(0) * mask).permute(1, 2, 0).cpu().numpy() * 255,
            # )
            # cv2.imwrite(
            #     "rgbs_gt.png",
            #     (rgbs_gt.squeeze(0) * mask).permute(1, 2, 0).cpu().numpy() * 255,
            # )

        #     psnr = calculate_psnr(
        #         rgbs_gt.squeeze(0).permute(1, 2, 0).cpu().numpy(),
        #         rgbs.squeeze(0).permute(1, 2, 0).cpu().numpy(),
        #         mask.permute(1, 2, 0).cpu().numpy(),
        #     )
        #     ssim = calculate_ssim(
        #         rgbs_gt.squeeze(0).permute(1, 2, 0).cpu().numpy(),
        #         rgbs.squeeze(0).permute(1, 2, 0).cpu().numpy(),
        #         mask.permute(1, 2, 0).cpu().numpy(),
        #     )
        #     lpips = 0
        # else:
        #     lpips = compare_lpips(rgbs, rgbs_gt).sum().item()
        #     ssim = compare_ssim(
        #         rgbs.numpy().squeeze(0),
        #         rgbs_gt.numpy().squeeze(0),
        #         win_size = 3,
        #         multichannel=True,
        #         data_range=1,
        #         channel_axis=0,
        #     )
        #     rgbs = rgbs.view(-1, 3).numpy()
        #     rgbs_gt = rgbs_gt.view(-1, 3).numpy()
        #     psnr = compare_psnr(rgbs, rgbs_gt, data_range=1)

        # total_ssim += ssim
        # total_psnr += psnr
        # total_lpips += lpips

        # import imageio

        # vis_u8 = (rgbs * 255).numpy().astype(np.uint8)
        # imageio.imwrite("test_eval.png", vis_u8)

        cnt += 1
#         print(
#             "curr psnr",
#             total_psnr / cnt,
#             "ssim",
#             total_ssim / cnt,
#             "lpips",
#             total_lpips / cnt,
#         )
# print(
#     "final psnr", total_psnr / cnt, "ssim", total_ssim / cnt, "lpips", total_lpips / cnt
# )
