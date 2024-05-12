import sys
import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
sys.path.append("dataloader")


import trainlib
from model import make_model, loss
from render import NeRFRenderer
import util
import numpy as np
import torch
import torch.nn.functional as F
from dotmap import DotMap
from dataset.dataloader import Dataset
import cv2
import pyrender
import trimesh


import random
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False

def extra_args(parser):
    parser.add_argument(
        "--batch_size", "-B", type=int, default=16, help="Object batch size ('SB')"
    )
    parser.add_argument(
        "--freeze_enc",
        action="store_true",
        default=None,
        help="Freeze encoder weights and only train MLP",
    )
    parser.add_argument(
        "--no_bbox_step",
        type=int,
        default=100000,
        help="Step to stop using bbox sampling",
    )
    parser.add_argument(
        "--fixed_test",
        action="store_true",
        default=None,
        help="Freeze encoder weights and only train MLP",
    )
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
        help="Stage of training, 1: optimize geometry, 2: optimize envmap",
    )
    parser.add_argument("--tet_scale", type=float, default=1.0, help="Scale of the tet")
    parser.add_argument(
        "--sphere_radius", type=float, default=1.0, help="Radius of the bounding sphere"
    )
    parser.add_argument("--ior", type=float, default=1.5, help="index of refraction")
    return parser


# parse args
args, conf = util.args.parse_args(extra_args, training=True, default_ray_batch_size=256)
device = util.get_cuda(args.gpu_id[0])

if args.stage != 1 and args.stage != 2:
    raise NotImplementedError()

# partition dataset
train_dset = Dataset(args.datadir, stage="train")
test_dset = Dataset(args.datadir, stage="test")

# network
net = make_model(conf["model"]).to(device=device)

# renderer
renderer = NeRFRenderer.from_conf(
    conf["renderer"],
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
    args=args,
).to(device=device)

# parallize
render_par = renderer.bind_parallel(net, args.gpu_id).eval()


class RRFTrainer(trainlib.Trainer):
    def __init__(self):
        super().__init__(net, train_dset, test_dset, args, conf["train"], device=device)
        self.renderer_state_path = "%s/%s/_renderer" % (
            self.args.checkpoints_path,
            self.args.name,
        )
        self.lambda_coarse = conf.get_float("loss.lambda_coarse")
        self.lambda_fine = conf.get_float("loss.lambda_fine", 1.0)
        print(
            "lambda coarse {} and fine {}".format(self.lambda_coarse, self.lambda_fine)
        )
        self.rgb_coarse_crit = loss.get_rgb_loss(conf["loss.rgb"], True)
        fine_loss_conf = conf["loss.rgb"]
        if "rgb_fine" in conf["loss"]:
            print("using fine loss")
            fine_loss_conf = conf["loss.rgb_fine"]
        self.rgb_fine_crit = loss.get_rgb_loss(fine_loss_conf, False)

        if args.stage == 1:
            net.eval()
            self.optim = torch.optim.Adam(
                [
                    {
                        "params": [
                            p
                            for n, p in renderer.named_parameters()
                            if ("sdf" in n) or ("deform" in n) and p.requires_grad
                        ],
                        "lr": 1.e-2, #0.001 1.e-4
                    },
                ],
            )
        elif args.stage == 2:
            self.optim = torch.optim.Adam(
                [
                    {
                        "params": [p for n, p in net.named_parameters()],
                        "lr": 0.01,  # 0.01 for ngp, 5e-4 for nerf
                    },
                ]
            )
        else:
            raise NotImplementedError()

        # load renderer paramters
        # if os.path.exists(self.renderer_state_path):
        #     renderer.load_state_dict(
        #         torch.load(self.renderer_state_path, map_location=device), False
        #     )

        # load mesh rendered in stage 1
        if args.stage == 2:
            if not args.use_sdf:
                renderer.init_tet(
                    mesh_path="data/learned_geo/" + args.name.split("_")[0] + ".obj"
                )
        elif args.stage != 1:
            raise NotImplementedError()

        self.z_near = train_dset.z_near
        self.z_far = train_dset.z_far

        self.use_bbox = args.no_bbox_step > 0

    def post_batch(self, epoch, batch):
        renderer.sched_step(args.batch_size)

    def extra_save_state(self, global_step):
        torch.save(renderer.state_dict(), self.renderer_state_path)
        mesh = renderer.export_mesh(global_step=global_step)
        geo_path = "data/learned_geo/"
        if not os.path.exists(geo_path):
            os.makedirs(geo_path)
        mesh.export(geo_path + args.name + str(global_step) + ".obj")

    def calc_losses(self, data, is_train=True, global_step=0):
        stage = args.stage
        image = data["images"][0].to(device=device)  # (3, H, W)
        pose = data["poses"][0].to(device=device)  # (4, 4)
        focal = data["focal"][0].to(device=device)  # (2)
        mvp = data["mvp"][0].to(device=device)  # (4, 4)
        mask = data["mask"][0].to(device=device).float()  # (H, W)
        _, H, W = image.shape  # (3, H, W)

        cam_rays = util.gen_rays(
            pose, W, H, focal, self.z_near, self.z_far
        )  # (H, W, 8)
        rgbs_gt = image * 0.5 + 0.5  # (3, H, W)
        rgbs_gt = rgbs_gt.permute(1, 2, 0).contiguous().reshape(-1, 3)  # (H * W, 3)

        apply_mask = False
        if apply_mask:
            mask_flatten = mask[:, :H, :].reshape(-1)
            hit_idx = torch.where(mask_flatten > 0.5)[0]
            perm = torch.randperm(hit_idx.numel())
            selected_indices = perm[: args.ray_batch_size]
            pix_inds = hit_idx[selected_indices]
        else:
            pix_inds = torch.randint(0, H * W, (args.ray_batch_size,))
        samp_rgbs_gt = rgbs_gt[pix_inds].unsqueeze(0)  # (1, ray_batch_size, 3)
        samp_rays = (
            cam_rays.view(-1, cam_rays.shape[-1])[pix_inds]
            .to(device=device)
            .unsqueeze(0)
        )  # (1, ray_batch_size, 8)
        loss_dict = {}

        if stage == 1:
            mask_loss = 0.0
            ek_loss = 0.0
            mask_, ek_loss = renderer.render_mask(
                samp_rays, mvp, h=H, w=W, global_step=global_step
            )
            # p = 0.8
            # mask1 = mask * p + (torch.ones_like(mask) - mask) * (1 - p)
            # mask2 = mask_ * p + (torch.ones_like(mask_) - mask_) * (1 - p)

            mask_gt = mask
            mask_pred = mask_
            mask_loss_mse = MSE_loss(mask_gt, mask_pred)
            
            mask_loss, iou = IoU_loss(mask_gt, mask_pred)

            loss_dict["mask"] = mask_loss.item()
            loss_dict["eikonal"] = ek_loss
            
            loss = mask_loss_mse + ek_loss
            # loss = mask_loss + ek_loss

            if global_step == 0: 
                monitor = "monitor"
                if os.path.exists(monitor):
                    for file_name in os.listdir(monitor):
                        file_path = os.path.join(monitor, file_name)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    os.rmdir(monitor)
                os.mkdir(monitor)

            if global_step % 200 == 100: 
                print(f"{global_step}, IoU: {iou}, IoU_Loss: {mask_loss}, MSE Loss: {mask_loss_mse}")
                # print(f"num 0={torch.where(mask_pred == 0)[0].numel()}, num 1={torch.where(mask_pred == 1)[0].numel()}, num other={torch.where((mask_pred != 0) & (mask_pred != 1))[0].numel()}")
                # print(f"num <0={torch.where(mask_pred < 0)[0].numel()}, num >0={torch.where(mask_pred > 0)[0].numel()}")
                mask_file_name = f"monitor/mask@step{global_step}.png"
                fixed_rgbs = mask_pred.squeeze(0).cpu().detach().numpy()
                cv2.imwrite(mask_file_name, fixed_rgbs * 255)
                maskgt_file_name = f"monitor/mask@step{global_step}GT.png"
                fixed_rgbs = mask_gt.squeeze(0).cpu().detach().numpy()
                cv2.imwrite(maskgt_file_name, fixed_rgbs * 255)
            if 0:
                # https://pyrender.readthedocs.io/en/latest/examples/quickstart.html#minimal-example-for-offscreen-rendering
                # If you’re using a headless server, make sure that you followed the guide for installing OSMesa. See Getting Pyrender Working with OSMesa.
                scene = pyrender.Scene()
                fuze_trimesh = trimesh.load('data/learned_geo/grasp_catglass_transparent3500.obj')
                # fuze_trimesh = trimesh.load('data/init/sphere.obj')
                mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
                scene.add(mesh)
                pyrender_camera = pyrender.IntrinsicsCamera(focal[0], focal[1], 240, 135, zfar=800.0)
                scene.add(pyrender_camera, pose=pose.cpu())
                light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,innerConeAngle=np.pi/16.0,outerConeAngle=np.pi/6.0)
                scene.add(light, pose=pose.cpu())
                r = pyrender.OffscreenRenderer(400, 400)
                color, depth = r.render(scene)


            if is_train:
                loss.backward()
            loss_dict["t"] = loss.item()
            return loss_dict
        elif stage == 2:
            render_dict = DotMap(render_par(samp_rays, want_weights=True))
            coarse = render_dict.coarse
            fine = render_dict.fine
            using_fine = len(fine) > 0
            rgb_loss = self.rgb_coarse_crit(coarse.rgb, samp_rgbs_gt)
            loss_dict["rc"] = rgb_loss.item() * self.lambda_coarse
            if using_fine:
                if apply_mask:
                    raise NotImplementedError()
                fine_loss = self.rgb_fine_crit(fine.rgb, samp_rgbs_gt)
                rgb_loss = rgb_loss * self.lambda_coarse + fine_loss * self.lambda_fine
                loss_dict["rf"] = fine_loss.item() * self.lambda_fine
            loss = rgb_loss

            if is_train:
                loss.backward()

            loss_dict["t"] = loss.item()
            return loss_dict

    def train_step(self, data, global_step):
        return self.calc_losses(data, is_train=True, global_step=global_step)

    def eval_step(self, data, global_step):
        renderer.eval()
        losses = self.calc_losses(data, is_train=False, global_step=global_step)
        renderer.train()
        return losses

    def vis_step(self, data, global_step, idx=None):
        image = data["images"][0].to(device=device)  # (3, H, W)
        pose = data["poses"][0].to(device=device)  # (4, 4)
        focal = data["focal"][0].to(device=device)  # (2)
        _, H, W = image.shape  # (3, H, W)

        cam_rays = util.gen_rays(
            pose, W, H, focal, self.z_near, self.z_far
        )  # (H, W, 8)
        rgbs_gt = image * 0.5 + 0.5  # (3, H, W)
        renderer.eval()
        gt = rgbs_gt.permute(1, 2, 0).cpu().numpy().reshape(H, W, 3)
        with torch.no_grad():
            test_rays = cam_rays  # (H, W, 8)
            test_rays = test_rays.reshape(1, H * W, -1)

            num_split = 1  # to avoid OOM
            test_rays_list = torch.chunk(test_rays, num_split, dim=1)
            alpha_coarse_np_list, rgb_coarse_np_list = [], []
            alpha_fine_np_list, rgb_fine_np_list = [], []
            for rays in test_rays_list:
                render_dict = DotMap(render_par(rays, want_weights=True))
                coarse = render_dict.coarse
                fine = render_dict.fine
                using_fine = len(fine) > 0
                alpha_coarse_np = coarse.weights[0].sum(dim=-1)
                rgb_coarse_np = coarse.rgb[0]
                alpha_coarse_np_list.append(alpha_coarse_np.cpu().numpy())
                rgb_coarse_np_list.append(rgb_coarse_np.cpu().numpy())

                if using_fine:
                    alpha_fine_np = fine.weights[0].sum(dim=1)
                    rgb_fine_np = fine.rgb[0]
                    alpha_fine_np_list.append(alpha_fine_np.cpu().numpy())
                    rgb_fine_np_list.append(rgb_fine_np.cpu().numpy())

            alpha_coarse_np = np.concatenate(alpha_coarse_np_list, axis=0).reshape(H, W)
            rgb_coarse_np = np.concatenate(rgb_coarse_np_list, axis=0).reshape(H, W, 3)

            if len(alpha_fine_np_list) > 0:
                alpha_fine_np = np.concatenate(alpha_fine_np_list, axis=0).reshape(H, W)
                rgb_fine_np = np.concatenate(rgb_fine_np_list, axis=0).reshape(H, W, 3)

        print("c rgb min {} max {}".format(rgb_coarse_np.min(), rgb_coarse_np.max()))
        print(
            "c alpha min {}, max {}".format(
                alpha_coarse_np.min(), alpha_coarse_np.max()
            )
        )

        vis_list = [
            gt,
            rgb_coarse_np,
            # alpha_coarse_cmap,
        ]

        vis_coarse = np.hstack(vis_list)
        vis = vis_coarse

        if using_fine:
            print("f rgb min {} max {}".format(rgb_fine_np.min(), rgb_fine_np.max()))
            print(
                "f alpha min {}, max {}".format(
                    alpha_fine_np.min(), alpha_fine_np.max()
                )
            )
            vis_list = [
                gt,
                rgb_fine_np,
                # alpha_fine_cmap,
            ]

            vis_fine = np.hstack(vis_list)
            vis = np.vstack((vis_coarse, vis_fine))
            rgb_psnr = rgb_fine_np
        else:
            rgb_psnr = rgb_coarse_np

        psnr = util.psnr(rgb_psnr, gt)
        vals = {"psnr": psnr}
        print("psnr", psnr)

        # set the renderer network back to train mode
        renderer.train()
        return vis, vals

    def test_step(self, data, global_step, idx=None):
        return self.eval_step(data, global_step)


def MSE_loss(gt, pred):
    """
    Args:
        gt (Tensor): shape (1, H, W)
        pred (Tensor): shape (1, H, W)
    """
    alpha = 1
    TP = (gt.bool() & pred.bool())
    FP = ((~gt.bool()) & pred.bool())
    FN = (gt.bool() & (~pred.bool()))
    TN = (~gt.bool() & (~pred.bool()))

    TP_ = TP * (pred - gt * alpha) ** 2
    FP_ = FP * (pred - gt * alpha) ** 2
    FN_ = FN * (pred - gt * alpha) ** 2
    TN_ = TN * (pred - gt * alpha) ** 2
    
    N = gt.numel()
    N_gt = gt.sum()
    N_TP = TP.sum().detach()
    N_FP = FP.sum().detach()
    N_FN = FN.sum().detach()
    N_TN = TN.sum().detach()
    a1 = 1 - N_TN / N #  / N_gt
    a2 = N_TN / N #  / N

    union = (a2 * TP_ + a1 * FP_).sum()

    loss = union # / N
    # loss = 1 - gt.sum() / union
    
    # loss = torch.nn.functional.mse_loss(gt * alpha, pred)

    return loss


def IoU_loss(gt, pred):
    """
    Args:
        gt (Tensor): shape (1, H, W)
        pred (Tensor): shape (1, H, W)
    """
    TP = (gt.bool() & pred.bool())
    FP = ((~gt.bool()) & pred.bool())
    FN = (gt.bool() & (~pred.bool()))

    TP_ = TP * pred # 
    FP_ = FP * pred # 
    FN_ = FN * gt
    # alpha = 2
    # TP_ = TP * (pred - alpha) ** 2
    # FP_ = FP * pred ** 2
    # FN_ = FN * alpha ** 2
    
    union = (TP_ + FP_ + FN_).sum()
    
    iou = TP_.sum() / union
    iou_loss = 1 - TP_.sum() / union

    return iou_loss, iou


trainer = RRFTrainer()
trainer.start()
