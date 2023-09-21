"""
Main model implementation
"""
import torch
from .code import PositionalEncoding
from .model_util import make_mlp
import torch.autograd.profiler as profiler
import os
import os.path as osp
import warnings


class PixelNeRFNet(torch.nn.Module):
    def __init__(self, conf):
        """
        :param conf PyHocon config subtree 'model'
        """
        super().__init__()
        self.viewdir_only = True
        self.use_viewdirs = True

        self.pos_encoder = PositionalEncoding.from_conf(conf["code"], d_in=3)
        self.dir_encoder = PositionalEncoding.from_conf(conf["code"], d_in=3)
        self.d_in = self.dir_encoder.d_out + (
            self.pos_encoder.d_out if not self.viewdir_only else 0
        )
        self.d_out = 4
        self.d_latent = 0
        self.mlp_coarse = make_mlp(
            conf["mlp_coarse"], self.d_in, self.d_latent, d_out=self.d_out
        )
        self.mlp_fine = make_mlp(
            conf["mlp_fine"],
            self.d_in,
            self.d_latent,
            d_out=self.d_out,
            allow_empty=True,
        )
        # Note: this is world -> camera, and bottom row is omitted
        self.register_buffer("poses", torch.empty(1, 3, 4), persistent=False)

    def forward(
        self,
        xyz=None,
        coarse=True,
        viewdirs=None,
    ):
        with profiler.record_function("model_inference"):
            SB, B, _ = viewdirs.shape

            if self.use_viewdirs:
                z_feature = self.dir_encoder(viewdirs.reshape(-1, 3))
            if not self.viewdir_only:
                pos_encode = self.pos_encoder(xyz.reshape(-1, 3))
                z_feature = torch.cat((z_feature, pos_encode), dim=1)
            mlp_input = z_feature

            # Run main NeRF network
            if coarse or self.mlp_fine is None:
                mlp_output = self.mlp_coarse(
                    mlp_input,
                    combine_inner_dims=(1, B),
                    combine_index=None,
                    dim_size=None,
                )
            else:
                mlp_output = self.mlp_fine(
                    mlp_input,
                    combine_inner_dims=(1, B),
                    combine_index=None,
                    dim_size=None,
                )

            # Interpret the output
            mlp_output = mlp_output.reshape(-1, B, self.d_out)

            rgb = mlp_output[..., :3]
            sigma = mlp_output[..., 3:4]

            output_list = [torch.sigmoid(rgb), torch.relu(sigma)]
            output = torch.cat(output_list, dim=-1)
            output = output.reshape(SB, B, -1)
        return output

    def load_weights(self, args, opt_init=False, strict=True, device=None):
        """
        Helper for loading weights according to argparse arguments.
        Your can put a checkpoint at checkpoints/<exp>/pixel_nerf_init to use as initialization.
        :param opt_init if true, loads from init checkpoint instead of usual even when resuming
        """
        # TODO: make backups
        if opt_init and not args.resume:
            return
        ckpt_name = (
            "pixel_nerf_init" if opt_init or not args.resume else "pixel_nerf_latest"
        )
        model_path = "%s/%s/%s" % (args.checkpoints_path, args.name, ckpt_name)

        if device is None:
            device = self.poses.device

        if os.path.exists(model_path):
            print("Load", model_path)
            self.load_state_dict(
                torch.load(model_path, map_location=device), strict=False
            )
        elif not opt_init:
            warnings.warn(
                (
                    "WARNING: {} does not exist, not loaded!! Model will be re-initialized.\n"
                    + "If you are trying to load a pretrained model, STOP since it's not in the right place. "
                    + "If training, unless you are startin a new experiment, please remember to pass --resume."
                ).format(model_path)
            )
        return self

    def save_weights(self, args, opt_init=False):
        """
        Helper for saving weights according to argparse arguments
        :param opt_init if true, saves from init checkpoint instead of usual
        """
        from shutil import copyfile

        ckpt_name = "pixel_nerf_init" if opt_init else "pixel_nerf_latest"
        backup_name = "pixel_nerf_init_backup" if opt_init else "pixel_nerf_backup"

        ckpt_path = osp.join(args.checkpoints_path, args.name, ckpt_name)
        ckpt_backup_path = osp.join(args.checkpoints_path, args.name, backup_name)

        if osp.exists(ckpt_path):
            copyfile(ckpt_path, ckpt_backup_path)
        torch.save(self.state_dict(), ckpt_path)
        return self
