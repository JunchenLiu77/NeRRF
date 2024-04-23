"""
NeRF differentiable renderer.
References:
https://github.com/bmild/nerf
https://github.com/kwea123/nerf_pl
"""
from builtins import max
from scipy.stats import norm
import torch
import torch.autograd.profiler as profiler
from dotmap import DotMap
import numpy as np
import trimesh
import mesh_raycast
import math
import os
import cubvh
import tqdm
from util.meshutils import clean_mesh
from util.meshutils import decimate_mesh
from util.util import copy_index
import nvdiffrast.torch as dr
from render.render import mesh
from render.render import render
from render.render import material
from render.render import mlptexture
from render.render import light
from PIL import Image
from render.geometry.dmtet import DMTet
from model.mlp import MLP
from model.encoding import get_encoder


class _RenderWrapper(torch.nn.Module):
    def __init__(self, net, renderer, simple_output):
        super().__init__()
        self.net = net
        self.renderer = renderer
        self.simple_output = simple_output

    def forward(self, rays, want_weights=False):
        if rays.shape[0] == 0:
            return (
                torch.zeros(0, 3, device=rays.device),
                torch.zeros(0, device=rays.device),
            )
        outputs = self.renderer(
            self.net,
            rays,
            want_weights=want_weights and not self.simple_output,
        )
        if self.simple_output:
            if self.renderer.using_fine:
                rgb = outputs.fine.rgb
                depth = outputs.fine.depth
            else:
                rgb = outputs.coarse.rgb
                depth = outputs.coarse.depth
            return rgb, depth
        else:
            # Make DotMap to dict to support DataParallel
            return outputs.toDict()


class NeRFRenderer(torch.nn.Module):
    """
    NeRF differentiable renderer
    :param n_coarse number of coarse (binned uniform) samples
    :param n_fine number of fine (importance) samples
    :param n_fine_depth number of expected depth samples
    :param noise_std noise to add to sigma. We do not use it
    :param depth_std noise for depth samples
    :param eval_batch_size ray batch size for evaluation
    :param white_bkgd if true, background color is white; else black
    :param sched ray sampling schedule. list containing 3 lists of equal length.
    sched[0] is list of iteration numbers,
    sched[1] is list of coarse sample numbers,
    sched[2] is list of fine sample numbers
    """

    def __init__(
        self,
        n_coarse=128,
        n_fine=64,
        n_fine_depth=32,
        noise_std=0.0,
        depth_std=0.4,
        eval_batch_size=100000,
        white_bkgd=False,
        sched=None,
        enable_refr=False,
        enable_refl=False,
        stage=1,
        tet_scale=1.0,
        sphere_radius=1.0,
        ior=1.5,
        use_cone=False,
        use_grid=False,
        use_sdf=False,
        use_progressive_encoder=False,
        sdf_threshold=5.0e-5,
        line_search_step=0.5,
        line_step_iters=1,
        sphere_tracing_iters=50,
        n_steps=100,
        n_secant_steps=8,
    ):
        super().__init__()
        self.sdf_threshold = sdf_threshold
        self.sphere_tracing_iters = sphere_tracing_iters
        self.line_step_iters = line_step_iters
        self.line_search_step = line_search_step
        self.n_steps = n_steps
        self.n_secant_steps = n_secant_steps

        self.stage = stage
        self.tet_scale = tet_scale
        self.sphere_radius = sphere_radius
        self.use_grid = use_grid
        self.use_sdf = use_sdf
        self.use_cone = use_cone
        self.use_progressive_encoder = use_progressive_encoder
        self.enable_refr = enable_refr
        self.enable_refl = enable_refl
        self.n_coarse = n_coarse
        self.ior = ior
        self.n_fine = n_fine
        self.n_fine_depth = n_fine_depth
        self.cone_samp_num = 16
        self.cone_alpha = math.radians(5)

        self.noise_std = noise_std
        self.depth_std = depth_std

        self.eval_batch_size = eval_batch_size
        self.white_bkgd = white_bkgd
        self.using_fine = self.n_fine > 0
        self.sched = sched
        if sched is not None and len(sched) == 0:
            self.sched = None
        self.register_buffer(
            "iter_idx", torch.tensor(0, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "last_sched", torch.tensor(0, dtype=torch.long), persistent=True
        )

        # config dmtet
        self.tet_grid_size = 128
        tets = np.load("data/tets/{}_tets.npz".format(self.tet_grid_size))
        self.verts = (
            -torch.tensor(tets["vertices"], dtype=torch.float32, device="cuda")
            * self.tet_scale
        )  # covers [-1, 1]
        self.indices = torch.tensor(tets["indices"], dtype=torch.long, device="cuda")
        self.dmtet = DMTet("cuda")
        # file_path = "128tetsmesh.obj"
        # with open(file_path, 'w') as f:
        #     # 写入顶点信息
        #     for v in self.verts:
        #         f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            
        #     # 写入面信息
        #     for face in self.indices:
        #         f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1} {face[3]+1}\n")

        if self.use_grid:
            # use grid to represent sdf and deformation, not suggested
            if stage == 1:
                mesh = trimesh.load("data/init/sphere.obj", force="mesh")
                scale = 1.5 / np.array(mesh.bounds[1] - mesh.bounds[0]).max()
                center = np.array(mesh.bounds[1] + mesh.bounds[0]) / 2
                mesh.vertices = (mesh.vertices - center) * scale

                BVH = cubvh.cuBVH(
                    mesh.vertices, mesh.faces
                )  # build with numpy.ndarray/torch.Tensor
                sdf, face_id, _ = BVH.signed_distance(
                    self.verts, return_uvw=False, mode="watertight"
                )
                sdf *= -1  # INNER is POSITIVE
                self.sdf = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
            else:
                self.sdf = torch.nn.Parameter(
                    torch.zeros_like(self.verts[..., 0]), requires_grad=False
                )
            self.register_parameter("sdf", self.sdf)

            self.deform = torch.nn.Parameter(
                torch.zeros_like(self.verts), requires_grad=True
            )
            self.register_parameter("deform", self.deform)
        else:
            # use mlp to represent sdf and deform
            self.encoder, self.in_dim = get_encoder("frequency_torch", input_dim=3)
            self.sdf_and_deform_mlp = MLP(self.in_dim, 4, 32, 3, False)  # 4, 32, 3
            self.encoder.to("cuda")
            self.sdf_and_deform_mlp.to("cuda")
            if stage == 1:
                # init sdf with base mesh
                print(f"[INFO] init sdf from base mesh")
                mesh = trimesh.load("data/init/sphere.obj", force="mesh")
                scale = (
                    2.0 / np.array(mesh.bounds[1] - mesh.bounds[0]).max()
                )  # if use eikonal dataset, change 1.0 to 0.2
                center = np.array(mesh.bounds[1] + mesh.bounds[0]) / 2
                mesh.vertices = (mesh.vertices - center) * scale

                BVH = cubvh.cuBVH(
                    mesh.vertices, mesh.faces
                )  # build with numpy.ndarray/torch.Tensor
                sdf, face_id, _ = BVH.signed_distance(
                    self.verts, return_uvw=False, mode="watertight"
                )
                sdf *= -1  # INNER is POSITIVE

                # pretraining
                loss_fn = torch.nn.MSELoss()
                optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-3)

                pretrain_iters = 10000*5
                batch_size = 10240
                print(f"[INFO] start SDF pre-training ")
                for i in tqdm.tqdm(range(pretrain_iters)):
                    rand_idx = torch.randint(0, self.verts.shape[0], (batch_size,))
                    p = self.verts[rand_idx]
                    ref_value = sdf[rand_idx]
                    output = self.sdf_and_deform_mlp(self.encoder(p))
                    loss = loss_fn(output[..., 0], ref_value)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if i % 1000 == 0:
                        print(f"[INFO] SDF pre-train: {loss.item()}")

                print(f"[INFO] SDF pre-train final loss: {loss.item()}")
                del mesh, BVH

        edges_idx = torch.tensor(
            [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long, device="cuda"
        )  # six edges for each tetrahedron.
        edges = self.indices[:, edges_idx].reshape(-1, 2)  # [M * 6, 2]
        edges = torch.sort(edges, dim=1)[0]
        self.edges = torch.unique(edges, dim=0)

        def initial_guess_material(geometry):
            kd_min, kd_max = torch.tensor(
                [0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device="cuda"
            ), torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
            ks_min, ks_max = torch.tensor(
                [0.0, 0.0, 0.0], dtype=torch.float32, device="cuda"
            ), torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
            nrm_min, nrm_max = torch.tensor(
                [-1.0, -1.0, -1.0], dtype=torch.float32, device="cuda"
            ), torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
            mlp_min = torch.cat((kd_min[0:3], ks_min, nrm_min), dim=0)
            mlp_max = torch.cat((kd_max[0:3], ks_max, nrm_max), dim=0)
            mlp_map_opt = mlptexture.MLPTexture3D(
                geometry.getAABB(), channels=9, min_max=[mlp_min, mlp_max]
            )
            mat = material.Material({"kd_ks_normal": mlp_map_opt})
            mat["bsdf"] = "pbr"
            return mat

        self.glctx = dr.RasterizeCudaContext()
        self.material = initial_guess_material(self)
        self.lgt = light.create_trainable_env_rnd(512, scale=0.0, bias=1.0)
        self.lgt.build_mips()
        self.material.requires_grad = False
        self.lgt.base.requires_grad = False

    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values

    def init_tet(self, mesh_path, geometry_name=None):
        # load mesh
        if not os.path.exists(mesh_path):
            raise FileNotFoundError("mesh not found: {}".format(mesh_path))
        mesh = trimesh.load_mesh(mesh_path)
        if geometry_name is not None:
            mesh = mesh.geometry[geometry_name]

        # init sdf
        BVH = cubvh.cuBVH(mesh.vertices, mesh.faces)
        sdf, _, _ = BVH.signed_distance(self.verts, return_uvw=False, mode="watertight")
        sdf *= -1  # INNER is POSITIVE, also make it stronger
        self.sdf = torch.nn.Parameter(
            torch.zeros_like(self.verts[..., 0]), requires_grad=False
        )
        self.deform = torch.nn.Parameter(
            torch.zeros_like(self.verts), requires_grad=False
        )
        self.sdf.data += sdf.to(self.sdf.data.dtype).clamp(-1, 1)
        sdf = self.sdf
        deform = torch.tanh(self.deform) / self.tet_grid_size
        verts, faces, _, _ = self.dmtet(self.verts + deform, sdf, self.indices)

        # get normals
        i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
        v0, v1, v2 = verts[i0, :], verts[i1, :], verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0)
        self.face_normals = face_normals / torch.sqrt(
            torch.clamp(
                torch.sum(face_normals * face_normals, -1, keepdim=True), min=1e-20
            )
        )

        verts = verts.detach().cpu().numpy()
        faces = faces.detach().cpu().numpy()
        self.mesh = trimesh.Trimesh(verts, faces, process=False)

    def sample_coarse(self, rays):
        """
        Stratified sampling. Note this is different from original NeRF slightly.
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :return (B, Kc)
        """
        device = rays.device
        near, far = rays[:, -2:-1], rays[:, -1:]  # (B, 1)

        step = 1.0 / self.n_coarse
        B = rays.shape[0]
        samp_num = self.n_coarse if not self.use_cone else self.cone_samp_num
        z_steps = torch.linspace(0, 1 - step, samp_num, device=device)  # (Kc)
        z_steps = z_steps.unsqueeze(0).repeat(B, 1)  # (B, Kc)
        z_steps += torch.rand_like(z_steps) * step
        return near * (1 - z_steps) + far * z_steps  # (B, Kc)

    def sample_fine(self, rays, weights):
        """
        Weighted stratified (importance) sample
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param weights (B, Kc)
        :return (B, Kf-Kfd)
        """
        device = rays.device
        B = rays.shape[0]

        weights = weights.detach() + 1e-5  # Prevent division by zero
        pdf = weights / torch.sum(weights, -1, keepdim=True)  # (B, Kc)
        cdf = torch.cumsum(pdf, -1)  # (B, Kc)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (B, Kc+1)

        u = torch.rand(
            B, self.n_fine - self.n_fine_depth, dtype=torch.float32, device=device
        )  # (B, Kf)
        inds = torch.searchsorted(cdf, u, right=True).float() - 1.0  # (B, Kf)
        inds = torch.clamp_min(inds, 0.0)
        z_steps = (inds + torch.rand_like(inds)) / self.n_coarse  # (B, Kf)
        near, far = rays[:, -2:-1], rays[:, -1:]  # (B, 1)
        z_samp = near * (1 - z_steps) + far * z_steps  # (B, Kf)
        return z_samp

    def sample_fine_depth(self, rays, depth, samp_num=32):
        """
        Sample around specified depth
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param depth (B)
        :return (B, Kfd)
        """
        z_samp = depth.unsqueeze(1).repeat((1, samp_num))
        z_samp += torch.randn_like(z_samp) * self.depth_std
        # Clamp does not support tensor bounds
        z_samp = torch.max(torch.min(z_samp, rays[:, -1:]), rays[:, -2:-1])
        return z_samp

    def sample_rays(self, rays, alpha=None, samp_num=128):
        B = rays.shape[0]
        ray_samp = rays.unsqueeze(1).repeat((1, samp_num, 1))

        confidence = norm(loc=0, scale=self.cone_alpha / 3)
        alpha = torch.tensor(self.cone_alpha).repeat((rays.shape[0]))

        cos_alpha = torch.cos(alpha).to(device=rays.device)
        cone_axis = rays[..., 3:6].unsqueeze(2)
        pole = torch.Tensor([0, 0, 1.0]).to(device=rays.device).unsqueeze(0)

        # calculate the roation matrix with Rodrigues formula
        cos_cone_axis = torch.sum(cone_axis.squeeze(-1) * pole, dim=-1)
        cone_num = cos_cone_axis.shape[0]
        I = torch.eye(3).unsqueeze(0).repeat(cone_num, 1, 1).to(device=rays.device)
        cross_dot = torch.cross(pole.repeat(cone_num, 1), cone_axis[..., 0])

        # skew-symmetric cross-product matrix
        cross_matrix = torch.zeros_like(I)
        cross_matrix[..., 0, 1] = -cross_dot[..., 2]
        cross_matrix[..., 0, 2] = cross_dot[..., 1]
        cross_matrix[..., 1, 0] = cross_dot[..., 2]
        cross_matrix[..., 1, 2] = -cross_dot[..., 0]
        cross_matrix[..., 2, 0] = -cross_dot[..., 1]
        cross_matrix[..., 2, 1] = cross_dot[..., 0]
        R_matrix = (
            I
            + cross_matrix
            + torch.matmul(cross_matrix, cross_matrix)
            / (1 + cos_cone_axis).unsqueeze(1).unsqueeze(1)
        )

        sampled_theta = (
            2
            * math.pi
            * torch.rand(samp_num * B).view(B, samp_num).to(device=rays.device)
        )  # random sample from [0,2*pi]

        cos_alpha = cos_alpha.unsqueeze(1).repeat((1, samp_num))
        sampled_z = (1 - cos_alpha) * torch.rand(samp_num * B).view(B, samp_num).to(
            device=rays.device
        ) + cos_alpha

        samp_confidence = torch.tensor(confidence.pdf(torch.acos(sampled_z.cpu())))
        tmp_z = torch.sqrt(1 - sampled_z**2)

        x = tmp_z * torch.cos(sampled_theta)
        y = tmp_z * torch.sin(sampled_theta)
        viewdir = ray_samp[..., 3:6].clone()
        viewdir[..., 0] = x
        viewdir[..., 1] = y
        viewdir[..., 2] = sampled_z

        R = R_matrix.unsqueeze(1).repeat((1, samp_num, 1, 1))
        v = viewdir.unsqueeze(2).repeat(1, 1, 3, 1)
        transformed_v = torch.sum(R * v, dim=-1)
        ray_samp[..., 3:6] = transformed_v
        return ray_samp, samp_confidence

    def adjust_normal(self, normals, in_dir):
        in_dot = (in_dir * normals).sum(dim=-1)
        mask = in_dot > 0
        normals[mask] = -normals[mask]  # make sure normal point to in_dir
        return normals

    def render_mask(self, rays, mvp, h, w, global_step=0):
        # rays:[B, N, 8]
        mvp = mvp.unsqueeze(0)  # [B, 4, 4]
        campos = rays[..., 0, 0:3]  # only need one ray per batch

        # get mesh
        if self.use_grid:
            sdf = self.sdf
            deform = torch.tanh(self.deform) / self.tet_grid_size
        else:
            pred = self.sdf_and_deform_mlp(
                self.encoder(self.verts, step_id=global_step)
                if self.use_progressive_encoder
                else self.encoder(self.verts)
            )
            sdf, deform = pred[:, 0], pred[:, 1:]
            deform = torch.tanh(deform) / self.tet_grid_size
        verts, faces, uvs, uv_idx = self.dmtet(self.verts + deform, sdf, self.indices)

        if not self.use_grid:
            # eikonal loss, regularize surface normal and surface sdf
            verts_ = verts.detach().clone().requires_grad_(True)
            pred_ = self.sdf_and_deform_mlp(
                self.encoder(verts_, step_id=global_step)
                if self.use_progressive_encoder
                else self.encoder(verts_)
            )
            verts_sdf = pred_[:, 0]
            d_points = torch.ones_like(
                verts_sdf, requires_grad=False, device=verts_sdf.device
            )
            verts_grad = torch.autograd.grad(
                outputs=verts_sdf,
                inputs=verts_,
                grad_outputs=d_points,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            verts_sdf_loss = 0
            verts_grad_loss = ((verts_grad.norm(2, dim=-1) - 1) ** 2).mean()
            eikonal_loss_coeff = 0.1
            ek_loss = verts_sdf_loss + eikonal_loss_coeff * verts_grad_loss
        else:
            ek_loss = 0

        # run mesh operations to generate tangent space
        imesh = mesh.Mesh(
            verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=self.material
        )
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)

        h = 272
        buffers = render.render_mesh(
            self.glctx,
            imesh,
            mvp,
            campos,
            self.lgt,
            [h, w],
            1,
            msaa=True,
            bsdf="normal",
        )

        def shade_normal():
            mask = buffers["shaded"][0, ..., :4]
            rgb = (mask[..., :3].detach().cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(rgb, mode="RGB")
            image.save("test_normal.png")

        shade_normal()
        mask = buffers["shaded"][..., 3:]
        return (mask[..., 0], ek_loss)

    def intersection_with_mesh(self, rays, mesh, far=False):
        vertices = mesh.vertices
        faces = vertices[mesh.faces]
        faces = np.array(faces, dtype="f4")
        ray_num = rays.shape[0]
        valid_mask = torch.zeros(ray_num, dtype=torch.bool)
        intersection_depth = torch.zeros(ray_num)
        intersection_face = []  # torch.zeros(ray_num)
        normals = torch.zeros(ray_num, 3)
        rays = rays.detach().cpu().numpy()
        ray_origins = rays[..., :3]  # 3*1
        ray_directions = rays[..., 3:6]  # 3*1
        vn = trimesh.smoothing.get_vertices_normals(mesh)
        for i in range(ray_num):
            result = mesh_raycast.raycast(ray_origins[i], ray_directions[i], mesh=faces)
            if len(result) > 0:
                if far:
                    intersection = max(result, key=lambda x: x["distance"])
                else:
                    intersection = min(result, key=lambda x: x["distance"])
                valid_mask[i] = True
                face_vert_idxs = mesh.faces[intersection["face"]]
                face_vert_xyzs = vertices[face_vert_idxs]
                face_vert_normals = vn[face_vert_idxs]
                face_vert_dis = np.linalg.norm(
                    intersection["point"] - face_vert_xyzs, axis=1
                )
                face_vn_coeffs = 1 / face_vert_dis
                face_vn_coeffs /= face_vn_coeffs.sum()
                normals[i] = torch.tensor(
                    (face_vert_normals * face_vn_coeffs[:, None]).sum(axis=0)
                )
                # normals[i] = torch.tensor(intersection["normal"])
                intersection_face.append(intersection["face"])
                intersection_depth[i] = intersection["distance"]
        return valid_mask, intersection_depth, normals, torch.tensor(intersection_face)

    def get_sphere_intersection(self, rays, r=1.0):
        B, _ = rays.shape
        cam_loc = rays[:, :3]
        ray_directions = rays[:, 3:6]
        ray_cam_dot = torch.sum(torch.mul(ray_directions, cam_loc), dim=1)
        under_sqrt = ray_cam_dot**2 - (cam_loc.norm(2, 1) ** 2 - r**2)
        mask_intersect = under_sqrt > 0

        sphere_intersections = torch.zeros(B, 2).cuda().float()
        sphere_intersections[mask_intersect] = (
            torch.sqrt(under_sqrt[mask_intersect]).unsqueeze(-1)
            * torch.Tensor([-1, 1]).cuda().float()
        )
        sphere_intersections[mask_intersect] -= ray_cam_dot.reshape(-1)[
            mask_intersect
        ].unsqueeze(-1)
        sphere_intersections = sphere_intersections.clamp_min(0.0)
        return sphere_intersections, mask_intersect

    def sphere_tracing(
        self,
        rays,
        mask_intersect,
        sphere_intersections,
    ):
        """Run sphere tracing algorithm for max iterations from both sides of unit sphere intersection"""
        B, _ = rays.shape
        cam_loc = rays[:, :3]
        ray_directions = rays[:, 3:6]
        sphere_intersections_points = sphere_intersections.unsqueeze(
            -1
        ) * ray_directions.unsqueeze(1) + cam_loc.unsqueeze(1)
        unfinished_mask_start = mask_intersect.clone()
        unfinished_mask_end = mask_intersect.clone()

        # Initialize start current points
        curr_start_points = torch.zeros(B, 3).cuda().float()
        curr_start_points[unfinished_mask_start] = sphere_intersections_points[
            :, 0, :
        ].reshape(-1, 3)[unfinished_mask_start]
        acc_start_dis = torch.zeros(B).cuda().float()
        acc_start_dis[unfinished_mask_start] = sphere_intersections.reshape(-1, 2)[
            unfinished_mask_start, 0
        ]

        # Initialize end current points
        curr_end_points = torch.zeros(B, 3).cuda().float()
        curr_end_points[unfinished_mask_end] = sphere_intersections_points[
            :, 1, :
        ].reshape(-1, 3)[unfinished_mask_end]
        acc_end_dis = torch.zeros(B).cuda().float()
        acc_end_dis[unfinished_mask_end] = sphere_intersections.reshape(-1, 2)[
            unfinished_mask_end, 1
        ]

        # Initizliae min and max depth
        min_dis = acc_start_dis.clone()
        max_dis = acc_end_dis.clone()

        # Iterate on the rays (from both sides) till finding a surface
        iters = 0

        next_sdf_start = torch.zeros_like(acc_start_dis).cuda()
        next_sdf_start[unfinished_mask_start] = -self.sdf_and_deform_mlp(
            self.encoder(curr_start_points[unfinished_mask_start])
        )[:, 0]

        next_sdf_end = torch.zeros_like(acc_end_dis).cuda()
        next_sdf_end[unfinished_mask_end] = -self.sdf_and_deform_mlp(
            self.encoder(curr_end_points[unfinished_mask_end])
        )[:, 0]

        # embed()
        # assert torch.all(next_sdf_start >= 0) and torch.all(next_sdf_end >= 0)
        while True:
            # Update sdf
            curr_sdf_start = torch.zeros_like(acc_start_dis).cuda()
            curr_sdf_start[unfinished_mask_start] = next_sdf_start[
                unfinished_mask_start
            ]
            curr_sdf_start[curr_sdf_start <= self.sdf_threshold] = 0

            curr_sdf_end = torch.zeros_like(acc_end_dis).cuda()
            curr_sdf_end[unfinished_mask_end] = next_sdf_end[unfinished_mask_end]
            curr_sdf_end[curr_sdf_end <= self.sdf_threshold] = 0

            # Update masks
            unfinished_mask_start = unfinished_mask_start & (
                curr_sdf_start > self.sdf_threshold
            )
            unfinished_mask_end = unfinished_mask_end & (
                curr_sdf_end > self.sdf_threshold
            )

            if (
                unfinished_mask_start.sum() == 0 and unfinished_mask_end.sum() == 0
            ) or iters == self.sphere_tracing_iters:
                break
            iters += 1

            # Make step
            # Update distance
            acc_start_dis = acc_start_dis + curr_sdf_start
            acc_end_dis = acc_end_dis - curr_sdf_end

            # Update points
            curr_start_points = cam_loc + acc_start_dis.unsqueeze(1) * ray_directions
            curr_end_points = cam_loc + acc_end_dis.unsqueeze(1) * ray_directions

            next_sdf_start = torch.zeros_like(acc_start_dis).cuda()
            next_sdf_start[unfinished_mask_start] = -self.sdf_and_deform_mlp(
                self.encoder(curr_start_points[unfinished_mask_start])
            )[:, 0]

            next_sdf_end = torch.zeros_like(acc_end_dis).cuda()
            next_sdf_end[unfinished_mask_end] = -self.sdf_and_deform_mlp(
                self.encoder(curr_end_points[unfinished_mask_end])
            )[:, 0]

            # Fix points which wrongly crossed the surface
            not_projected_start = next_sdf_start < 0
            not_projected_end = next_sdf_end < 0
            not_proj_iters = 0
            while (
                not_projected_start.sum() > 0 or not_projected_end.sum() > 0
            ) and not_proj_iters < self.line_step_iters:
                # Step backwards
                acc_start_dis[not_projected_start] -= (
                    (1 - self.line_search_step) / (2**not_proj_iters)
                ) * curr_sdf_start[not_projected_start]
                curr_start_points[not_projected_start] = (
                    cam_loc + acc_start_dis.unsqueeze(1) * ray_directions
                )[not_projected_start]

                acc_end_dis[not_projected_end] += (
                    (1 - self.line_search_step) / (2**not_proj_iters)
                ) * curr_sdf_end[not_projected_end]
                curr_end_points[not_projected_end] = (
                    cam_loc + acc_end_dis.unsqueeze(1) * ray_directions
                )[not_projected_end]

                # Calc sdf
                next_sdf_start[not_projected_start] = -self.sdf_and_deform_mlp(
                    self.encoder(curr_start_points[not_projected_start])
                )[:, 0]

                next_sdf_end[not_projected_end] = -self.sdf_and_deform_mlp(
                    self.encoder(curr_end_points[not_projected_end])
                )[:, 0]

                # Update mask
                not_projected_start = next_sdf_start < 0
                not_projected_end = next_sdf_end < 0
                not_proj_iters += 1

            unfinished_mask_start = unfinished_mask_start & (
                acc_start_dis < acc_end_dis
            )
            unfinished_mask_end = unfinished_mask_end & (acc_start_dis < acc_end_dis)

        return (
            curr_start_points,
            curr_end_points,
            unfinished_mask_start,
            unfinished_mask_end,
            acc_start_dis,
            acc_end_dis,
            min_dis,
            max_dis,
        )

    def ray_sampler(self, rays, sampler_min_max, sampler_mask, far=False):
        """Sample the ray in a given range and run secant on rays which have sign transition"""
        B, _ = rays.shape
        cam_loc = rays[:, :3]
        ray_directions = rays[:, 3:6]

        sampler_pts = torch.zeros(B, 3).cuda().float()
        sampler_dists = torch.zeros(B).cuda().float()

        intervals_dist = torch.linspace(0, 1, steps=self.n_steps).cuda()
        pts_intervals = sampler_min_max[:, 0].unsqueeze(-1) + intervals_dist * (
            sampler_min_max[:, 1] - sampler_min_max[:, 0]
        ).unsqueeze(-1)
        points = cam_loc.reshape(B, 1, 3) + pts_intervals.unsqueeze(
            -1
        ) * ray_directions.unsqueeze(1)

        # Get the non convergent rays
        mask_intersect_idx = torch.nonzero(sampler_mask).flatten()
        points = points.reshape((-1, self.n_steps, 3))[sampler_mask, :, :]
        pts_intervals = pts_intervals.reshape((-1, self.n_steps))[sampler_mask]

        sdf_val_all = []
        for pnts in torch.split(points.reshape(-1, 3), 100000, dim=0):
            sdf_val_all.append(-self.sdf_and_deform_mlp(self.encoder(pnts))[:, 0])
        sdf_val = torch.cat(sdf_val_all).reshape(-1, self.n_steps)

        tmp = torch.sign(sdf_val) * torch.arange(
            self.n_steps, 0, -1
        ).cuda().float().reshape(
            (1, self.n_steps)
        )  # Force argmin to return the first min value
        sampler_pts_ind = torch.argmin(tmp, -1) if not far else torch.argmax(tmp, -1)
        sampler_pts[mask_intersect_idx] = points[
            torch.arange(points.shape[0]), sampler_pts_ind, :
        ]
        sampler_dists[mask_intersect_idx] = pts_intervals[
            torch.arange(pts_intervals.shape[0]), sampler_pts_ind
        ]

        # true_surface_pts = object_mask[sampler_mask]
        net_surface_pts = sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind] < 0

        # take points with minimal SDF value for P_out pixels
        # p_out_mask = ~(true_surface_pts & net_surface_pts)
        p_out_mask = ~net_surface_pts
        n_p_out = p_out_mask.sum()
        if n_p_out > 0:
            out_pts_idx = torch.argmin(sdf_val[p_out_mask, :], -1)
            sampler_pts[mask_intersect_idx[p_out_mask]] = points[p_out_mask, :, :][
                torch.arange(n_p_out), out_pts_idx, :
            ]
            sampler_dists[mask_intersect_idx[p_out_mask]] = pts_intervals[
                p_out_mask, :
            ][torch.arange(n_p_out), out_pts_idx]

        # Get Network object mask
        sampler_net_obj_mask = sampler_mask.clone()
        sampler_net_obj_mask[mask_intersect_idx[~net_surface_pts]] = False

        # Run Secant method
        secant_pts = net_surface_pts
        n_secant_pts = secant_pts.sum()
        if n_secant_pts > 0:
            # Get secant z predictions
            z_high = pts_intervals[
                torch.arange(pts_intervals.shape[0]), sampler_pts_ind
            ][secant_pts]
            sdf_high = sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind][
                secant_pts
            ]
            z_low = pts_intervals[secant_pts][
                torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1
            ]
            sdf_low = sdf_val[secant_pts][
                torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1
            ]
            cam_loc_secant = (
                cam_loc.unsqueeze(1)
                # .repeat(1, num_pixels, 1)
                .reshape((-1, 3))[mask_intersect_idx[secant_pts]]
            )
            ray_directions_secant = ray_directions.reshape((-1, 3))[
                mask_intersect_idx[secant_pts]
            ]
            z_pred_secant = self.secant(
                sdf_low,
                sdf_high,
                z_low,
                z_high,
                cam_loc_secant,
                ray_directions_secant,
            )
            # Get points
            sampler_pts[mask_intersect_idx[secant_pts]] = (
                cam_loc_secant + z_pred_secant.unsqueeze(-1) * ray_directions_secant
            )
            sampler_dists[mask_intersect_idx[secant_pts]] = z_pred_secant
        return sampler_pts, sampler_net_obj_mask, sampler_dists

    def secant(self, sdf_low, sdf_high, z_low, z_high, cam_loc, ray_directions):
        """Runs the secant method for interval [z_low, z_high] for n_secant_steps"""
        z_pred = -sdf_low * (z_high - z_low) / (sdf_high - sdf_low) + z_low
        for i in range(self.n_secant_steps):
            p_mid = cam_loc + z_pred.unsqueeze(-1) * ray_directions
            sdf_mid = -self.sdf_and_deform_mlp(self.encoder(p_mid))[:, 0]
            ind_low = sdf_mid > 0
            if ind_low.sum() > 0:
                z_low[ind_low] = z_pred[ind_low]
                sdf_low[ind_low] = sdf_mid[ind_low]
            ind_high = sdf_mid < 0
            if ind_high.sum() > 0:
                z_high[ind_high] = z_pred[ind_high]
                sdf_high[ind_high] = sdf_mid[ind_high]
            z_pred = -sdf_low * (z_high - z_low) / (sdf_high - sdf_low) + z_low
        return z_pred

    def intersection_with_sdf(self, rays, far=False):
        B, _ = rays.shape
        sphere_intersections, mask_intersect = self.get_sphere_intersection(
            rays, r=self.sphere_radius
        )
        (
            curr_start_points,
            curr_end_points,
            unfinished_mask_start,
            unfinished_mask_end,
            acc_start_dis,
            acc_end_dis,
            min_dis,
            max_dis,
        ) = self.sphere_tracing(
            rays,
            mask_intersect,
            sphere_intersections,
        )

        # hit and non convergent rays
        network_object_mask = acc_start_dis < acc_end_dis
        # The non convergent rays should be handled by the sampler
        sampler_mask = unfinished_mask_start if not far else unfinished_mask_end
        sampler_net_obj_mask = torch.zeros_like(sampler_mask).bool().cuda()
        if sampler_mask.sum() > 0:
            sampler_min_max = torch.zeros((B, 2)).cuda()
            sampler_min_max.reshape(-1, 2)[sampler_mask, 0] = acc_start_dis[
                sampler_mask
            ]
            sampler_min_max.reshape(-1, 2)[sampler_mask, 1] = acc_end_dis[sampler_mask]
            sampler_pts, sampler_net_obj_mask, sampler_dists = self.ray_sampler(
                rays, sampler_min_max, sampler_mask, far=far
            )
            if not far:
                curr_start_points[sampler_mask] = sampler_pts[sampler_mask]
                acc_start_dis[sampler_mask] = sampler_dists[sampler_mask]
            else:
                curr_end_points[sampler_mask] = sampler_pts[sampler_mask]
                acc_end_dis[sampler_mask] = sampler_dists[sampler_mask]
            network_object_mask[sampler_mask] = sampler_net_obj_mask[sampler_mask]

        valid_mask = network_object_mask
        intersection_depth = acc_start_dis if not far else acc_end_dis

        # access normals
        hit_points = curr_start_points if not far else curr_end_points
        with torch.enable_grad():
            hit_points = hit_points.clone().detach().requires_grad_(True)
            pred = self.sdf_and_deform_mlp(self.encoder(hit_points))
            hit_points_sdf = pred[:, 0]
            d_points = torch.ones_like(
                hit_points_sdf, requires_grad=False, device=hit_points_sdf.device
            )
            hit_points_sdf.requires_grad_(True)
        verts_grad = torch.autograd.grad(
            outputs=hit_points_sdf,
            inputs=hit_points,
            grad_outputs=d_points,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        normals = torch.zeros(B, 3)
        normals[valid_mask] = verts_grad[valid_mask].to(device=normals.device)
        normals[valid_mask] = normals[valid_mask] / normals[valid_mask].norm(
            2, dim=1
        ).unsqueeze(-1)
        return valid_mask, intersection_depth, normals

    def trace_ray_sdf(
        self, rays, model=None, n=1.5, reflection=False, refraction=False
    ):
        """
        Traces rays and performs reflection and refraction when hitting the surface.
        :param rays: [N,8], depths: [N,1], normals: [N,3]
        """
        # first intersection
        (
            valid_mask,
            intersection_depth,
            normals,
        ) = self.intersection_with_sdf(rays, far=False)
        normals = normals.to(device=rays.device)
        normals = normals[valid_mask]
        valid_mask = valid_mask.to(device=rays.device)
        hit_rays = rays[valid_mask]
        init_depth = intersection_depth[valid_mask].to(device=rays.device)
        hit_pts = hit_rays[..., :3] + init_depth.unsqueeze(1) * hit_rays[..., 3:6]
        in_dir = hit_rays[..., 3:6]
        normals = self.adjust_normal(normals, in_dir)
        cos_i = (-in_dir * normals).sum(dim=1)

        # reflection
        if reflection:
            refl_rays = torch.zeros_like(hit_rays)
            refl_rays[:, :3] = hit_pts
            if self.stage == 3:
                model_input = torch.cat([hit_pts, normals], dim=-1)
                if valid_mask.sum() == 0:
                    normals_ = torch.zeros_like(hit_pts)
                else:
                    normals_ = model(model_input)
                    normals_ = normals_ / (
                        torch.norm(normals_, dim=-1).unsqueeze(-1) + 1e-9
                    )
            refl_dir = in_dir + 2 * cos_i.unsqueeze(-1) * (
                normals if self.stage != 3 else normals_
            )
            refl_dir = refl_dir / torch.norm(refl_dir, dim=-1).unsqueeze(-1)
            refl_rays[:, 3:6] = refl_dir
            refl_rays[:, 6:] = rays[valid_mask][:, 6:]
            if self.stage == 3:
                dir_delta = (
                    1 - torch.nn.functional.cosine_similarity(normals, normals_, dim=-1)
                ).sum()

            if not refraction:
                return (
                    rays,
                    valid_mask,
                    None,
                    refl_rays,
                    None,
                    (None if self.stage != 3 else dir_delta),
                )

        if self.stage == 3:
            raise NotImplementedError
        # first refraction
        refr_rays = torch.zeros_like(hit_rays)
        n_ = 1 / n
        cos_o = torch.sqrt(1 - (n_**2) * (1 - cos_i**2))
        refr_dir = n_ * in_dir + (n_ * cos_i - cos_o).unsqueeze(-1) * normals
        refr_dir = refr_dir / torch.norm(refr_dir, dim=-1).unsqueeze(-1)
        refr_rays[:, :3] = hit_pts - 10 * refr_dir
        refr_rays[:, 3:6] = refr_dir
        fresnel_1 = self.Fresnel_term(n, in_dir, refr_dir, normals)

        # second intersection
        (
            valid_mask_,
            intersection_depth_,
            normals_,
        ) = self.intersection_with_sdf(refr_rays, far=True)

        normals_ = normals_.to(device=rays.device)
        normals_ = normals_[valid_mask_]
        valid_mask_ = valid_mask_.to(device=rays.device)
        hit_rays_ = refr_rays[valid_mask_]
        zdepth = intersection_depth_[valid_mask_].to(device=rays.device)
        hit_pts_ = hit_rays_[..., :3] + zdepth.unsqueeze(1) * hit_rays_[..., 3:6]
        in_dir_ = hit_rays_[..., 3:6]
        normals_ = self.adjust_normal(normals_, in_dir_)

        # second refraction
        refr_rays_ = torch.zeros_like(hit_rays_)
        refr_rays_[:, :3] = hit_pts_
        cos_i_ = (-in_dir_ * normals_).sum(dim=1)
        tmp = 1 - (n**2) * (1 - cos_i_**2)
        tmp[tmp < 0] = 0.0
        cos_o_ = torch.sqrt(tmp)
        refr_dir_ = n * in_dir_ + (n * cos_i_ - cos_o_).unsqueeze(-1) * normals_
        refr_dir_ = refr_dir_ / torch.norm(refr_dir_, dim=-1).unsqueeze(-1)
        refr_rays_[:, 3:6] = refr_dir_

        # dealing with rays missing the second intersection (unwatertight surface)
        invalid_mask_ = (valid_mask_ == False).to(device=rays.device)
        refr_rays_ = copy_index(refr_rays_, invalid_mask_, refr_rays[invalid_mask_])
        out_rays = copy_index(rays, valid_mask_, refr_rays_)
        fresnel = 1 - fresnel_1
        return out_rays, valid_mask, None, None, fresnel, None

    def Fresnel_term(self, n, in_dir, out_dir, normal):
        in_dot = (in_dir * normal).sum(-1)
        out_dot = (out_dir * normal).sum(-1)

        F = ((in_dot - n * out_dot) / (in_dot + n * out_dot)) ** 2 + (
            (n * in_dot - out_dot) / (n * in_dot + out_dot)
        ) ** 2
        return F / 2

    def trace_ray_mesh(
        self, rays, model=False, n=1.5, reflection=False, refraction=False
    ):
        """
        Traces rays and performs reflection and refraction when hitting the surface.
        :param rays: [N,8], depths: [N,1], normals: [N,3]
        """
        face_normals = self.face_normals
        mesh = self.mesh
        # first intersection
        (
            valid_mask,
            intersection_depth,
            normals,
            intersection_face,
        ) = self.intersection_with_mesh(rays, mesh, far=False)
        if len(intersection_face) == 0:
            normals = face_normals[[]]
        else:
            normals = face_normals[intersection_face]
        normals = normals.to(device=rays.device)
        valid_mask = valid_mask.to(device=rays.device)
        hit_rays = rays[valid_mask]
        init_depth = intersection_depth[valid_mask].to(device=rays.device)
        hit_pts = hit_rays[..., :3] + init_depth.unsqueeze(1) * hit_rays[..., 3:6]
        in_dir = hit_rays[..., 3:6]
        normals = self.adjust_normal(normals, in_dir)
        cos_i = (-in_dir * normals).sum(dim=1)

        # reflection
        if reflection:
            refl_rays = torch.zeros_like(hit_rays)
            refl_rays[:, :3] = hit_pts
            if self.stage == 3:
                model_input = torch.cat([hit_pts, normals], dim=-1)
                if len(intersection_face) == 0:
                    normals_ = torch.zeros_like(hit_pts)
                else:
                    normals_ = model(model_input)
                    normals_ = normals_ / (
                        torch.norm(normals_, dim=-1).unsqueeze(-1) + 1e-6
                    )
                dir_delta = torch.acos(
                    (normals * normals_).sum(-1) / (torch.norm(normals, dim=-1) + 1e-6)
                )
            refl_dir = in_dir + 2 * cos_i.unsqueeze(-1) * (
                normals if self.stage != 3 else normals_
            )
            refl_dir = refl_dir / torch.norm(refl_dir, dim=-1).unsqueeze(-1)
            refl_rays[:, 3:6] = refl_dir
            refl_rays[:, 6:] = rays[valid_mask][:, 6:]

            if not refraction:
                return (
                    rays,
                    valid_mask,
                    None,
                    refl_rays,
                    None,
                    (None if self.stage != 3 else dir_delta),
                )

        # first refraction
        if self.stage == 3:
            raise NotImplementedError
        refr_rays = torch.zeros_like(hit_rays)
        refr_rays[:, :3] = hit_pts
        n_ = 1 / n
        cos_o = torch.sqrt(1 - (n_**2) * (1 - cos_i**2))
        refr_dir = n_ * in_dir + (n_ * cos_i - cos_o).unsqueeze(-1) * normals
        refr_dir = refr_dir / torch.norm(refr_dir, dim=-1).unsqueeze(-1)
        refr_rays[:, 3:6] = refr_dir
        fresnel_1 = self.Fresnel_term(n, in_dir, refr_dir, normals)

        # second intersection
        (
            valid_mask_,
            intersection_depth_,
            normals_,
            intersection_face_,
        ) = self.intersection_with_mesh(refr_rays, mesh, far=True)
        normals_ = normals_.to(device=rays.device)
        if len(intersection_face_) != 0:
            normals_[valid_mask_] = face_normals[intersection_face_]
        else:
            normals_[valid_mask_] = face_normals[[]]
        zdepth = intersection_depth_.to(device=rays.device)
        normals_ = normals_.to(device=rays.device)
        hit_pts_ = refr_rays[..., :3] + zdepth.unsqueeze(1) * refr_rays[..., 3:6]
        in_dir_ = refr_rays[..., 3:6]
        normals_ = self.adjust_normal(normals_, in_dir_)

        # second refraction
        refr_rays_ = torch.zeros_like(hit_rays)
        refr_rays_[:, :3] = hit_pts_
        cos_i = (-in_dir_ * normals_).sum(dim=1)
        tmp = 1 - (n**2) * (1 - cos_i**2)
        tmp[tmp < 0] = 0.0
        cos_o = torch.sqrt(tmp)
        if self.stage == 3:
            model_input = torch.cat([in_dir_, hit_pts_, normals_], dim=-1)
            if len(intersection_face_) == 0:
                dir_delta = torch.zeros_like(hit_pts_)
            else:
                dir_delta = model(model_input)
        refr_dir_ = (
            n * in_dir_
            + (n * cos_i - cos_o).unsqueeze(-1) * normals_
            + (dir_delta if self.stage == 3 else 0)
        )
        refr_dir_ = refr_dir_ / torch.norm(refr_dir_, dim=-1).unsqueeze(-1)
        refr_rays_[:, 3:6] = refr_dir_

        # dealing with rays missing the second intersection (unwatertight surface)
        invalid_mask_ = (valid_mask_ == False).to(device=rays.device)
        refr_rays_ = copy_index(refr_rays_, invalid_mask_, refr_rays[invalid_mask_])

        # update depth: intersection with planes
        refr_rays_[:, 7] = rays[valid_mask][:, 7] - init_depth - zdepth
        new_depth = init_depth

        out_rays = copy_index(rays, valid_mask, refr_rays_)
        fresnel = 1 - fresnel_1
        return (
            out_rays,
            valid_mask,
            new_depth,
            None,
            fresnel,
            (None if self.stage != 3 else dir_delta),
        )

    def composite(self, model, rays, z_samp, coarse=True, sb=0):
        """
        Render RGB and depth for each ray using NeRF alpha-compositing formula,
        given sampled positions along each ray (see sample_*)
        :param model should return (B, (r, g, b, sigma)) when called with (B, (x, y, z))
        should also support 'coarse' boolean argument
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param z_samp z positions sampled for each ray (B, K)
        :param coarse whether to evaluate using coarse NeRF
        :param sb super-batch dimension; 0 = disable
        :return weights (B, K), rgb (B, 3), depth (B)
        """

        with profiler.record_function("renderer_composite"):
            B, K = z_samp.shape
            deltas = z_samp[:, 1:] - z_samp[:, :-1]  # (B, K-1)
            delta_inf = rays[:, -1:] - z_samp[:, -1:]
            deltas = torch.cat([deltas, delta_inf], -1)  # (B, K)

            points = (
                rays[:, None, :3] + z_samp.unsqueeze(2) * rays[:, None, 3:6]
            )  # (B, K, 3)
            points = points.reshape(-1, 3)  # (B*K, 3)

            use_viewdirs = hasattr(model, "use_viewdirs") and model.use_viewdirs
            val_all = []
            if sb > 0:
                points = points.reshape(
                    sb, -1, 3
                )  # (SB, B'*K, 3) B' is real ray batch size
                eval_batch_size = (self.eval_batch_size - 1) // sb + 1
                eval_batch_dim = 1
            else:
                eval_batch_size = self.eval_batch_size
                eval_batch_dim = 0

            split_points = torch.split(points, eval_batch_size, dim=eval_batch_dim)
            if use_viewdirs:
                dim1 = K
                viewdirs = rays[:, None, 3:6].expand(-1, dim1, -1)  # (B, K, 3)
                if sb > 0:
                    viewdirs = viewdirs.reshape(sb, -1, 3)  # (SB, B'*K, 3)
                else:
                    viewdirs = viewdirs.reshape(-1, 3)  # (B*K, 3)
                split_viewdirs = torch.split(
                    viewdirs, eval_batch_size, dim=eval_batch_dim
                )
                for pnts, dirs in zip(split_points, split_viewdirs):
                    val_all.append(model(xyz=pnts, coarse=coarse, viewdirs=dirs))
            else:
                raise NotImplementedError
            points = None
            viewdirs = None
            # (B*K, 4) OR (SB, B'*K, 4)
            out = torch.cat(val_all, dim=eval_batch_dim)
            out = out.reshape(B, K, -1)  # (B, K, 4)

            rgbs = out[..., :3]  # (B, K, 3)
            sigmas = out[..., 3]  # (B, K)

            if self.training and self.noise_std > 0.0:
                sigmas = sigmas + torch.randn_like(sigmas) * self.noise_std

            alphas = 1 - torch.exp(-deltas * torch.relu(sigmas))  # (B, K)
            alphas_shifted = torch.cat(
                [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1
            )  # (B, K+1) = [1, a1, a2, ...]
            T = torch.cumprod(alphas_shifted, -1)  # (B)
            weights = alphas * T[:, :-1]  # (B, K)

            # tag
            if model.viewdir_only:
                rgb_final = torch.mean(rgbs, -2)  # (B, 3)
                depth_final = torch.mean(z_samp, -1)  # (B)
            else:
                rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # (B, 3)
                depth_final = torch.sum(weights * z_samp, -1)  # (B)

            if self.white_bkgd:
                # white background
                pix_alpha = weights.sum(dim=1)  # (B), pixel alpha
                rgb_final = rgb_final + 1 - pix_alpha.unsqueeze(-1)  # (B, 3)

            normal_final = None
            return (weights, rgb_final, depth_final, normal_final)

    def forward(
        self,
        model,
        rays,
        want_weights=False,
    ):
        """
        :model nerf model, should return (SB, B, (r, g, b, sigma))
        when called with (SB, B, (x, y, z)), for multi-object:
        SB = 'super-batch' = size of object batch,
        B  = size of per-object ray batch.
        Should also support 'coarse' boolean argument for coarse NeRF.
        :param rays ray spec [origins (3), directions (3), near (1), far (1)] (SB, B, 8)
        :param want_weights if true, returns compositing weights (SB, B, K)
        :return render dict
        """
        refraction = self.enable_refr
        reflection = self.enable_refl
        reflection_only = reflection & (not refraction)
        trace_ray = self.trace_ray_sdf if self.use_sdf else self.trace_ray_mesh

        with profiler.record_function("renderer_forward"):
            if self.sched is not None and self.last_sched.item() > 0:
                self.n_coarse = self.sched[1][self.last_sched.item() - 1]
                self.n_fine = self.sched[2][self.last_sched.item() - 1]

            assert len(rays.shape) == 3
            superbatch_size = rays.shape[0]
            rays = rays.reshape(-1, 8)  # (SB * B, 8)

            # use cone to sample more rays
            ray_sampling = self.use_cone
            n = self.ior
            if refraction or reflection:
                (
                    rays,
                    valid_mask,
                    new_depth,
                    reflect_rays,
                    fresnel,
                    dir_deltas,
                ) = trace_ray(
                    rays,
                    n=n,
                    reflection=reflection,
                    refraction=refraction,
                )
                if ray_sampling:
                    (
                        rays_,
                        valid_mask_,
                        new_depth_,
                        reflect_rays_,
                        fresnel_,
                        dir_deltas_,
                    ) = trace_ray(
                        rays_,
                        n=n,
                        reflection=reflection,
                        refraction=refraction,
                    )

            z_coarse = self.sample_coarse(rays)  # (B, Kc)

            # if only reflection, replace rays with reflect_rays
            if reflection_only:
                rays = copy_index(rays, valid_mask, reflect_rays)

            if ray_sampling:
                if not model.viewdir_only:
                    raise NotImplementedError
                else:
                    samp_rays, samp_confidence = self.sample_rays(
                        rays, samp_num=self.cone_samp_num
                    )  # (B, z_coarse, 8)
                    coarse_composite = self.composite(
                        model, rays, z_coarse, coarse=True, sb=superbatch_size
                    )
                    coarse_composite_hit = [
                        torch.zeros(
                            [valid_mask.sum(), z_coarse.shape[1]], device=rays.device
                        ),
                        torch.zeros([valid_mask.sum(), 3], device=rays.device),
                        torch.zeros([valid_mask.sum()], device=rays.device),
                        None,
                    ]
                    # apply confidence coeffident to each sample
                    samp_confidence = samp_confidence.to(device=rays.device)
                    for i in range(self.cone_samp_num):
                        coarse_composite_hit_tmp = self.composite(
                            model,
                            samp_rays[valid_mask, i, :],  # (valid, K, 8)
                            z_coarse[valid_mask, :],
                            coarse=True,
                            sb=superbatch_size,
                        )
                        coarse_composite_hit[0] += (
                            samp_confidence[valid_mask, i].unsqueeze(1)
                            * coarse_composite_hit_tmp[0]
                        )
                        coarse_composite_hit[1] += (
                            samp_confidence[valid_mask, i].unsqueeze(1)
                            * coarse_composite_hit_tmp[1]
                        )
                        coarse_composite_hit[2] += (
                            samp_confidence[valid_mask, i] * coarse_composite_hit_tmp[2]
                        )

                    sum_confidence = torch.sum(
                        samp_confidence[valid_mask], dim=1, dtype=torch.float32
                    )
                    coarse_composite[0][valid_mask] = coarse_composite_hit[
                        0
                    ] / sum_confidence.unsqueeze(
                        1
                    )  # weigths
                    coarse_composite[1][valid_mask] = coarse_composite_hit[
                        1
                    ] / sum_confidence.unsqueeze(
                        1
                    )  # rgb
                    coarse_composite[2][valid_mask] = (
                        coarse_composite_hit[2] / sum_confidence
                    )  # depth
            else:
                coarse_composite = self.composite(
                    model, rays, z_coarse, coarse=True, sb=superbatch_size
                )

            if not reflection_only:
                if reflection:
                    coarse_reflect_composite = self.composite(
                        model,
                        reflect_rays,
                        z_coarse[valid_mask],
                        coarse=True,
                        sb=superbatch_size,
                    )

                if refraction and fresnel is not None:
                    rgb = coarse_composite[1]
                    rgb[valid_mask] = fresnel.unsqueeze(-1) * rgb[valid_mask]
                    if reflection:
                        rgb[valid_mask] = (
                            rgb[valid_mask]
                            + (1 - fresnel.unsqueeze(-1)) * coarse_reflect_composite[1]
                        )
                    coarse_composite = list(coarse_composite)
                    coarse_composite[1] = rgb
                    coarse_composite = tuple(coarse_composite)

            outputs = DotMap(
                coarse=self._format_outputs(
                    coarse_composite,
                    superbatch_size,
                    want_weights=want_weights,
                ),
                dir_deltas=dir_deltas if self.stage == 3 else None,
            )

            if self.using_fine:
                all_samps = [z_coarse]
                if self.n_fine - self.n_fine_depth > 0:
                    all_samps.append(
                        self.sample_fine(rays, coarse_composite[0].detach())
                    )  # (B, Kf - Kfd)
                if self.n_fine_depth > 0:
                    all_samps.append(
                        self.sample_fine_depth(
                            rays, coarse_composite[2], self.n_fine_depth
                        )
                    )  # (B, Kfd)
                z_combine = torch.cat(all_samps, dim=-1)  # (B, Kc + Kf)
                z_combine_sorted, argsort = torch.sort(z_combine, dim=-1)

                if ray_sampling:
                    raise NotImplementedError
                else:
                    fine_composite = self.composite(
                        model,
                        rays,
                        z_combine_sorted,
                        coarse=False,
                        sb=superbatch_size,
                    )

                if not reflection_only:
                    if reflection:
                        fine_reflect_composite = self.composite(
                            model,
                            reflect_rays,
                            z_combine_sorted[valid_mask],
                            coarse=False,
                            sb=superbatch_size,
                        )

                    if refraction and fresnel is not None:
                        rgb = fine_composite[1]
                        rgb[valid_mask] = fresnel.unsqueeze(-1) * rgb[valid_mask]
                        if reflection:
                            rgb[valid_mask] = (
                                rgb[valid_mask]
                                + (1 - fresnel.unsqueeze(-1))
                                * fine_reflect_composite[1]
                            )
                        fine_composite = list(fine_composite)
                        fine_composite[1] = rgb
                        fine_composite = tuple(fine_composite)

                outputs.fine = self._format_outputs(
                    fine_composite,
                    superbatch_size,
                    want_weights=want_weights,
                )
            return outputs

    def _format_outputs(
        self,
        rendered_outputs,
        superbatch_size,
        want_weights=False,
    ):
        weights, rgb, depth, normal = rendered_outputs
        if superbatch_size > 0:
            rgb = rgb.reshape(superbatch_size, -1, 3)
            depth = depth.reshape(superbatch_size, -1)
            weights = weights.reshape(superbatch_size, -1, weights.shape[-1])
            if normal is not None:
                normal = normal.reshape(superbatch_size, -1, 3)
        ret_dict = DotMap(rgb=rgb, depth=depth)
        if want_weights:
            ret_dict.weights = weights
        if normal is not None:
            ret_dict.normal = normal
        return ret_dict

    def sched_step(self, steps=1):
        """
        Called each training iteration to update sample numbers
        according to schedule
        """
        if self.sched is None:
            return
        self.iter_idx += steps
        while (
            self.last_sched.item() < len(self.sched[0])
            and self.iter_idx.item() >= self.sched[0][self.last_sched.item()]
        ):
            self.n_coarse = self.sched[1][self.last_sched.item()]
            self.n_fine = self.sched[2][self.last_sched.item()]
            print(
                "INFO: NeRF sampling resolution changed on schedule ==> c",
                self.n_coarse,
                "f",
                self.n_fine,
            )
            self.last_sched += 1

    def export_mesh(
        self,
        decimate_target=-1,
        global_step=0,
    ):
        if self.use_grid:
            sdf = self.sdf
            deform = torch.tanh(self.deform) / self.tet_grid_size
        else:
            pred = self.sdf_and_deform_mlp(
                self.encoder(self.verts, step_id=global_step)
                if self.use_progressive_encoder
                else self.encoder(self.verts)
            )
            sdf, deform = pred[:, 0], pred[:, 1:]
            deform = torch.tanh(deform) / self.tet_grid_size
        vertices, triangles, uvs, uv_idx = self.dmtet(
            self.verts + deform, sdf, self.indices
        )

        vertices = vertices.detach().cpu().numpy()
        triangles = triangles.detach().cpu().numpy()

        # clean
        vertices = vertices.astype(np.float32)
        triangles = triangles.astype(np.int32)
        vertices, triangles = clean_mesh(
            vertices, triangles, remesh=True, remesh_size=0.01
        )

        # decimation
        if decimate_target > 0 and triangles.shape[0] > decimate_target:
            vertices, triangles = decimate_mesh(vertices, triangles, decimate_target)
        mesh = trimesh.Trimesh(
            vertices, triangles, process=False
        )  # important, process=True leads to seg fault...

        return mesh

    @classmethod
    def from_conf(
        cls,
        conf,
        white_bkgd=False,
        eval_batch_size=100000,
        enable_refr=False,
        enable_refl=False,
        stage=1,
        tet_scale=1.0,
        sphere_radius=1.0,
        ior=1.5,
        use_cone=False,
        use_grid=False,
        use_sdf=False,
        use_progressive_encoder=False,
    ):
        return cls(
            conf.get_int("n_coarse", 128),
            conf.get_int("n_fine", 64),
            n_fine_depth=conf.get_int("n_fine_depth", 32),
            noise_std=conf.get_float("noise_std", 0.0),
            depth_std=conf.get_float("depth_std", 0.01),
            white_bkgd=conf.get_float("white_bkgd", white_bkgd),
            eval_batch_size=conf.get_int("eval_batch_size", eval_batch_size),
            sched=conf.get_list("sched", None),
            enable_refr=enable_refr,
            enable_refl=enable_refl,
            stage=stage,
            tet_scale=tet_scale,
            sphere_radius=sphere_radius,
            ior=ior,
            use_cone=use_cone,
            use_grid=use_grid,
            use_sdf=use_sdf,
            use_progressive_encoder=use_progressive_encoder,
        )

    def bind_parallel(self, net, gpus=None, simple_output=False):
        """
        Returns a wrapper module compatible with DataParallel.
        Specifically, it renders rays with this renderer
        but always using the given network instance.
        Specify a list of GPU ids in 'gpus' to apply DataParallel automatically.
        :param net A PixelNeRF network
        :param gpus list of GPU ids to parallize to. If length is 1,
        does not parallelize
        :param simple_output only returns rendered (rgb, depth) instead of the
        full render output map. Saves data tranfer cost.
        :return torch module
        """
        wrapped = _RenderWrapper(net, self, simple_output=simple_output)
        if gpus is not None and len(gpus) > 1:
            print("Using multi-GPU", gpus)
            wrapped = torch.nn.DataParallel(wrapped, gpus, dim=1)
        return wrapped
