import torch
import numpy as np
import tinycudann as tcnn


# modified from ngp in https://github.com/KAIR-BAIR/nerfacc/blob/433130618da036d64581e07dc1bf5520bd213129/examples/radiance_fields/ngp.py#L83
class SphereNGPRadianceField(torch.nn.Module):
    """Instance-NGP Radiance Field variation, in which grid is conditioned on a sphere."""

    def __init__(self) -> None:
        super().__init__()
        self.viewdir_only = True
        self.use_viewdirs = True
        self.use_decoder = False
        self.use_xyz_coords = True
        self.num_dim = 3 if self.use_xyz_coords else 2
        self.base_resolution = 8
        self.max_resolution = 8192
        self.geo_feat_dim = 15 if self.use_decoder else 3
        self.n_levels = 17
        self.log2_hashmap_size = 21

        per_level_scale = np.exp(
            (np.log(self.max_resolution) - np.log(self.base_resolution))
            / (self.n_levels - 1)
        ).tolist()

        if self.viewdir_only:
            self.mlp_base = tcnn.NetworkWithInputEncoding(
                n_input_dims=self.num_dim,
                n_output_dims=self.geo_feat_dim + 1,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": self.n_levels,
                    "n_features_per_level": 4,
                    "log2_hashmap_size": self.log2_hashmap_size,
                    "base_resolution": self.base_resolution,
                    "per_level_scale": per_level_scale,
                    "interpolation": "Linear",  # todo
                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )
        else:
            self.pos_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "Frequency",
                    "n_frequencies": 8,
                },
            )
            self.dir_encoding = tcnn.Encoding(
                n_input_dims=self.num_dim,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": self.n_levels,
                    "n_features_per_level": 4,
                    "log2_hashmap_size": self.log2_hashmap_size,
                    "base_resolution": self.base_resolution,
                    "per_level_scale": per_level_scale,
                    "interpolation": "Linear",
                },
            )
            self.mlp = tcnn.Network(
                n_input_dims=self.pos_encoding.n_output_dims
                + self.dir_encoding.n_output_dims,
                n_output_dims=self.geo_feat_dim + 1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )

        if self.use_decoder:
            self.mlp_head = tcnn.Network(
                n_input_dims=self.geo_feat_dim,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )

    def forward(
        self,
        xyz=None,
        coarse=True,
        viewdirs=None,
    ):
        SB, B, _ = viewdirs.shape
        viewdirs = viewdirs.reshape(-1, 3)
        if self.use_xyz_coords:
            positions = viewdirs
        else:
            theta = torch.acos(viewdirs[:, 2])
            phi = torch.atan2(viewdirs[:, 1], viewdirs[:, 0])
            theta_norm, phi_norm = theta / torch.pi, (phi + torch.pi) / (2 * torch.pi)
            theta_norm, phi_norm = theta, phi
            positions = torch.stack((theta_norm, phi_norm), dim=-1)
        if self.viewdir_only:
            positions = (
                self.mlp_base(positions)
                .view(list(positions.shape[:-1]) + [1 + self.geo_feat_dim])
                .to(positions)
            )
        else:
            positions = self.pos_encoding(positions)
            positions = torch.cat((positions, self.dir_encoding(viewdirs)), dim=-1)
            positions = (
                self.mlp(positions)
                .view(list(positions.shape[:-1]) + [1 + self.geo_feat_dim])
                .to(positions)
            )
        density_before_activation, embedding = torch.split(
            positions, [1, self.geo_feat_dim], dim=-1
        )
        if self.use_decoder:
            h = embedding.reshape(-1, self.geo_feat_dim)
            rgb = (
                self.mlp_head(h).reshape(list(embedding.shape[:-1]) + [3]).to(embedding)
            )
        else:
            rgb = embedding
        rgb = torch.sigmoid(rgb)
        output_list = [rgb, density_before_activation]
        output = torch.cat(output_list, dim=-1)
        output = output.reshape(SB, B, -1)
        return output
