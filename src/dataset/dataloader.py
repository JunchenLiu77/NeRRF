import imageio
import numpy as np
import cv2
import json
import math
import os
import sys
from os.path import join
from PIL import Image

sys.path.append("..")
import dataset.utils as api_utils
import torch
from torchvision import transforms
from PIL import Image
from scipy import ndimage


def get_image_to_tensor_balanced(image_size):
    ops = []
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    ops.append(transforms.Resize(image_size))
    return transforms.Compose(ops)


def get_focal(meta_path, image_size, dataset_type):
    W, H = image_size
    camera_pose = np.eye(4, dtype=np.float32)
    with open(meta_path) as f:
        meta_data = json.load(f)
        camera_pose = np.array(meta_data["cam_world_pose"])
        if dataset_type == "blender":
            dx = math.radians(60)
            fx = (W / 2) / math.tan(dx / 2)
            fy = fx
        elif dataset_type == "eikonal":
            fx = fy = meta_data["f"]
        else:
            raise NotImplementedError
    camera_pose[..., 3] = camera_pose[..., 3]
    return fx, fy, camera_pose


def get_mvp(focal, pose, image_size, far=50, near=0.001):
    W, H = image_size
    projection = np.array(
        [
            [2 * focal / W, 0, 0, 0],
            [0, -2 * focal / H, 0, 0],
            [0, 0, -(far + near) / (far - near), -(2 * far * near) / (far - near)],
            [0, 0, -1, 0],
        ],
        dtype=np.float32,
    )
    return projection @ np.linalg.inv(pose)  # [4, 4]


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        stage="train",
        dataset_type="blender",  # if use eikonal dataset, change this to "eikonal"
    ):
        super().__init__()
        self.split = stage
        self.type = dataset_type
        if self.type == "blender":
            self.image_size = (480, 270)
            self.z_near, self.z_far = 0.02, 80
            self.image_dir = data_dir
            self.depth_dir = data_dir + "/../../depth/" + os.path.basename(data_dir)
            self.mask_dir = data_dir + "/../../mask/" + os.path.basename(data_dir)
            self.meta_dir = data_dir + "/../../meta"
            if self.split == "train":
                self.image_list = [str(2 * d) for d in range(49,50)] #50
            else:
                self.image_list = [str(2 * d + 1) for d in range(1)] #39
        elif self.type == "eikonal":
            self.image_size = (672, 504)
            self.z_near, self.z_far = 0.02, 3
            self.image_dir = data_dir + "/images"
            self.mask_dir = data_dir + "/mask"
            self.meta_dir = data_dir + "/meta"
            file_names = os.listdir(self.mask_dir)
            if self.split == "train":
                self.image_list = [name[5:-4] for name in file_names][:50]
            else:
                self.image_list = [name[5:-4] for name in file_names][50:89]
        else:
            raise NotImplementedError
        self.image2tensor = get_image_to_tensor_balanced(
            (self.image_size[1], self.image_size[0])
        )
        name, focal, poses, images, mvps, masks = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for i in range(len(self.image_list)):
            result = self.__get_single_item__(i)
            name.append(result["name"])
            focal.append(result["focal"])
            poses.append(result["poses"])
            images.append(result["images"])
            mvps.append(result["mvp"])
            masks.append(result["mask"])

        results = {
            "name": name,
            "focal": torch.stack(focal),
            "poses": torch.stack(poses),
            "images": torch.stack(images),
            "mvp": torch.stack(mvps),
            "mask": torch.stack(masks),
        }
        self.results = results

    def __len__(self):
        return len(self.image_list)

    def __get_single_item__(self, index):
        name = self.image_list[index]
        if self.type == "blender":
            img_name = name + "_0001.png"
            meta_name = name + "_meta_0000.json"

            mask_name = "mask_" + name + "_0001.png"
            mask_path = join(self.mask_dir, mask_name)
            mask = imageio.imread(mask_path)
            resized_mask = np.where(mask == 255, 1, mask)
            resized_mask = ndimage.zoom(resized_mask, (272 / resized_mask.shape[0], 480 / resized_mask.shape[1]))
            mask = torch.tensor(resized_mask)

            # depth_name = name + "_depth_0001.exr"
            # depth_path = join(self.depth_dir, depth_name)
            # depth = api_utils.exr_loader(depth_path, ndim=1)
            # dd = cv2.resize(depth, dsize=(480, 272))
            # mask = dd < 100

            mask = torch.tensor(mask).unsqueeze(0)
        elif self.type == "eikonal":
            img_name = name + ".JPG"
            meta_name = name + ".json"
            mask_path = join(
                self.mask_dir,
                "mask_" + name + ".png",
            )
            image = Image.open(mask_path)
            density = np.array(image)[:, :, 0]
            mask = density > 0.5
            image = Image.fromarray(mask.astype("uint8") * 255)
            image.save("test_mask.png")
            mask = torch.tensor(mask).unsqueeze(0)
        else:
            raise NotImplementedError

        image_path = join(self.image_dir, img_name)
        img = imageio.imread(image_path)[..., :3]
        img = self.image2tensor(img)

        meta_path = join(self.meta_dir, meta_name)
        fx, fy, camera_pose = get_focal(meta_path, self.image_size, self.type)
        mvp = get_mvp(fx, camera_pose, self.image_size)
        result = {
            "name": name,
            "focal": torch.tensor((fx, fy), dtype=torch.float32),
            "poses": torch.tensor(camera_pose, dtype=torch.float32),
            "images": img,
            "mvp": torch.tensor(mvp, dtype=torch.float32),
            "mask": mask,
        }
        return result

    def __getitem__(self, index):
        result = {
            "name": self.results["name"][index],
            "focal": self.results["focal"][index],
            "poses": self.results["poses"][index],
            "images": self.results["images"][index],
            "mask": self.results["mask"][index],
            "mvp": self.results["mvp"][index],
        }
        return result
