from typing import Sequence, Union

import numpy as np
import PIL.Image
import torch
from safetensors.torch import load_file

from .net import build_model

adaface_models = {
    "adaface_ir50_webface4m": "./models/adaface_ir50_webface4m.safetensors",
}


def load_pretrained_model(architecture, pretrained_path):
    model = build_model(architecture)
    model.load_state_dict(load_file(pretrained_path))
    model.eval()
    return model


class AdaFace:
    def __init__(self, pretrained_path, device="cuda", architecture="ir_50"):
        self.model = load_pretrained_model(architecture, pretrained_path)
        self.model.to(device)
        self.device = device

    @staticmethod
    def to_input(pil_rgb_image: PIL.Image.Image):
        np_img = np.array(pil_rgb_image.resize(size=(112, 112), resample=PIL.Image.LANCZOS))
        brg_img = ((np_img[:, :, ::-1] / 255.0) - 0.5) / 0.5
        return torch.from_numpy(brg_img.transpose(2, 0, 1)).float()

    @torch.inference_mode()
    def __call__(
        self, pil_images: Union[PIL.Image.Image, Sequence[PIL.Image.Image]], bgr_image_tensor=None, norm=True, **kwargs
    ):
        if bgr_image_tensor is None:
            pil_images = [pil_images] if isinstance(pil_images, PIL.Image.Image) else pil_images
            bgr_image_tensor = torch.stack([self.to_input(pil_image) for pil_image in pil_images]).to(self.device)

        return self.model(bgr_image_tensor, norm=norm, **kwargs)

def adaface_ir50_webface4m(pretrained_path=adaface_models["adaface_ir50_webface4m"], device="cuda"):
    return AdaFace(pretrained_path, device=device, architecture="ir_50")


class AdaFaceEmbedder:
    def __init__(self, path: str = "/mnt/share/cq8/public/adaface/adaface_ir50_webface4m.safetensors", norm=True):
        self.model = adaface_ir50_webface4m(path, device="cuda")

    def forward(self, image):
        embeddings = self.model(image, disable_output_layer=True)
        return embeddings.view(-1, 512, 49).permute(0, 2, 1)
