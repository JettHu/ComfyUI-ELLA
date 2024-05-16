from contextlib import suppress

import torch
from comfy import model_management


def escape_important(text):
    return text.replace("\\)", "\0\1").replace("\\(", "\0\2")


def unescape_important(text):
    return text.replace("\0\1", ")").replace("\0\2", "(")


def parse_parentheses(string):
    result = []
    current_item = ""
    nesting_level = 0
    for char in string:
        if char == "(":
            if nesting_level == 0:
                if current_item:
                    result.append(current_item)
                    current_item = "("
                else:
                    current_item = "("
            else:
                current_item += char
            nesting_level += 1
        elif char == ")":
            nesting_level -= 1
            if nesting_level == 0:
                result.append(current_item + ")")
                current_item = ""
            else:
                current_item += char
        else:
            current_item += char
    if current_item:
        result.append(current_item)
    return result


def _remove_weights(string):
    a = parse_parentheses(string)
    out = []
    for x in a:
        if len(x) >= 2 and x[-1] == ")" and x[0] == "(":
            x = x[1:-1]
            xx = x.rfind(":")
            if xx > 0:
                with suppress(Exception):
                    x = x[:xx]
            out += _remove_weights(x)
        else:
            out += [x]
    return out


def remove_weights(text: str):
    text = escape_important(text)
    parsed_weights = _remove_weights(text)
    return "".join([unescape_important(segment) for segment in parsed_weights])


class Output:
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, item):
        setattr(self, key, item)


def clip_resize(image, size=224):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=image.device, dtype=image.dtype)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=image.device, dtype=image.dtype)
    image = image.movedim(-1, 1)
    image = torch.nn.functional.interpolate(image, size=(size, size), mode="bicubic", antialias=True)
    image = torch.clip((255.0 * image), 0, 255).round() / 255.0
    return (image - mean.view([3, 1, 1])) / std.view([3, 1, 1])


def clip_vision_encode(clip_vision, image, crop=False):
    # resize + center crop 224
    if crop:
        return clip_vision.encode_image(image)
    # resize to 224
    model_management.load_model_gpu(clip_vision.patcher)
    pixel_values = clip_resize(image.to(clip_vision.load_device)).float()
    out = clip_vision.model(pixel_values=pixel_values, intermediate_output=-2)
    outputs = Output()
    outputs["last_hidden_state"] = out[0].to(model_management.intermediate_device())
    outputs["image_embeds"] = out[2].to(model_management.intermediate_device())
    outputs["penultimate_hidden_states"] = out[1].to(model_management.intermediate_device())
    return outputs
