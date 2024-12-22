#!/usr/bin/env python3
import base64
import json
import numpy
import os
import pickle
import requests
import subprocess
import time
import torch
import torchvision.transforms.functional as F
from pathlib import Path
from PIL import Image, ImageSequence
from torchvision.io import decode_image
from torchvision.transforms import ToPILImage, ToTensor


cwd = os.path.dirname(__file__)
comfy_root = os.path.dirname(os.path.dirname(cwd))
checkpoints_dir = os.path.join(os.path.join(comfy_root, "models"), "checkpoints")
outputs_dir = os.path.join(comfy_root, "output")


class AsyncDiffSVDPipelineLoader:
    @classmethod
    def INPUT_TYPES(s):
        models = [d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d))]
        return {
            "required": {
                "model": (models,), 
                "pipeline_type": (
                    list([
                        "svd",
                    ]),
                    {
                        "default": "svd",
                    }
                ),
                "nproc_per_node": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 8192,
                        "step": 1
                    }
                ),
            }
        }

    RETURN_TYPES = ("ASYNCDIFF_PIPELINE",)
    FUNCTION = "launch_host"
    CATEGORY = "AsyncDiff"        

    def launch_host(self, model, pipeline_type, nproc_per_node):
        cmd = [
            'torchrun',
            f'--nproc_per_node={nproc_per_node}',
            f'{cwd}/host.py',

            f'--model={checkpoints_dir}/{model}',
            f'--pipeline_type={pipeline_type}',
            # f'--scheduler={scheduler}',
        ]

        process = subprocess.Popen(cmd)
        host = 'http://localhost:6000'
        while True:
            try:
                response = requests.get(f'{host}/initialize')
                if response.status_code == 200 and response.json().get("status") == "initialized":
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)

        return (f"{host}/generate", )


class AsyncDiffSVDSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("ASYNCDIFF_PIPELINE",),
                "image": ("IMAGE",),
                "width": (
                    "INT",
                    {
                        "default": 512,
                        "min": 0,
                        "max": 8192,
                        "step": 8
                    }
                ),
                "height": (
                    "INT",
                    {
                        "default": 512,
                        "min": 0,
                        "max": 8192,
                        "step": 8
                    }
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2 ** 32 - 1
                    }
                ),
                "steps": (
                    "INT",
                    {
                        "default": 30,
                        "min": 1,
                        "max": 1024
                    }
                ),
                "decode_chunk_size": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 1024
                    }
                ),
                "num_frames": (
                    "INT",
                    {
                        "default": 25,
                        "min": 1,
                        "max": 1024
                    }
                ),
            }
        }


    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "AsyncDiff"

    def generate(self, pipeline, image, width, height, steps, seed, decode_chunk_size, num_frames):
        url = pipeline
        image = image.squeeze(0)        # NHWC -> HWC
        image = image.permute(2, 0, 1)  # HWC -> CHW
        image = ToPILImage()(image)
        pickled_image = pickle.dumps(image)
        b64_image = base64.b64encode(pickled_image).decode("utf-8")
        data = {
            "image": b64_image,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "seed": seed,
            "decode_chunk_size": decode_chunk_size,
            "num_frames": num_frames,
            "output_path": outputs_dir,
        }
        response = requests.post(url, json=data)
        response_data = response.json()
        output_base64 = response_data.get("output")
        if output_base64:
            print("Media generated")
            output_bytes = base64.b64decode(output_base64)
            images = pickle.loads(output_bytes)
            images = images.frames[0]
            tensors = []
            for i in images:
                tensor = ToTensor()(i)              # CHW
                tensor = tensor.permute(1, 2, 0)    # CHW -> HWC
                tensors.append(tensor)
            return (torch.stack(tuple(tensors)),)   # HWC -> NHWC
        else:
            print("No media generated")
            return (None,)


NODE_CLASS_MAPPINGS = {
    "AsyncDiffSVDPipelineLoader": AsyncDiffSVDPipelineLoader,
    "AsyncDiffSVDSampler": AsyncDiffSVDSampler,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "AsyncDiffSVDPipelineLoader": "AsyncDiffSVDPipelineLoader",
    "AsyncDiffSVDSampler": "AsyncDiffSVDSampler",
}
