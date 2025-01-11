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
models_dir = os.path.join(os.path.join(comfy_root, "models"))
checkpoints_dir = os.path.join(models_dir, "checkpoints")
outputs_dir = os.path.join(comfy_root, "output")

models = [d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d))]


pipeline_process = None


class AsyncDiffSVDPipelineLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (models,),
                # "scheduler": (
                #     list([
                #         "ddim",
                #         "euler",
                #         "euler_a",
                #         "dpm_2",
                #         "dpm_2_a",
                #         "dpmpp_2m",
                #         "dpmpp_2m_sde",
                #         "dpmpp_sde",
                #         "heun",
                #         "lms",
                #         "pndm",
                #         "unipc",
                #     ]),
                #     {
                #         "default": "dpmpp_2m",
                #     }
                # ),
                "nproc_per_node": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 8192,
                        "step": 1
                    }
                ),
                "model_n": (
                    list([
                        2,
                        3,
                        4
                    ]),
                    {
                        "default": 2,
                    }
                ),
                "stride": (
                    list([
                        1,
                        2
                    ]),
                    {
                        "default": 1,
                    }
                ),
                "time_shift": (
                    "BOOLEAN",
                    {
                        "default": False,
                    }
                ),
                "variant": (
                    list([
                        "bf16",
                        "fp16",
                        "fp32"
                    ]),
                    {
                        "default": "fp16",
                    }
                ),
            }
        }

    RETURN_TYPES = ("ASYNCDIFF_PIPELINE",)
    FUNCTION = "launch_host"
    CATEGORY = "AsyncDiff"        

    def launch_host(self, model, nproc_per_node, model_n, stride, time_shift, variant, scheduler=None):
        cmd = [
            'torchrun',
            f'--nproc_per_node={nproc_per_node}',
            f'{cwd}/host.py',

            f'--model={checkpoints_dir}/{model}',
            f'--pipeline_type=svd',
            f'--model_n={model_n}',
            f'--stride={stride}',
            f'--time_shift={time_shift}',
            f'--variant={variant}',
            # f'--scheduler={scheduler}',
        ]

        global pipeline_process
        pipeline_process = subprocess.Popen(cmd)
        host = 'http://localhost:6000'
        while True:
            try:
                response = requests.get(f'{host}/initialize')
                if response.status_code == 200 and response.json().get("status") == "initialized":
                    print("\n###### Pipeline is ready ######\n")
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)

        return (f"{host}/generate", )


class AsyncDiffImg2VidSampler:
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
                "motion_bucket_id": (
                    "INT",
                    {
                        "default": 180,
                        "min": 1,
                        "max": 1024
                    }
                ),
                "noise_aug_strength": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0,
                        "max": 1024,
                        "step": 0.01
                    }
                ),
                "shutdown_pipeline": (
                    "BOOLEAN",
                    {
                        "default": True,
                    }
                ),
            }
        }


    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "AsyncDiff"

    def generate(
        self,
        pipeline, image, width, height, steps, seed, decode_chunk_size,
        num_frames, motion_bucket_id, noise_aug_strength, shutdown_pipeline
    ):
        global pipeline_process
        if pipeline_process is None:
            return (None,)
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
            "motion_bucket_id": motion_bucket_id,
            "noise_aug_strength": noise_aug_strength,
            "output_path": outputs_dir,
        }
        response = requests.post(url, json=data)
        response_data = response.json()
        output_base64 = response_data.get("output")
        if shutdown_pipeline:
            pipeline_process.terminate()
            pipeline_process = None
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
    "AsyncDiffImg2VidSampler": AsyncDiffImg2VidSampler,
    "AsyncDiffSVDPipelineLoader": AsyncDiffSVDPipelineLoader,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "AsyncDiffImg2VidSampler": "AsyncDiffImg2VidSampler",
    "AsyncDiffSVDPipelineLoader": "AsyncDiffSVDPipelineLoader",
}
