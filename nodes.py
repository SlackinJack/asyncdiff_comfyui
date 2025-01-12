#!/usr/bin/env python3
import base64
import json
import os
import pickle
import requests
import subprocess
import time
import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor


cwd = os.path.dirname(__file__)
comfy_root = os.path.dirname(os.path.dirname(cwd))
models_dir = os.path.join(os.path.join(comfy_root, "models"))
checkpoints_dir = os.path.join(models_dir, "checkpoints")
outputs_dir = os.path.join(comfy_root, "output")
models = [d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d))]


host_address = 'http://localhost:6000'
host_process = None
host_address_generate = f'{host_address}/generate'
host_address_initialize = f'{host_address}/initialize'


class ADPipelineConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (models,),
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
                "scale_input": (
                    "BOOLEAN",
                    {
                        "default": False,
                    }
                ),
                "scale_percentage": (
                    "FLOAT",
                    {
                        "default": 75.0,
                        "min": 0.1,
                        "max": 200.0,
                        "step": 0.1
                    }
                ),
                "enable_model_cpu_offload": (
                    "BOOLEAN",
                    {
                        "default": False,
                    }
                ),
                "enable_sequential_cpu_offload": (
                    "BOOLEAN",
                    {
                        "default": False,
                    }
                ),
                "enable_tiling": (
                    "BOOLEAN",
                    {
                        "default": False,
                    }
                ),
                "enable_slicing": (
                    "BOOLEAN",
                    {
                        "default": False,
                    }
                ),
                "xformers_efficient": (
                    "BOOLEAN",
                    {
                        "default": False,
                    }
                ),
            }
        }

    RETURN_TYPES = ("AD_CONFIG",)
    FUNCTION = "get_config"
    CATEGORY = "AsyncDiff"        

    def get_config(self, model, nproc_per_node, model_n, stride, time_shift, variant, scale_input, scale_percentage,
        enable_model_cpu_offload, enable_sequential_cpu_offload, enable_tiling, enable_slicing, xformers_efficient
    ):
        return (
            {
                "model": model,
                "nproc_per_node": nproc_per_node,
                "model_n": model_n,
                "stride": stride,
                "time_shift": time_shift,
                "variant": variant,
                "scale_input": scale_input,
                "scale_percentage": scale_percentage,
                "enable_model_cpu_offload": enable_model_cpu_offload,
                "enable_sequential_cpu_offload": enable_sequential_cpu_offload,
                "enable_tiling": enable_tiling,
                "enable_slicing": enable_slicing,
                "xformers_efficient": xformers_efficient,
            },
        )


class ADSVDSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "config": ("AD_CONFIG",),
                "image": ("IMAGE",),
                "width": (
                    "INT",
                    {
                        "default": 512,
                        "min": 8,
                        "max": 32768,
                        "step": 8
                    }
                ),
                "height": (
                    "INT",
                    {
                        "default": 512,
                        "min": 8,
                        "max": 32768,
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
                        "default": 40,
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
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "AsyncDiff"

    def generate(
        self,
        width, height, steps, seed, decode_chunk_size, num_frames,
        motion_bucket_id, noise_aug_strength, image, config
    ):
        config["pipeline_type"] = "svd"
        launch_host_process(config)
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
        response = requests.post(host_address_generate, json=data)
        response_data = response.json()
        output_base64 = response_data.get("output")
        close_host_process()
        if output_base64:
            print("Media generated")
            output_bytes = base64.b64decode(output_base64)
            images = pickle.loads(output_bytes)
            tensors = []
            for i in images:
                tensor = ToTensor()(i)              # CHW
                tensor = tensor.permute(1, 2, 0)    # CHW -> HWC
                tensors.append(tensor)
            return (torch.stack(tuple(tensors)),)   # HWC -> NHWC
        else:
            assert False, "No media generated"


class ADUpscaleSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "config": ("AD_CONFIG",),
                "image": ("IMAGE",),
                "positive_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": ""
                    }
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": ""
                    }
                ),
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
                        "default": 60,
                        "min": 1,
                        "max": 1024
                    }
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "AsyncDiff"

    def generate(self, positive_prompt, negative_prompt, steps, seed, image, config):
        config["pipeline_type"] = "sdup"
        # config["scheduler"] = scheduler
        if image.size(0) > 1:
            images = list(torch.unbind(image, 0))   # NHWC -> [HWC], len == N
        else:
            images = [image.squeeze(0)]             # NHWC -> [HWC], len == 1
        tensors = []
        i = 0
        launch_host_process(config)
        for im in images:
            i += 1
            im = im.permute(2, 0, 1)                # HWC -> CHW
            im = ToPILImage()(im)
            pickled_image = pickle.dumps(im)
            b64_image = base64.b64encode(pickled_image).decode("utf-8")
            data = {
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "image": b64_image,
                "num_inference_steps": steps,
                "seed": seed,
                "output_path": outputs_dir,
            }
            print(f"Upscaling image: {i}/{len(images)}")
            response = requests.post(host_address_generate, json=data)
            response_data = response.json()
            output_base64 = response_data.get("output")
            if output_base64:
                print(f"Finished upscaling image: {i}/{len(images)}")
                output_bytes = base64.b64decode(output_base64)
                im2 = pickle.loads(output_bytes)
                tensor_image = ToTensor()(im2)                  # CHW
                tensor_image = tensor_image.permute(1, 2, 0)    # CHW -> HWC
                tensors.append(tensor_image)
            else:
                if len(images) == 1:
                    assert False, "No media generated"
                else:
                    print("Error processing an image.")
        close_host_process()
        return (torch.stack(tuple(tensors)),)                   # HWC -> NHWC


def launch_host_process(config):
    global host_process

    if host_process is not None:
        close_host_process()

    cmd = [
        'torchrun',
        f'--nproc_per_node={config.get("nproc_per_node")}',
        f'{cwd}/host.py',

        '--host_mode=comfyui',
        f'--model={checkpoints_dir}/{config.get("model")}',
        f'--pipeline_type={config.get("pipeline_type")}',
        f'--model_n={config.get("model_n")}',
        f'--stride={config.get("stride")}',
        f'--time_shift={config.get("time_shift")}',
        f'--variant={config.get("variant")}',
        # f'--scheduler={config.get("scheduler")}',
    ]

    if config.get("scale_input"):
        cmd.append('--scale_input')
        cmd.append(f'--scale_percentage={config.get("scale_percentage")}')

    if config.get("enable_model_cpu_offload"):
        cmd.append('--enable_model_cpu_offload')

    if config.get("enable_sequential_cpu_offload"):
        cmd.append('--enable_sequential_cpu_offload')

    if config.get("enable_tiling"):
        cmd.append('--enable_tiling')

    if config.get("enable_slicing"):
        cmd.append('--enable_slicing')

    if config.get("xformers_efficient"):
        cmd.append('--xformers_efficient')

    host_process = subprocess.Popen(cmd)
    
    connection_attempts = 0
    while True:
        try:
            response = requests.get(host_address_initialize)
            if response.status_code == 200 and response.json().get("status") == "initialized":
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
        connection_attempts += 1
        if connection_attempts >= 60:
            assert False, "Failed to launch host. Check logs for details."


def close_host_process():
    global host_process
    if host_process is not None:
        host_process.terminate()
        host_process = None
    return


NODE_CLASS_MAPPINGS = {
    "ADPipelineConfig": ADPipelineConfig,
    "ADSVDSampler": ADSVDSampler,
    "ADUpscaleSampler": ADUpscaleSampler,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "ADPipelineConfig": "ADPipelineConfig",
    "ADSVDSampler": "ADSVDSampler",
    "ADUpscaleSampler": "ADUpscaleSampler",
}
