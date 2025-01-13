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


host_address = 'http://localhost:6000'
host_process = None
host_address_generate = f'{host_address}/generate'
host_address_initialize = f'{host_address}/initialize'


INT_MAX = 2 ** 32 - 1
BOOLEAN_DEFAULT_FALSE = ("BOOLEAN", { "default": False })


SCHEDULERS = (
    list(["ddim", "euler", "euler_a", "dpm_2", "dpm_2_a", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_sde", "heun", "lms", "pndm", "unipc"]),
    { "default": "dpmpp_2m" }
)
VARIANT =           (["bf16", "fp16", "fp32"],  { "default": "fp16" })
MODEL_N =           ([2, 3, 4],                 { "default": 2 })
STRIDE =            ([1, 2],                    { "default": 1 })
NPROC_PER_NODE =    ("INT",                     { "default": 2,     "min": 1,       "max": INT_MAX, "step": 1 })
SCALE_PERCENTAGE =  ("FLOAT",                   { "default": 75.0,  "min": 0.01,    "max": INT_MAX, "step": 0.01 })


PROMPT =                ("STRING",  { "default": "",    "multiline": True })
RESOLUTION =            ("INT",     { "default": 512,   "min": 8,   "max": INT_MAX, "step": 8 })
SEED =                  ("INT",     { "default": 0,     "min": 0,   "max": INT_MAX, "step": 1 })
STEPS =                 ("INT",     { "default": 60,    "min": 1,   "max": INT_MAX, "step": 1 })
DECODE_CHUNK_SIZE =     ("INT",     { "default": 8,     "min": 1,   "max": INT_MAX, "step": 1 })
NUM_FRAMES =            ("INT",     { "default": 25,    "min": 1,   "max": INT_MAX, "step": 1 })
MOTION_BUCKET_ID =      ("INT",     { "default": 180,   "min": 1,   "max": INT_MAX, "step": 1 })
CFG =                   ("FLOAT",   { "default": 3.5,   "min": 0,   "max": INT_MAX, "step": 0.1 })
NOISE_AUG_STRENGTH =    ("FLOAT",   { "default": 0.01,  "min": 0,   "max": INT_MAX, "step": 0.01 })


cwd = os.path.dirname(__file__)
comfy_root = os.path.dirname(os.path.dirname(cwd))
models_dir = os.path.join(os.path.join(comfy_root, "models"))
checkpoints_dir = os.path.join(models_dir, "checkpoints")
outputs_dir = os.path.join(comfy_root, "output")
models = [d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d))]


class ADPipelineConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model":                            (models,),
                "nproc_per_node":                   NPROC_PER_NODE,
                "model_n":                          MODEL_N,
                "stride":                           STRIDE,
                "time_shift":                       BOOLEAN_DEFAULT_FALSE,
                "variant":                          VARIANT,
                "scale_input":                      BOOLEAN_DEFAULT_FALSE,
                "scale_percentage":                 SCALE_PERCENTAGE,
              # "enable_model_cpu_offload":         BOOLEAN_DEFAULT_FALSE,
              # "enable_sequential_cpu_offload":    BOOLEAN_DEFAULT_FALSE,
                "enable_tiling":                    BOOLEAN_DEFAULT_FALSE,
                "enable_slicing":                   BOOLEAN_DEFAULT_FALSE,
                "xformers_efficient":               BOOLEAN_DEFAULT_FALSE,
            }
        }

    RETURN_TYPES = ("AD_CONFIG",)
    FUNCTION = "get_config"
    CATEGORY = "AsyncDiff"        

    def get_config(self, model, nproc_per_node, model_n, stride, time_shift, variant, scale_input, scale_percentage,
        enable_tiling, enable_slicing, xformers_efficient, enable_model_cpu_offload=False, enable_sequential_cpu_offload=False
    ):
        return (
            {
                "model":                            model,
                "nproc_per_node":                   nproc_per_node,
                "model_n":                          model_n,
                "stride":                           stride,
                "time_shift":                       time_shift,
                "variant":                          variant,
                "scale_input":                      scale_input,
                "scale_percentage":                 scale_percentage,
                "enable_model_cpu_offload":         enable_model_cpu_offload,
                "enable_sequential_cpu_offload":    enable_sequential_cpu_offload,
                "enable_tiling":                    enable_tiling,
                "enable_slicing":                   enable_slicing,
                "xformers_efficient":               xformers_efficient,
            },
        )


class ADSD1Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "config":           ("AD_CONFIG",),
                "positive_prompt":  PROMPT,
                "negative_prompt":  PROMPT,
                "width":            RESOLUTION,
                "height":           RESOLUTION,
                "scheduler":        SCHEDULERS,
                "seed":             SEED,
                "steps":            STEPS,
                "guidance_scale":   CFG,
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "AsyncDiff"

    def generate(self, positive_prompt, negative_prompt, width, height, scheduler, steps, guidance_scale, seed, config):
        config["pipeline_type"] = "sd1"
        config["scheduler"] = scheduler
        launch_host_process(config)
        data = {
            "positive_prompt":      positive_prompt,
            "negative_prompt":      negative_prompt,
            "width":                width,
            "height":               height,
            "num_inference_steps":  steps,
            "guidance_scale":       guidance_scale,
            "seed":                 seed,
            "output_path":          outputs_dir,
        }
        response = requests.post(host_address_generate, json=data)
        response_data = response.json()
        output_base64 = response_data.get("output")
        close_host_process()
        if output_base64:
            output_bytes = base64.b64decode(output_base64)
            im2 = pickle.loads(output_bytes)
            tensor_image = ToTensor()(im2)                      # CHW
            tensor_image = tensor_image.unsqueeze(0)            # CHW -> NCHW
            tensor_image = tensor_image.permute(0, 2, 3, 1)     # NCHW -> NHWC
            return (tensor_image,)
        else:
            assert False, "No media generated"


class ADSD2Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "config":           ("AD_CONFIG",),
                "positive_prompt":  PROMPT,
                "negative_prompt":  PROMPT,
                "width":            RESOLUTION,
                "height":           RESOLUTION,
                "scheduler":        SCHEDULERS,
                "seed":             SEED,
                "steps":            STEPS,
                "guidance_scale":   CFG,
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "AsyncDiff"

    def generate(self, positive_prompt, negative_prompt, width, height, scheduler, steps, guidance_scale, seed, config):
        config["pipeline_type"] = "sd2"
        config["scheduler"] = scheduler
        launch_host_process(config)
        data = {
            "positive_prompt":      positive_prompt,
            "negative_prompt":      negative_prompt,
            "width":                width,
            "height":               height,
            "num_inference_steps":  steps,
            "guidance_scale":       guidance_scale,
            "seed":                 seed,
            "output_path":          outputs_dir,
        }
        response = requests.post(host_address_generate, json=data)
        response_data = response.json()
        output_base64 = response_data.get("output")
        close_host_process()
        if output_base64:
            output_bytes = base64.b64decode(output_base64)
            im2 = pickle.loads(output_bytes)
            tensor_image = ToTensor()(im2)                      # CHW
            tensor_image = tensor_image.unsqueeze(0)            # CHW -> NCHW
            tensor_image = tensor_image.permute(0, 2, 3, 1)     # NCHW -> NHWC
            return (tensor_image,)
        else:
            assert False, "No media generated"


class ADSD3Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "config":           ("AD_CONFIG",),
                "positive_prompt":  PROMPT,
                "negative_prompt":  PROMPT,
                "width":            RESOLUTION,
                "height":           RESOLUTION,
                "scheduler":        SCHEDULERS,
                "seed":             SEED,
                "steps":            STEPS,
                "guidance_scale":   CFG,
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "AsyncDiff"

    def generate(self, positive_prompt, negative_prompt, width, height, scheduler, steps, guidance_scale, seed, config):
        config["pipeline_type"] = "sd3"
        config["scheduler"] = scheduler
        launch_host_process(config)
        data = {
            "positive_prompt":  positive_prompt,
            "negative_prompt":  negative_prompt,
            "width":            width,
            "height":           height,
            "guidance_scale":   guidance_scale,
            "seed":             seed,
            "output_path":      outputs_dir,
        }
        response = requests.post(host_address_generate, json=data)
        response_data = response.json()
        output_base64 = response_data.get("output")
        close_host_process()
        if output_base64:
            output_bytes = base64.b64decode(output_base64)
            im2 = pickle.loads(output_bytes)
            tensor_image = ToTensor()(im2)                      # CHW
            tensor_image = tensor_image.unsqueeze(0)            # CHW -> NCHW
            tensor_image = tensor_image.permute(0, 2, 3, 1)     # NCHW -> NHWC
            return (tensor_image,)
        else:
            assert False, "No media generated"


class ADSDXLSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "config":           ("AD_CONFIG",),
                "positive_prompt":  PROMPT,
                "negative_prompt":  PROMPT,
                "width":            RESOLUTION,
                "height":           RESOLUTION,
                "scheduler":        SCHEDULERS,
                "seed":             SEED,
                "steps":            STEPS,
                "guidance_scale":   CFG,
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "AsyncDiff"

    def generate(self, positive_prompt, negative_prompt, width, height, scheduler, steps, guidance_scale, seed, config):
        config["pipeline_type"] = "sdxl"
        config["scheduler"] = scheduler
        launch_host_process(config)
        data = {
            "positive_prompt":      positive_prompt,
            "negative_prompt":      negative_prompt,
            "width":                width,
            "height":               height,
            "num_inference_steps":  steps,
            "guidance_scale":       guidance_scale,
            "seed":                 seed,
            "output_path":          outputs_dir,
        }
        response = requests.post(host_address_generate, json=data)
        response_data = response.json()
        output_base64 = response_data.get("output")
        close_host_process()
        if output_base64:
            output_bytes = base64.b64decode(output_base64)
            im2 = pickle.loads(output_bytes)
            tensor_image = ToTensor()(im2)                      # CHW
            tensor_image = tensor_image.unsqueeze(0)            # CHW -> NCHW
            tensor_image = tensor_image.permute(0, 2, 3, 1)     # NCHW -> NHWC
            return (tensor_image,)
        else:
            assert False, "No media generated"


class ADSVDSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "config":               ("AD_CONFIG",),
                "image":                ("IMAGE",),
                "width":                RESOLUTION,
                "height":               RESOLUTION,
                "seed":                 SEED,
                "steps":                STEPS,
                "guidance_scale":       CFG,
                "decode_chunk_size":    DECODE_CHUNK_SIZE,
                "num_frames":           NUM_FRAMES,
                "motion_bucket_id":     MOTION_BUCKET_ID,
                "noise_aug_strength":   NOISE_AUG_STRENGTH,
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "AsyncDiff"

    def generate(
        self,
        width, height, steps, guidance_scale, seed, decode_chunk_size, num_frames,
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
            "image":                b64_image,
            "width":                width,
            "height":               height,
            "num_inference_steps":  steps,
            "guidance_scale":       guidance_scale,
            "seed":                 seed,
            "decode_chunk_size":    decode_chunk_size,
            "num_frames":           num_frames,
            "motion_bucket_id":     motion_bucket_id,
            "noise_aug_strength":   noise_aug_strength,
            "output_path":          outputs_dir,
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
                "config":           ("AD_CONFIG",),
                "image":            ("IMAGE",),
                "positive_prompt":  PROMPT,
                "negative_prompt":  PROMPT,
                "scheduler":        SCHEDULERS,
                "seed":             SEED,
                "steps":            STEPS,
                "guidance_scale":   CFG,
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "AsyncDiff"

    def generate(self, config, image, positive_prompt, negative_prompt, scheduler, steps, guidance_scale, seed):
        config["pipeline_type"] = "sdup"
        config["scheduler"] = scheduler
        if image.size(0) > 1:
            images = list(torch.unbind(image, 0))   # NHWC -> [HWC], len == N
        else:
            images = [image.squeeze(0)]             # NHWC -> [HWC], len == 1
        tensors = []
        i = 0
        launch_host_process(config)
        for im in images:
            i += 1
            print(f"Upscaling image: {i}/{len(images)}")
            im = im.permute(2, 0, 1)                # HWC -> CHW
            im = ToPILImage()(im)
            pickled_image = pickle.dumps(im)
            b64_image = base64.b64encode(pickled_image).decode("utf-8")
            data = {
                "image":                b64_image,
                "positive_prompt":      positive_prompt,
                "negative_prompt":      negative_prompt,
                "num_inference_steps":  steps,
                "guidance_scale":       guidance_scale,
                "seed":                 seed,
                "output_path":          outputs_dir,
            }

            try:
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
                        print("No media generated")
                    else:
                        print(f"Error processing image: {i}/{len(images)}")
            except Exception as e:
                print("Error getting data from server.")
                print(str(e))
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
    ]

    if config.get("pipeline_type") in ["sdup"]:
        cmd.append(f'--scheduler={config.get("scheduler")}')

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
    "ADSD1Sampler":     ADSD1Sampler,
    "ADSD2Sampler":     ADSD2Sampler,
    "ADSD3Sampler":     ADSD3Sampler,
    "ADSDXLSampler":    ADSDXLSampler,
    "ADSVDSampler":     ADSVDSampler,
    "ADUpscaleSampler": ADUpscaleSampler,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "ADPipelineConfig": "ADPipelineConfig",
    "ADSD1Sampler":     "ADSD1Sampler",
    "ADSD2Sampler":     "ADSD2Sampler",
    "ADSD3Sampler":     "ADSD3Sampler",
    "ADSDXLSampler":    "ADSDXLSampler",
    "ADSVDSampler":     "ADSVDSampler",
    "ADUpscaleSampler": "ADUpscaleSampler",
}
