#!/usr/bin/env python3
import base64
import json
import os
import pickle
import requests
import subprocess
import time
import torch
from glob import glob
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor


host_address            = 'http://localhost:6000'
host_process            = None
host_address_generate   = f'{host_address}/generate'
host_address_initialize = f'{host_address}/initialize'
connection_attempts_max = 60


cwd             = os.path.dirname(__file__)
comfy_root      = os.path.dirname(os.path.dirname(cwd))
outputs_dir     = os.path.join(comfy_root, "output")
models_dir      = os.path.join(os.path.join(comfy_root, "models"))
checkpoints_dir = os.path.join(models_dir, "checkpoints")
models          = [d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d))]


def getModelSubfolder(folderIn):
    model_extensions = ["bin", "safetensors"]
    out = []
    for path in os.walk(folderIn):
        for f in glob(os.path.join(path[0], '**/*.*'), recursive=True):
            if f.split(".")[-1] in model_extensions:
                out.append(f.replace(folderIn + "/", ""))
    return out


loras_dir       = os.path.join(models_dir, "loras")
loras           = getModelSubfolder(loras_dir)
ipadapter_dir   = os.path.join(models_dir, "ipadapter")
ipadapters      = getModelSubfolder(ipadapter_dir)


INT_MAX                 = 2 ** 32 - 1
INT_MIN                 = -1 * INT_MAX
BOOLEAN_DEFAULT_FALSE   = ("BOOLEAN", { "default": False })
CONFIG                  = ("AD_CONFIG",)
IMAGE                   = ("IMAGE",)
LORA                    = ("AD_LORA",)
LORA_LIST               = (loras,)
MODEL                   = ("AD_MODEL",)
MODEL_LIST              = (models,)
IPADAPTER               = ("AD_IPADAPTER",)
IPADAPTER_LIST          = (ipadapters,)


SCHEDULERS = (
    list([
        "ddim",
        "euler",
        "euler_a",
        "dpm_2",
        "dpm_2_a",
        "dpmpp_2m",
        "dpmpp_2m_sde",
        "dpmpp_sde",
        "heun",
        "lms",
        "pndm",
        "unipc"
    ]),
    {
        "default": "dpmpp_2m"
    }
)
VARIANT                 = (["bf16", "fp16", "fp32"],  { "default": "fp16" })
MODEL_N                 = ("INT",                     { "default": 2,     "min": 2,       "max": 4,       "step": 1 })
STRIDE                  = ("INT",                     { "default": 1,     "min": 1,       "max": 2,       "step": 1 })
NPROC_PER_NODE          = ("INT",                     { "default": 2,     "min": 1,       "max": 4,       "step": 1 })
SCALE_PERCENTAGE        = ("FLOAT",                   { "default": 75.0,  "min": 0.01,    "max": INT_MAX, "step": 0.01 })


PROMPT                  = ("STRING",                  { "default": "",    "multiline": True })
RESOLUTION              = ("INT",                     { "default": 512,   "min": 8,       "max": INT_MAX, "step": 8 })
SEED                    = ("INT",                     { "default": 0,     "min": 0,       "max": INT_MAX, "step": 1 })
STEPS                   = ("INT",                     { "default": 60,    "min": 1,       "max": INT_MAX, "step": 1 })
DECODE_CHUNK_SIZE       = ("INT",                     { "default": 8,     "min": 1,       "max": INT_MAX, "step": 1 })
NUM_FRAMES              = ("INT",                     { "default": 25,    "min": 1,       "max": INT_MAX, "step": 1 })
MOTION_BUCKET_ID        = ("INT",                     { "default": 180,   "min": 1,       "max": INT_MAX, "step": 1 })
CFG                     = ("FLOAT",                   { "default": 3.5,   "min": 0,       "max": INT_MAX, "step": 0.1 })
NOISE_AUG_STRENGTH      = ("FLOAT",                   { "default": 0.01,  "min": 0,       "max": INT_MAX, "step": 0.01 })
LORA_WEIGHT             = ("FLOAT",                   { "default": 1.00,  "min": INT_MIN, "max": INT_MAX, "step": 0.01 })
IP_ADAPTER_SCALE        = ("FLOAT",                   { "default": 1.00,  "min": INT_MIN, "max": INT_MAX, "step": 0.01 })


class ADModelSelector:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "model": MODEL_LIST } }

    RETURN_TYPES    = MODEL
    FUNCTION        = "get_model"
    CATEGORY        = "AsyncDiff"        

    def get_model(self, model):
        return (model,)


class ADLoraSelector:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "lora": LORA_LIST, "weight": LORA_WEIGHT } }

    RETURN_TYPES    = LORA
    FUNCTION        = "get_lora_with_weight"
    CATEGORY        = "AsyncDiff"        

    def get_lora_with_weight(self, lora, weight):
        return ([{ "lora": loras_dir + "/" + lora, "weight": weight }],)


class ADIPAdapterSelector:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "ip_adapter": IPADAPTER_LIST, "scale": IP_ADAPTER_SCALE } }

    RETURN_TYPES    = IPADAPTER
    FUNCTION        = "get_ipadapter_with_scale"
    CATEGORY        = "AsyncDiff"        

    def get_ipadapter_with_scale(self, ip_adapter, scale):
        return ({ "ip_adapter": ip_adapter, "scale": scale },)


class ADMultiLoraSelector:
    @classmethod
    def INPUT_TYPES(s):
        return { "optional": { "lora_1": LORA, "lora_2": LORA, "lora_3": LORA, "lora_4": LORA } }

    RETURN_TYPES    = LORA
    FUNCTION        = "get_multi_lora"
    CATEGORY        = "AsyncDiff"        

    def get_multi_lora(self, **kwargs):
        loras = []
        for k, v in kwargs.items():
            for lora in v:
                loras.append({ "lora": lora.get("lora"), "weight": lora.get("weight") })
        return (loras,)


class ADPipelineConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
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

    RETURN_TYPES    = CONFIG
    FUNCTION        = "get_config"
    CATEGORY        = "AsyncDiff"        

    def get_config(
        self, nproc_per_node, model_n, stride, time_shift, variant,
        scale_input, scale_percentage, enable_tiling, enable_slicing, xformers_efficient,
        enable_model_cpu_offload=False, enable_sequential_cpu_offload=False
    ):
        return (
            {
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


class ADADSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model":            MODEL,
                "motion_adapter":   MODEL,
                "config":           CONFIG,
                "positive_prompt":  PROMPT,
                "negative_prompt":  PROMPT,
                "width":            RESOLUTION,
                "height":           RESOLUTION,
                "seed":             SEED,
                "steps":            STEPS,
                "guidance_scale":   CFG,
                "num_frames":       NUM_FRAMES,
            }
        }

    RETURN_TYPES    = IMAGE
    FUNCTION        = "generate"
    CATEGORY        = "AsyncDiff"

    def generate(
        self,
        model, motion_adapter, config, positive_prompt, negative_prompt,
        width, height, seed, steps, guidance_scale, num_frames
    ):
        config["pipeline_type"]     = "ad"
        config["model"]             = model
        config["motion_adapter"]    = motion_adapter
        launch_host_process(config)

        data = {
            "positive_prompt":      positive_prompt,
            "negative_prompt":      negative_prompt,
            "width":                width,
            "height":               height,
            "seed":                 seed,
            "num_inference_steps":  steps,
            "guidance_scale":       guidance_scale,
            "num_frames":           num_frames,
        }
        response = get_result(data)
        close_host_process()
        if response is not None:
            print("Media generated")
            output_bytes = base64.b64decode(response)
            images = pickle.loads(output_bytes)
            tensors = []
            for i in images:
                tensors.append(convert_image_to_hwc_tensor(i))
            return (torch.stack(tuple(tensors)),)   # HWC -> NHWC
        else:
            assert False, "No media generated"


class ADADIPASampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model":            MODEL,
                "motion_adapter":   MODEL,
                "ip_adapter":       IPADAPTER,
                "image":            IMAGE,
                "config":           CONFIG,
                "positive_prompt":  PROMPT,
                "negative_prompt":  PROMPT,
                "width":            RESOLUTION,
                "height":           RESOLUTION,
                "seed":             SEED,
                "steps":            STEPS,
                "guidance_scale":   CFG,
                "num_frames":       NUM_FRAMES,
            }
        }

    RETURN_TYPES    = IMAGE
    FUNCTION        = "generate"
    CATEGORY        = "AsyncDiff"

    def generate(
        self,
        model, motion_adapter, ip_adapter, image, config, positive_prompt, negative_prompt,
        width, height, seed, steps, guidance_scale, num_frames
    ):
        config["pipeline_type"]     = "ad"
        config["model"]             = model
        config["motion_adapter"]    = motion_adapter
        config["ip_adapter"]        = ip_adapter.get("ip_adapter")
        config["ip_adapter_scale"]  = ip_adapter.get("scale")
        launch_host_process(config)

        image = image.squeeze(0)        # NHWC -> HWC
        b64_image = convert_tensor_to_b64(image)
        data = {
            "image":                b64_image,
            "positive_prompt":      positive_prompt,
            "negative_prompt":      negative_prompt,
            "width":                width,
            "height":               height,
            "seed":                 seed,
            "num_inference_steps":  steps,
            "guidance_scale":       guidance_scale,
            "num_frames":           num_frames,
        }
        response = get_result(data)
        close_host_process()
        if response is not None:
            print("Media generated")
            output_bytes = base64.b64decode(response)
            images = pickle.loads(output_bytes)
            tensors = []
            for i in images:
                tensors.append(convert_image_to_hwc_tensor(i))
            return (torch.stack(tuple(tensors)),)   # HWC -> NHWC
        else:
            assert False, "No media generated"


class ADSD1Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model":            MODEL,
                "config":           CONFIG,
                "positive_prompt":  PROMPT,
                "negative_prompt":  PROMPT,
                "width":            RESOLUTION,
                "height":           RESOLUTION,
                "scheduler":        SCHEDULERS,
                "seed":             SEED,
                "steps":            STEPS,
                "guidance_scale":   CFG,
            },
            "optional": {
                "lora":             LORA,
            }
        }

    RETURN_TYPES    = IMAGE
    FUNCTION        = "generate"
    CATEGORY        = "AsyncDiff"

    def generate(self, model, config, positive_prompt, negative_prompt, width, height, scheduler, seed, steps, guidance_scale, lora=None):
        config["pipeline_type"]     = "sd1"
        config["model"]             = model
        config["scheduler"]         = scheduler
        if lora is not None:
            config["lora"] = lora
        launch_host_process(config)

        data = {
            "positive_prompt":      positive_prompt,
            "negative_prompt":      negative_prompt,
            "width":                width,
            "height":               height,
            "seed":                 seed,
            "num_inference_steps":  steps,
            "guidance_scale":       guidance_scale,
        }
        response = get_result(data)
        close_host_process()
        if response is not None:
            return (convert_b64_to_nhwc_tensor(response),)
        else:
            assert False, "No media generated"


class ADSD2Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model":            MODEL,
                "config":           CONFIG,
                "positive_prompt":  PROMPT,
                "negative_prompt":  PROMPT,
                "width":            RESOLUTION,
                "height":           RESOLUTION,
                "scheduler":        SCHEDULERS,
                "seed":             SEED,
                "steps":            STEPS,
                "guidance_scale":   CFG,
            },
            "optional": {
                "lora":             LORA,
            }
        }

    RETURN_TYPES    = IMAGE
    FUNCTION        = "generate"
    CATEGORY        = "AsyncDiff"

    def generate(self, model, config, positive_prompt, negative_prompt, width, height, scheduler, seed, steps, guidance_scale, lora=None):
        config["pipeline_type"]     = "sd2"
        config["model"]             = model
        config["scheduler"]         = scheduler
        if lora is not None:
            config["lora"] = lora
        launch_host_process(config)

        data = {
            "positive_prompt":      positive_prompt,
            "negative_prompt":      negative_prompt,
            "width":                width,
            "height":               height,
            "seed":                 seed,
            "num_inference_steps":  steps,
            "guidance_scale":       guidance_scale,
        }
        response = get_result(data)
        close_host_process()
        if response is not None:
            return (convert_b64_to_nhwc_tensor(response),)
        else:
            assert False, "No media generated"


class ADSD3Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model":            MODEL,
                "config":           CONFIG,
                "positive_prompt":  PROMPT,
                "negative_prompt":  PROMPT,
                "width":            RESOLUTION,
                "height":           RESOLUTION,
                "scheduler":        SCHEDULERS,
                "seed":             SEED,
                "steps":            STEPS,
                "guidance_scale":   CFG,
            },
            "optional": {
                "lora":             LORA,
            }
        }

    RETURN_TYPES    = IMAGE
    FUNCTION        = "generate"
    CATEGORY        = "AsyncDiff"

    def generate(self, model, config, positive_prompt, negative_prompt, width, height, scheduler, seed, steps, guidance_scale, lora=None):
        config["pipeline_type"]     = "sd3"
        config["model"]             = model
        config["scheduler"]         = scheduler
        if lora is not None:
            config["lora"] = lora
        launch_host_process(config)

        data = {
            "positive_prompt":      positive_prompt,
            "negative_prompt":      negative_prompt,
            "width":                width,
            "height":               height,
            "seed":                 seed,
            "num_inference_steps":  steps,
            "guidance_scale":       guidance_scale,
        }
        response = get_result(data)
        close_host_process()
        if response is not None:
            return (convert_b64_to_nhwc_tensor(response),)
        else:
            assert False, "No media generated"


class ADSDXLSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model":            MODEL,
                "config":           CONFIG,
                "positive_prompt":  PROMPT,
                "negative_prompt":  PROMPT,
                "width":            RESOLUTION,
                "height":           RESOLUTION,
                "scheduler":        SCHEDULERS,
                "seed":             SEED,
                "steps":            STEPS,
                "guidance_scale":   CFG,
            },
            "optional": {
                "lora":            LORA,
            }
        }

    RETURN_TYPES    = IMAGE
    FUNCTION        = "generate"
    CATEGORY        = "AsyncDiff"

    def generate(self, model, config, positive_prompt, negative_prompt, width, height, scheduler, seed, steps, guidance_scale, lora=None):
        config["pipeline_type"]     = "sdxl"
        config["model"]             = model
        config["scheduler"]         = scheduler
        if lora is not None:
            config["lora"] = lora
        launch_host_process(config)

        data = {
            "positive_prompt":      positive_prompt,
            "negative_prompt":      negative_prompt,
            "width":                width,
            "height":               height,
            "seed":                 seed,
            "num_inference_steps":  steps,
            "guidance_scale":       guidance_scale,
        }
        response = get_result(data)
        close_host_process()
        if response is not None:
            return (convert_b64_to_nhwc_tensor(response),)
        else:
            assert False, "No media generated"


class ADSVDSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model":                MODEL,
                "config":               CONFIG,
                "image":                IMAGE,
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

    RETURN_TYPES    = IMAGE
    FUNCTION        = "generate"
    CATEGORY        = "AsyncDiff"

    def generate(
        self,
        model, config, image, width, height, seed, steps, guidance_scale,
        decode_chunk_size, num_frames, motion_bucket_id, noise_aug_strength
    ):
        config["pipeline_type"]     = "svd"
        config["model"]             = model
        launch_host_process(config)

        image = image.squeeze(0)        # NHWC -> HWC
        b64_image = convert_tensor_to_b64(image)
        data = {
            "image":                b64_image,
            "width":                width,
            "height":               height,
            "seed":                 seed,
            "num_inference_steps":  steps,
            "guidance_scale":       guidance_scale,
            "decode_chunk_size":    decode_chunk_size,
            "num_frames":           num_frames,
            "motion_bucket_id":     motion_bucket_id,
            "noise_aug_strength":   noise_aug_strength,
        }
        response = get_result(data)
        close_host_process()
        if response is not None:
            print("Media generated")
            output_bytes = base64.b64decode(response)
            images = pickle.loads(output_bytes)
            tensors = []
            for i in images:
                tensors.append(convert_image_to_hwc_tensor(i))
            return (torch.stack(tuple(tensors)),)   # HWC -> NHWC
        else:
            assert False, "No media generated"


class ADSDUpscaleSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model":            MODEL,
                "config":           CONFIG,
                "image":            IMAGE,
                "positive_prompt":  PROMPT,
                "negative_prompt":  PROMPT,
                "scheduler":        SCHEDULERS,
                "seed":             SEED,
                "steps":            STEPS,
                "guidance_scale":   CFG,
            }
        }

    RETURN_TYPES    = IMAGE
    FUNCTION        = "generate"
    CATEGORY        = "AsyncDiff"

    def generate(self, model, config, image, positive_prompt, negative_prompt, scheduler, seed, steps, guidance_scale):
        config["pipeline_type"]     = "sdup"
        config["model"]             = model
        config["scheduler"]         = scheduler
        launch_host_process(config)

        if image.size(0) > 1:
            images = list(torch.unbind(image, 0))   # NHWC -> [HWC], len == N
        else:
            images = [image.squeeze(0)]             # NHWC -> [HWC], len == 1
        tensors = []
        i = 0
        for im in images:
            i += 1
            print(f"Upscaling image: {i}/{len(images)}")
            b64_image = convert_tensor_to_b64(im)
            data = {
                "image":                b64_image,
                "positive_prompt":      positive_prompt,
                "negative_prompt":      negative_prompt,
                "seed":                 seed,
                "num_inference_steps":  steps,
                "guidance_scale":       guidance_scale,
            }

            try:
                response = get_result(data)
                if response is not None:
                    print(f"Finished upscaling image: {i}/{len(images)}")
                    output_bytes = base64.b64decode(response)
                    im2 = pickle.loads(output_bytes)
                    tensors.append(convert_image_to_hwc_tensor(im2))
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

    if config.get("pipeline_type") in ["sd1", "sd2", "sd3", "sdup", "sdxl"]:
        cmd.append(f'--scheduler={config.get("scheduler")}')

    if config.get("pipeline_type") in ["ad"]:
        cmd.append(f'--motion_adapter={checkpoints_dir}/{config.get("motion_adapter")}')

    if config.get("scale_input"):
        cmd.append('--scale_input')
        cmd.append(f'--scale_percentage={config.get("scale_percentage")}')

    if config.get("lora"):
        cmd.append(f'--lora={json.dumps(config.get("lora"))}')

    if config.get("ip_adapter"):
        cmd.append(f'--ip_adapter={ipadapter_dir}/{config.get("ip_adapter")}')
        cmd.append(f'--ip_adapter_scale={config.get("ip_adapter_scale")}')

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
        if connection_attempts >= connection_attempts_max:
            assert False, "Failed to launch host. Check logs for details."


def close_host_process():
    global host_process
    if host_process is not None:
        host_process.terminate()
        host_process = None
    return


def get_result(data):
    data["output_path"] = outputs_dir
    response = requests.post(host_address_generate, json=data)
    response_data = response.json()
    output_base64 = response_data.get("output")
    return output_base64


def convert_b64_to_nhwc_tensor(b64):
    output_bytes = base64.b64decode(b64)
    im2 = pickle.loads(output_bytes)
    tensor_image = ToTensor()(im2)                      # CHW
    tensor_image = tensor_image.unsqueeze(0)            # CHW -> NCHW
    tensor_image = tensor_image.permute(0, 2, 3, 1)     # NCHW -> NHWC
    return tensor_image


def convert_image_to_hwc_tensor(image):
    tensor = ToTensor()(image)          # CHW
    tensor = tensor.permute(1, 2, 0)    # CHW -> HWC
    return tensor


def convert_tensor_to_b64(tensor):
    im = tensor.permute(2, 0, 1)        # HWC -> CHW
    im = ToPILImage()(im)
    pickled_image = pickle.dumps(im)
    b64_image = base64.b64encode(pickled_image).decode("utf-8")
    return b64_image


NODE_CLASS_MAPPINGS = {
    "ADADSampler":          ADADSampler,
    "ADADIPASampler":       ADADIPASampler,
    "ADIPAdapterSelector":  ADIPAdapterSelector,
    "ADLoraSelector":       ADLoraSelector,
    "ADModelSelector":      ADModelSelector,
    "ADMultiLoraSelector":  ADMultiLoraSelector,
    "ADPipelineConfig":     ADPipelineConfig,
    "ADSD1Sampler":         ADSD1Sampler,
    "ADSD2Sampler":         ADSD2Sampler,
    "ADSD3Sampler":         ADSD3Sampler,
    "ADSDUpscaleSampler":   ADSDUpscaleSampler,
    "ADSDXLSampler":        ADSDXLSampler,
    "ADSVDSampler":         ADSVDSampler,
    
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "ADADSampler":          "ADADSampler",
    "ADADIPASampler":       "ADADIPASampler",
    "ADIPAdapterSelector":  "ADIPAdapterSelector",
    "ADLoraSelector":       "ADLoraSelector",
    "ADModelSelector":      "ADModelSelector",
    "ADMultiLoraSelector":  "ADMultiLoraSelector",
    "ADPipelineConfig":     "ADPipelineConfig",
    "ADSD1Sampler":         "ADSD1Sampler",
    "ADSD2Sampler":         "ADSD2Sampler",
    "ADSD3Sampler":         "ADSD3Sampler",
    "ADSDUpscaleSampler":   "ADSDUpscaleSampler",
    "ADSDXLSampler":        "ADSDXLSampler",
    "ADSVDSampler":         "ADSVDSampler",
}
