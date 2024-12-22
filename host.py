# https://github.com/xdit-project/xDiT/blob/1c31746e2f903e791bc2a41a0bc23614958e46cd/comfyui-xdit/host.py

import argparse
import base64
import json
import logging
import os
import pickle
import random
import safetensors
import string
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torchvision.io import decode_image

from compel import Compel, ReturnedEmbeddingsType
from flask import Flask, request, jsonify
from PIL import Image

from diffusers import StableVideoDiffusionPipeline
from AsyncDiff.asyncdiff.async_sd import AsyncDiff
from diffusers.utils import load_image, export_to_video, export_to_gif

from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)

app = Flask(__name__)

os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

pipe = None
engine_config = None
input_config = None
local_rank = None
logger = None
initialized = False
async_diff = None


def get_scheduler(scheduler_name, current_scheduler_config):
    scheduler_class = get_scheduler_class(scheduler_name)
    scheduler_config = get_scheduler_config(scheduler_name, current_scheduler_config)
    return scheduler_class.from_config(scheduler_config)


def get_scheduler_class(scheduler_name):
    if scheduler_name.startswith("k_"):
        scheduler_name.replace("k_", "", 1)

    match scheduler_name:
        case "ddim":            return DDIMScheduler
        case "euler":           return EulerDiscreteScheduler
        case "euler_a":         return EulerAncestralDiscreteScheduler
        case "dpm_2":           return KDPM2DiscreteScheduler
        case "dpm_2_a":         return KDPM2AncestralDiscreteScheduler
        case "dpmpp_2m":        return DPMSolverMultistepScheduler
        case "dpmpp_2m_sde":    return DPMSolverMultistepScheduler
        case "dpmpp_sde":       return DPMSolverSinglestepScheduler
        case "heun":            return HeunDiscreteScheduler
        case "lms":             return LMSDiscreteScheduler
        case "pndm":            return PNDMScheduler
        case "unipc":           return UniPCMultistepScheduler
        case _:                 raise NotImplementedError


def get_scheduler_config(scheduler_name, current_scheduler_config):
    if scheduler_name.startswith("k_"):
        current_scheduler_config["use_karras_sigmas"] = True
    match scheduler_name:
        case "dpmpp_2m":
            current_scheduler_config["algorithm_type"] = "dpmsolver++"
            current_scheduler_config["solver_order"] = 2
        case "dpmpp_2m_sde":
            current_scheduler_config["algorithm_type"] = "sde-dpmsolver++"
    return current_scheduler_config


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)   
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--model_n", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--warm_up", type=int, default=1)
    parser.add_argument("--time_shift", type=bool, default=False)
    # added args
    parser.add_argument("--pipeline_type", type=str, default=None, choices=["epic", "sd1", "sd2", "sd3", "sdup", "sdxl", "svd"])
    # parser.add_argument("--scheduler", type=str, default="ddim")
    args = parser.parse_args()
    return args


def setup_logger():
    global logger
    global local_rank
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank {local_rank}] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)


@app.route("/initialize", methods=["GET"])
def check_initialize():
    global initialized
    if initialized:
        return jsonify({"status": "initialized"}), 200
    else:
        return jsonify({"status": "initializing"}), 202


def initialize():
    global pipe, local_rank, async_diff, initialized
    mp.set_start_method("spawn", force=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    setup_logger()
    logger.info(f"Initializing model on GPU: {torch.cuda.current_device()}")

    args = get_args()
    assert args.model is not None, "No model provided"
    assert args.pipeline_type is not None, "No pipeline type provided"

    # Load the conditioning image
    image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true")
    file_name = "rocket"
    image = image.resize((512, 288))

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        use_safetensors=True,
        low_cpu_mem_usage=False, #True
        local_files_only=True,
    )
    # pipe.scheduler = get_scheduler(args.scheduler, pipe.scheduler.config)

    async_diff = AsyncDiff(
        pipe,
        args.pipeline_type,
        model_n=args.model_n,
        stride=args.stride,
        time_shift=args.time_shift
    )

    pipe.set_progress_bar_config(disable=dist.get_rank() != 0)

    # warm up
    logger.info("Starting warmup run")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    async_diff.reset_state(warm_up=args.warm_up)
    pipe(image, decode_chunk_size=1, num_inference_steps=2, width=320, height=320)
    logger.info("Model initialization completed")
    initialized = True
    return


def generate_image_parallel(image, width, height, num_inference_steps, seed, num_frames, decode_chunk_size, output_path):
    global pipe, async_diff
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    args = get_args()

    # inference
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    async_diff.reset_state(warm_up=args.warm_up)

    image = load_image(image)
    tensors = pipe(
        image,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        decode_chunk_size=decode_chunk_size,
        output_type="pil", # "pt"
        # motion_bucket_id=50,
        # noise_aug_strength=0.1,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time

    # chars = string.ascii_lowercase + string.digits
    # file_name = ''.join(random.choices(chars, k=16))
    # file_path = f"{output_path}/{file_name}.gif"
    # if dist.get_rank() == 0:
    #     export_to_video(frames, f"{file_path}.mp4", fps=7)
    #     export_to_gif(frames, f"{file_path}")

    return tensors, elapsed_time

@app.route("/generate", methods=["POST"])
def generate_image():
    logger.info("Received POST request for image generation")
    data = request.json
    image               = data.get("image", None)
    width               = data.get("width", 512)
    height              = data.get("height", 512)
    num_inference_steps = data.get("num_inference_steps", 30)
    seed                = data.get("seed", 1)
    num_frames          = data.get("num_frames", 25)
    decode_chunk_size   = data.get("decode_chunk_size", 8)
    output_path         = data.get("output_path", None)

    assert image is not None, "No image provided"
    assert output_path is not None, "No output path provided"

    logger.info(
        "Request parameters:\n"
        f"width={width}\n"
        f"height={height}\n"
        f"steps={num_inference_steps}\n"
        f"seed={seed}\n"
        f"num_frames={num_frames}\n"
        f"decode_chunk_size={decode_chunk_size}\n"
        f"output_path={output_path}"
    )
    # Broadcast request parameters to all processes
    image = base64.b64decode(image)
    image = pickle.loads(image)
    params = [image, width, height, num_inference_steps, seed, num_frames, decode_chunk_size, output_path]
    dist.broadcast_object_list(params, src=0)
    logger.info("Parameters broadcasted to all processes")

    tensors, elapsed_time = generate_image_parallel(*params)

    output = pickle.dumps(tensors)
    output_b64 = base64.b64encode(output).decode("utf-8")
    response = {
        "message": "Image generated successfully",
        "elapsed_time": f"{elapsed_time:.2f} sec",
        "output": output_b64,
    }

    logger.info("Sending response")
    return jsonify(response)


def run_host():
    if dist.get_rank() == 0:
        logger.info("Starting Flask host on rank 0")
        app.run(host="0.0.0.0", port=6000)
    else:
        while True:
            params = [None] * 8 # len(params) of generate_image_parallel()
            logger.info(f"Rank {dist.get_rank()} waiting for tasks")
            dist.broadcast_object_list(params, src=0)
            if params[0] is None:
                logger.info("Received exit signal, shutting down")
                break
            logger.info(f"Received task with parameters: {params}")
            generate_image_parallel(*params)


if __name__ == "__main__":
    initialize()

    logger.info(
        f"Process initialized. Rank: {dist.get_rank()}, Local Rank: {os.environ.get('LOCAL_RANK', 'Not Set')}"
    )
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    run_host()
