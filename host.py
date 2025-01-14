import argparse
import base64
import json
import logging
import os
import pickle
import safetensors
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from compel import Compel, ReturnedEmbeddingsType
from flask import Flask, request, jsonify
from PIL import Image

from AsyncDiff.asyncdiff.async_animate import AsyncDiff as AsyncDiffAD
from AsyncDiff.asyncdiff.async_sd import AsyncDiff as AsyncDiffSD
from diffusers import (
    AnimateDiffPipeline,
    MotionAdapter,
    StableDiffusion3Pipeline,
    StableDiffusionPipeline,
    StableDiffusionUpscalePipeline,
    StableDiffusionXLPipeline,
    StableVideoDiffusionPipeline,
)
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
    parser.add_argument("--model_n", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--stride", type=int, default=1, choices=[1, 2])
    parser.add_argument("--warm_up", type=int, default=1)
    parser.add_argument("--time_shift", type=bool, default=False)

    # added args
    parser.add_argument("--host_mode", type=str, default=None, choices=["comfyui", "localai"], help="Host operation mode")
    parser.add_argument("--pipeline_type", type=str, default=None, choices=["ad", "sd1", "sd2", "sd3", "sdup", "sdxl", "svd"])
    parser.add_argument("--motion_adapter", type=str, default=None)
    parser.add_argument("--variant", type=str, default="fp16", help="PyTorch variant [bf16/fp16/fp32]")
    parser.add_argument("--scheduler", type=str, default="ddim")
    parser.add_argument("--enable_model_cpu_offload", action="store_true")
    parser.add_argument("--enable_sequential_cpu_offload", action="store_true")
    parser.add_argument("--enable_tiling", action="store_true")
    parser.add_argument("--enable_slicing", action="store_true")
    parser.add_argument("--xformers_efficient", action="store_true")
    parser.add_argument("--scale_input", action="store_true")
    parser.add_argument("--scale_percentage", type=float, default=75.0)
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
    assert args.host_mode is not None, "Please specify a host operation mode."
    assert args.model is not None, "No model provided"
    assert args.variant in ["bf16", "fp16", "fp32"], "Unsupported variant"
    match args.variant:
        case "bf16":
            torch_dtype = torch.bfloat16
        case "fp16":
            torch_dtype = torch.float16
        case _:
            torch_dtype = torch.float32

    assert args.pipeline_type is not None, "No pipeline type provided"
    match args.pipeline_type:
        case "ad":
            assert args.motion_adapter is not None, "No motion adapter provided"
            adapter = MotionAdapter.from_pretrained(
                args.motion_adapter,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )
            pipe = AnimateDiffPipeline.from_pretrained(
                args.model,
                motion_adapter=adapter,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )
            scheduler = DDIMScheduler.from_pretrained(
                args.model,
                subfolder="scheduler",
                clip_sample=False,
                timestep_spacing="linspace",
                beta_schedule="linear",
                steps_offset=1,
            )
            pipe.scheduler = scheduler
        case "sd1":
            pipe = StableDiffusionPipeline.from_pretrained(
                args.model,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )
        case "sd2":
            pipe = StableDiffusionPipeline.from_pretrained(
                args.model,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )
        case "sd3":
            pipe = StableDiffusion3Pipeline.from_pretrained(
                args.model,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )
        case "sdup":
            pipe = StableDiffusionUpscalePipeline.from_pretrained(
                args.model,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )
        case "sdxl":
            pipe = StableDiffusionXLPipeline.from_pretrained(
                args.model,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )
        case "svd":
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                args.model,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )
        case _: raise NotImplementedError

    if args.pipeline_type in ["sd1", "sd2", "sd3", "sdup", "sdxl"]:
        pipe.scheduler = get_scheduler(args.scheduler, pipe.scheduler.config)

    if args.pipeline_type in ["ad"]:
        async_diff = AsyncDiffAD(
            pipe,
            args.pipeline_type,
            model_n=args.model_n,
            stride=args.stride,
            time_shift=args.time_shift,
        )
    else:
        async_diff = AsyncDiffSD(
            pipe,
            args.pipeline_type,
            model_n=args.model_n,
            stride=args.stride,
            time_shift=args.time_shift,
        )

    pipe.set_progress_bar_config(disable=dist.get_rank() != 0)

    if args.pipeline_type != "svd":
        if args.enable_slicing:
            pipe.enable_vae_slicing()
        if args.enable_tiling:
            pipe.enable_vae_tiling()
        if args.xformers_efficient:
            pipe.enable_xformers_memory_efficient_attention()
    if args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()

    # warm up
    logger.info("Starting warmup run")
    def get_warmup_image():
        image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true")
        image = image.resize((512, 288))
        return image
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    async_diff.reset_state(warm_up=args.warm_up)
    match args.pipeline_type:
        case "ad":
            pipe(
                prompt="test",
                negative_prompt="blurry",
                num_frames=15,
                guidance_scale=3.5,
                num_inference_steps=10,
            )
        case "sdup":
            pipe(
                prompt="detailed",
                negative_prompt="blurry",
                image=get_warmup_image(),
                num_inference_steps=10,
            )
        case "svd":
            pipe(
                image=get_warmup_image(),
                decode_chunk_size=1,
                num_inference_steps=10,
                width=320,
                height=320,
            )
        case _:
            pipe(
                prompt="test",
                negative_prompt="blurry",
                num_inference_steps=10,
            )

    torch.cuda.empty_cache()

    logger.info("Model initialization completed")
    initialized = True
    return


def generate_image_parallel(
    positive_prompt, negative_prompt, image, width, height,
    num_inference_steps, cfg, seed, num_frames, decode_chunk_size,
    motion_bucket_id, noise_aug_strength, output_path,
):
    global pipe, async_diff
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    args = get_args()

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    async_diff.reset_state(warm_up=args.warm_up)

    if args.pipeline_type in ["sdup", "svd"]:
        assert image is not None, "No image provided for an image pipeline."
        image = load_image(image)
        if args.scale_input:
            percentage = args.scale_percentage / 100
            image = image.resize(
                (
                    int(image.size[0] * percentage),
                    int(image.size[1] * percentage)
                ),
                Image.Resampling.LANCZOS
            )

    match args.pipeline_type:
        case "ad":
            output = pipe(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=cfg,
                num_frames=num_frames,
                output_type="pil",
            ).frames[0]
        case "sdup":
            output = pipe(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                image=image,
                num_inference_steps=num_inference_steps,
                guidance_scale=cfg,
                output_type="pil",
            ).images[0]
        case "svd":
            output = pipe(
                image,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                min_guidance_scale=cfg, # TODO: this should be config-able
                num_frames=num_frames,
                decode_chunk_size=decode_chunk_size,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
                output_type="pil",
            ).frames[0]
        case _:
            output = pipe(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=cfg,
                output_type="pil",
            ).images[0]

    end_time = time.time()
    elapsed_time = end_time - start_time

    torch.cuda.empty_cache()

    if dist.get_rank() == 0:
        if args.host_mode == "comfyui":
            return output, elapsed_time
        elif args.host_mode == "localai":
            match args.pipeline_type:
                case "svd":
                    output_path = output_path.replace(".png", "")
                    # export_to_video(output, f"{output_path}.mp4", fps=7)
                    export_to_gif(output, f"{output_path}.gif")
                case _:
                    output.save(output_path)
            return output_path, elapsed_time


@app.route("/generate", methods=["POST"])
def generate_image():
    args = get_args()

    logger.info("Received POST request for image generation")
    data = request.json
    positive_prompt     = data.get("positive_prompt", "")
    negative_prompt     = data.get("negative_prompt", None)
    image               = data.get("image", None)
    width               = data.get("width", 512)
    height              = data.get("height", 512)
    num_inference_steps = data.get("num_inference_steps", 30)
    guidance_scale      = data.get("guidance_scale", 3.5)
    seed                = data.get("seed", 1)
    num_frames          = data.get("num_frames", 25)
    decode_chunk_size   = data.get("decode_chunk_size", 8)
    motion_bucket_id    = data.get("motion_bucket_id", 180)
    noise_aug_strength  = data.get("noise_aug_strength", 0.1)
    output_path         = data.get("output_path", None)

    assert (image is not None or len(positive_prompt) > 0), "No input provided"
    assert output_path is not None, "No output path provided"

    if image is not None:
        if args.host_mode == "localai":
            image = Image.open(image)
        elif args.host_mode == "comfyui":
            image = base64.b64decode(image)
            image = pickle.loads(image)

    logger.info(
        "Request parameters:\n"
        # f"image={image}\n"
        f"positive_prompt={positive_prompt}\n"
        f"negative_prompt={negative_prompt}\n"
        f"width={width}\n"
        f"height={height}\n"
        f"steps={num_inference_steps}\n"
        f"cfg={guidance_scale}\n"
        f"seed={seed}\n"
        f"num_frames={num_frames}\n"
        f"decode_chunk_size={decode_chunk_size}\n"
        f"motion_bucket_id={motion_bucket_id}\n"
        f"noise_aug_strength={noise_aug_strength}\n"
        f"output_path={output_path}"
    )

    # Broadcast request parameters to all processes
    params = [
        positive_prompt,
        negative_prompt,
        image,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        seed,
        num_frames,
        decode_chunk_size,
        motion_bucket_id,
        noise_aug_strength,
        output_path
    ]
    dist.broadcast_object_list(params, src=0)
    logger.info("Parameters broadcasted to all processes")

    output, elapsed_time = generate_image_parallel(*params)

    response = {
        "message": "Image generated successfully",
        "elapsed_time": f"{elapsed_time:.2f} sec",
    }
    if args.host_mode == "comfyui":
        output = pickle.dumps(output)
        output_b64 = base64.b64encode(output).decode("utf-8")
        response["output"] = output_b64
    elif args.host_mode == "localai":
        response["output_path"] = output
    logger.info("Sending response")
    return jsonify(response)


def run_host():
    if dist.get_rank() == 0:
        logger.info("Starting Flask host on rank 0")
        app.run(host="0.0.0.0", port=6000)
    else:
        while True:
            params = [None] * 13 # len(params) of generate_image_parallel()
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
