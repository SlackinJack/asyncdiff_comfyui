files_to_patch = [
    {
        "file_name": "AsyncDiff/asyncdiff/async_sd.py",
        "replace": [
            {
                "from": "def __init__(self, pipeline, model_n=2, stride=1, warm_up=1, time_shift=False):",
                "to":   "def __init__(self, pipeline, pipeline_type, model_n=2, stride=1, warm_up=1, time_shift=False):",
            },
            {
                "from": "self.pipe_id = pipeline.config._name_or_path",
                "to":   "self.pipe_id = pipeline_type",
            },
        ],
    },
    {
        "file_name": "AsyncDiff/asyncdiff/pipe_config.py",
        "replace": [
            {
                "from": "stabilityai/stable-diffusion-3-medium-diffusers",
                "to":   "sd3",
            },
            {
                "from": "stabilityai/stable-video-diffusion-img2vid-xt",
                "to":   "svd",
            },
            {
                "from": "stabilityai/stable-diffusion-2-1",
                "to":   "sd2",
            },
            {
                "from": "runwayml/stable-diffusion-v1-5",
                "to":   "sd1",
            },
            {
                "from": "pipe_id == \"stabilityai/stable-diffusion-xl-base-1.0\" or pipe_id == \"RunDiffusion/Juggernaut-X-v10\" or pipe_id == \"diffusers/stable-diffusion-xl-1.0-inpainting-0.1\"",
                "to":   "pipe_id == \"sdxl\"",
            },
            {
                "from": "emilianJR/epiCRealism",
                "to":   "epic",
            },
            {
                "from": "stabilityai/stable-diffusion-x4-upscaler",
                "to":   "sdup",
            },
        ],
    },
]


for f in files_to_patch:
    file_name = f.get("file_name")
    try:
        print(f"Patching file: {file_name}")
        with open(file_name, "r") as file:              data = file.read()                                  # read file
        with open(file_name + ".bak", "w") as file:     file.write(data)                                    # create backup
        for r in f.get("replace"):                      data = data.replace(r.get("from"), r.get("to"))     # replace strings
        with open(file_name, "w") as file:              file.write(data)                                    # overwrite file
        print(f"File patched: {file_name}")
    except Exception as e:
        print(f"Failed to patch file: {file_name}")
        print(str(e))

