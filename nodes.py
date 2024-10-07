import torch

try:
    from comfy.model_patcher import ModelPatcher

    # #BACKEND = "ComfyUI"
except ImportError:
    try:
        from ldm_patched.modules.model_patcher import ModelPatcher

        # #BACKEND = "reForge"
    except ImportError:
        from backend.patcher.base import ModelPatcher

        # #BACKEND = "Forge"

class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0
        
    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average
        
def project( v0: torch.Tensor, v1: torch.Tensor,):
    dtype = v0.dtype
    #v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)
    
def normalized_guidance( pred_cond: torch.Tensor, pred_uncond: torch.Tensor, guidance_scale: float, momentum_buffer: MomentumBuffer = None, eta: float = 1.0, norm_threshold: float = 0.0,):
    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    diff_parallel, diff_orthogonal = project(diff, pred_cond)
    normalized_update = diff_orthogonal + eta * diff_parallel
    pred_guided = pred_cond + (guidance_scale - 1) * normalized_update
    
    return pred_guided

class APG_ImYourCFGNow:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "scale": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "momentum": ("FLOAT", {"default": -0.5, "min": -1.5, "max": 0.5, "step": 0.1, "round": 0.01}),
                "norm_threshold": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 50.0, "step": 0.5, "round": 0.01}),
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/unet"

    def patch(
        self,
        model: ModelPatcher,
        scale: float = 12.0,
        momentum: float = -0.5,
        norm_threshold: float = 15.0,
        eta: float = 1.0,
    ):
        
        momentum_buffer = MomentumBuffer(momentum)

        def apg_function(args):
            cond = args["cond"]
            uncond = args["uncond"]

            return normalized_guidance(cond, uncond, scale, momentum_buffer, eta, norm_threshold)

        m = model.clone()
        m.set_model_sampler_cfg_function(apg_function, momentum_buffer)

        return (m,)
        

NODE_CLASS_MAPPINGS = {
    "APG_ImYourCFGNow": APG_ImYourCFGNow,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "APG_ImYourCFGNow": "APG_ImYourCFGNow",
}
