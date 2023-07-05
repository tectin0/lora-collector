from typing import Any, List, Dict

"""
The following is a "shallow" copy of the StableDiffusionProcessing class from the original repo only for the purpose of type hinting.
"""


class StableDiffusionProcessing:
    """
    The first set of paramaters: sd_models -> do_not_reload_embeddings represent the minimum required to create a StableDiffusionProcessing
    """

    cached_uc = [None, None]
    cached_c = [None, None]

    def __init__(
        self,
        sd_model=None,
        outpath_samples=None,
        outpath_grids=None,
        prompt: str = "",
        styles: List[str] = None,
        seed: int = -1,
        subseed: int = -1,
        subseed_strength: float = 0,
        seed_resize_from_h: int = -1,
        seed_resize_from_w: int = -1,
        seed_enable_extras: bool = True,
        sampler_name: str = None,
        batch_size: int = 1,
        n_iter: int = 1,
        steps: int = 50,
        cfg_scale: float = 7.0,
        width: int = 512,
        height: int = 512,
        restore_faces: bool = False,
        tiling: bool = False,
        do_not_save_samples: bool = False,
        do_not_save_grid: bool = False,
        extra_generation_params: Dict[Any, Any] = None,
        overlay_images: Any = None,
        negative_prompt: str = None,
        eta: float = None,
        do_not_reload_embeddings: bool = False,
        denoising_strength: float = 0,
        ddim_discretize: str = None,
        s_min_uncond: float = 0.0,
        s_churn: float = 0.0,
        s_tmax: float = None,
        s_tmin: float = 0.0,
        s_noise: float = 1.0,
        override_settings: Dict[str, Any] = None,
        override_settings_restore_afterwards: bool = True,
        sampler_index: int = None,
        script_args: list = None,
    ):
        self.outpath_samples: str = outpath_samples
        self.outpath_grids: str = outpath_grids
        self.prompt: str = prompt
        self.prompt_for_display: str = None
        self.negative_prompt: str = negative_prompt or ""
        self.styles: list = styles or []
        self.seed: int = seed
        self.subseed: int = subseed
        self.subseed_strength: float = subseed_strength
        self.seed_resize_from_h: int = seed_resize_from_h
        self.seed_resize_from_w: int = seed_resize_from_w
        self.sampler_name: str = sampler_name
        self.batch_size: int = batch_size
        self.n_iter: int = n_iter
        self.steps: int = steps
        self.cfg_scale: float = cfg_scale
        self.width: int = width
        self.height: int = height
        self.restore_faces: bool = restore_faces
        self.tiling: bool = tiling
        self.do_not_save_samples: bool = do_not_save_samples
        self.do_not_save_grid: bool = do_not_save_grid
        self.extra_generation_params: dict = extra_generation_params or {}
        self.overlay_images = overlay_images
        self.eta = eta
        self.do_not_reload_embeddings = do_not_reload_embeddings
        self.paste_to = None
        self.color_corrections = None
        self.denoising_strength: float = denoising_strength
        self.sampler_noise_scheduler_override = None
        self.ddim_discretize = ddim_discretize or opts.ddim_discretize
        self.s_min_uncond = s_min_uncond or opts.s_min_uncond
        self.s_churn = s_churn or opts.s_churn
        self.s_tmin = s_tmin or opts.s_tmin
        self.s_tmax = s_tmax or float(
            "inf"
        )  # not representable as a standard ui option
        self.s_noise = s_noise or opts.s_noise
        self.override_settings = {
            k: v
            for k, v in (override_settings or {}).items()
            if k not in shared.restricted_opts
        }
        self.override_settings_restore_afterwards = override_settings_restore_afterwards
        self.is_using_inpainting_conditioning = False
        self.disable_extra_networks = False
        self.token_merging_ratio = 0
        self.token_merging_ratio_hr = 0

        if not seed_enable_extras:
            self.subseed = -1
            self.subseed_strength = 0
            self.seed_resize_from_h = 0
            self.seed_resize_from_w = 0

        self.scripts = None
        self.script_args = script_args
        self.all_prompts = None
        self.all_negative_prompts = None
        self.all_seeds = None
        self.all_subseeds = None
        self.iteration = 0
        self.is_hr_pass = False
        self.sampler = None

        self.prompts = None
        self.negative_prompts = None
        self.extra_network_data = None
        self.seeds = None
        self.subseeds = None

        self.step_multiplier = 1
        self.cached_uc = StableDiffusionProcessing.cached_uc
        self.cached_c = StableDiffusionProcessing.cached_c
        self.uc = None
        self.c = None

    @property
    def sd_model(self) -> Any:
        pass

    def txt2img_image_conditioning(self, x, width=None, height=None) -> Any:
        pass

    def depth2img_image_conditioning(self, source_image) -> Any:
        pass

    def edit_image_conditioning(self, source_image) -> Any:
        pass

    def unclip_image_conditioning(self, source_image) -> Any:
        pass

    def inpainting_image_conditioning(
        self, source_image, latent_image, image_mask=None
    ) -> Any:
        pass

    def img2img_image_conditioning(
        self, source_image, latent_image, image_mask=None
    ) -> Any:
        pass

    def init(self, all_prompts, all_seeds, all_subseeds):
        pass

    def sample(
        self,
        conditioning,
        unconditional_conditioning,
        seeds,
        subseeds,
        subseed_strength,
        prompts,
    ):
        pass

    def close(self):
        pass

    def get_token_merging_ratio(self, for_hr=False) -> int | Any:
        pass

    def setup_prompts(self):
        pass

    def get_conds_with_caching(
        self, function, required_prompts, steps, caches, extra_network_data
    ) -> list[Any]:
        """
        Returns the result of calling function(shared.sd_model, required_prompts, steps)
        using a cache to store the result if the same arguments have been used before.

        cache is an array containing two elements. The first element is a tuple
        representing the previously used arguments, or None if no arguments
        have been used before. The second element is where the previously
        computed result is stored.

        caches is a list with items described above.
        """
        pass

    def setup_conds(self):
        pass

    def parse_extra_network_prompts(self):
        pass
