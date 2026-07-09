#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = "./output/"
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._camera_model = "FISHEYE" #FISHEYE/PINHOLE
        self.dataset = "AUTO" #AUTO/COLMAP/BLENDER/SCANNETPP
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cpu" #"cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01 
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 300
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 0.1
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        # --- GaussianWrapping surface-alignment (GUTWrap Path A) ---
        # Schedule follows the official GW config (configs/normal_field/default.yaml):
        # normal field trains AFTER standard densification ends; wrapping runs as a
        # few discrete rounds on the top error-quantile only.
        self.normal_lr = 0.001
        self.normal_weight = 0.05            # L_N weight
        self.depth_normal_weight = 0.05      # L_DN weight; 0.0 disables the 3rd render pass
        self.normal_from_iter = 20000        # L_N/L_DN start (GW: 20_001)
        self.normal_error_threshold = 0.1    # floor on mean per-pixel error for wrapping
        self.densify_and_wrap_from_iter = 22000   # GW: 22_000
        self.densify_and_wrap_until_iter = 26000  # GW: 26_000
        self.densify_and_wrap_interval = 1000     # GW: every 1_000
        self.wrap_quantile = 0.05            # clone only the top 5% mean-error Gaussians (GW)
        # --- GaussianWrapping multiview NCC+geo consistency (Session F2) ---
        # ported/adapted from GaussianWrapping regularization/multiview_gggs.py
        # config defaults; generalized to PH+EQ via a new CUDA kernel
        # (submodules/multiview_ncc/) instead of GW's pinhole-only
        # warp_patch_ncc (see GUTWrap_Discussion/3DGEERGW_EXECUTION.md Session
        # F2 for the full rationale). All default OFF / bit-identical to
        # pre-Session-F2 behavior when --multiview is not passed.
        self.multiview = False                    # master flag, default OFF
        self.multiview_from_iter = 7000            # GW: start_multiview in [7000,15000]
        self.multiview_ncc_weight = 0.6            # GW: multi_view_ncc_weight
        self.multiview_geo_weight = 0.02           # GW: multi_view_geo_weight
        self.multiview_patch_size = 3              # GW: multi_view_patch_size (radius; 7x7 patch)
        self.multiview_pixel_noise_th = 1.0        # GW: multi_view_pixel_noise_th
        self.multiview_max_angle = 30.0            # GW: multi_view_max_angle (degrees)
        self.multiview_num = 8                     # GW: multi_view_num (neighbor cameras)
        self.multiview_min_dis_relative = 0.002    # GW: multi_view_min_dis_relative
        self.multiview_max_dis_relative = 0.3      # GW: multi_view_max_dis_relative
        self.multiview_znear_relative = 0.02       # relative to scene radius (GW uses znear_relative similarly)
        # Reference-side depth source for the multiview losses (the 2026-07-07/08
        # depth-gradient saga, kept selectable for A/B isolation):
        #   "median"   -- the rasterizer's natively differentiable median depth
        #                 (Session G implicit backward + opacity relief valve;
        #                 GW-faithful production default).
        #   "st"       -- straight-through surrogate on a DETACHED median
        #                 (fix2/fix3 method, exact value + crossing-gaussian
        #                 position gradient only; no opacity path).
        #   "invdepth" -- 1/expected_invdepth (fix1 method; NOT a surface
        #                 depth, p50 19% off median -- kept only to reproduce
        #                 that failure, do not use for real runs).
        self.multiview_ref_depth = "median"
        # --- F2 completion: flatten loss, gaussian cap, GW feature LRs ---
        self.flatten_weight = 0.0            # GW gaussian_flattening_loss weight; 0.0 = OFF
        self.flatten_from_iter = 7000        # flatten loss active from this iter (when flatten_weight > 0)
        self.max_gaussians = 0               # cap on gaussian count; 0 = off (no growth gating)
        self.feature_dc_lr = 0.0             # override for f_dc param-group LR; 0.0 = use feature_lr
        self.feature_rest_lr = 0.0           # override for f_rest param-group LR; 0.0 = use feature_lr/20
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
