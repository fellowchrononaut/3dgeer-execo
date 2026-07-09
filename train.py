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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, match_mask_to_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
import cv2
from utils.ray_normals import depth_to_normals_via_rays, view_normals_to_world, world_to_view_normals, get_ray_dirs_view
from gaussian_renderer import render_normal_field
from utils.multiview import compute_nearest_cameras, geo_loss, eq_warp_patch_ncc, ref_to_neighbor_RT, straight_through_median_depth, expected_depth_from_invdepth
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, fov_mod, sample_step, mask_path, sibr_mask_refcam=None,
             render_model='BEAP', raymap_path=None, focal_scaling=1.0, distortion_scaling=1.0, mirror_shift=0.0):
    os.makedirs('tmp', exist_ok=True)
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    dataset.fov_mod = fov_mod
    dataset.sample_step = sample_step
    dataset.render_model = render_model
    dataset.focal_scaling = focal_scaling
    dataset.distortion_scaling = distortion_scaling
    dataset.mirror_shift = mirror_shift
    dataset.raymap = None
    if raymap_path is not None and os.path.exists(raymap_path):
        dataset.raymap = np.load(raymap_path)
    scene = Scene(dataset, gaussians, shuffle=False)

    # --- GaussianWrapping multiview NCC+geo consistency (Session F2) ---
    # Default OFF (opt.multiview=False); one-time nearest-camera precompute,
    # pure extrinsics geometry (see utils/multiview.py::compute_nearest_cameras).
    multiview_nearest_cameras = None
    if opt.multiview:
        multiview_nearest_cameras = compute_nearest_cameras(
            scene.getTrainCameras(),
            scene_radius=scene.cameras_extent,
            multi_view_max_angle=opt.multiview_max_angle,
            multi_view_min_dis_relative=opt.multiview_min_dis_relative,
            multi_view_max_dis_relative=opt.multiview_max_dis_relative,
            multi_view_num=opt.multiview_num,
        )

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    ema_LN_for_log = 0.0
    ema_LDN_for_log = 0.0
    ema_ncc_for_log = 0.0
    ema_geo_for_log = 0.0
    ema_flatten_for_log = 0.0

    # Pre-compute viewer extra params so MiniCam uses the correct render mode.
    _render_model_map = {"BEAP": 0, "KB": 1, "EQ": 1, "PH": 2}
    render_model_int = _render_model_map.get(render_model, 0)
    cam_extra_params: dict = {}
    if render_model in ("KB", "EQ", "PH"):
        train_cams = scene.getTrainCameras()
        if train_cams:
            ref_cam = train_cams[0]
            cam_extra_params["focal_x"] = ref_cam.focal_x
            cam_extra_params["focal_y"] = ref_cam.focal_y
            cam_extra_params["principal_x"] = ref_cam.principal_x
            cam_extra_params["principal_y"] = ref_cam.principal_y
            if render_model in ("KB", "EQ"):
                cam_extra_params["distortion_coeffs"] = ref_cam.distortion_coeffs
                cam_extra_params["raymap"] = ref_cam.raymap

    print("mask_path:", mask_path)
    valid_mask = None
    if mask_path is not None:
        valid_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        valid_mask = np.repeat(valid_mask[None, ...], 3, axis=0)
        valid_mask = torch.tensor(valid_mask)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                extra_params = {"sample_step": sample_step, "render_model_int": render_model_int, **cam_extra_params}
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, width, height = network_gui.receive(extra_params)
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    if sibr_mask_refcam is not None:
                        print("Applying SIBR mask to network image {}".format(sibr_mask_refcam))
                        net_mask = custom_cam.get_viewpoint_mask(sibr_mask_refcam)
                        net_mask = torch.tensor(np.repeat(net_mask[None, ...], 3, axis=0))
                        net_image[net_mask == 0] = 0.0
                    net_image = torch.nn.functional.interpolate(net_image[None, ...], (height, width), mode='bilinear')[0]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                print(e)
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        if valid_mask is not None:
            image[match_mask_to_image(valid_mask, image) == 0] = 0.0
        # Loss
        gt_image = viewpoint_cam.sampled_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        if iteration % 500 == 0:
            sv = image.permute(1,2,0).detach().cpu().numpy()
            sv = np.clip(sv, 0.0, 1.0)
            sv = (sv * 255).astype(np.uint8)
            cv2.imwrite(f'./tmp/tmp_{iteration:06d}.png', sv[:,:,[2,1,0]])

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        # --- GW surface-alignment losses ---
        median_depth = render_pkg["median_depth"]                # (1,H,W), differentiable (Session G)
        L_N = torch.zeros((), device="cuda"); L_DN = torch.zeros((), device="cuda")
        shape_map = None
        target_n_world = None
        normals_active = iteration >= opt.normal_from_iter and opt.normal_weight > 0
        multiview_active = opt.multiview and iteration >= opt.multiview_from_iter
        # L_DN co-activates with multiview, exactly as GW/GGGS couple them:
        # one reg_kick_on gate (regularization_from_iter=7000) turns on BOTH
        # lambda_depth_normal=0.05 and the patchmatch losses (GGGS train.py:159-179,
        # GW train.py:773-774; the published-run script overrides neither).
        # Running multiview alone 7k-20k grew needle spikes + transparency
        # holes (sessF2_mv_r1_fix2 post-mortem).
        dn_active = opt.depth_normal_weight > 0 and (normals_active or multiview_active)
        if normals_active or dn_active:
            with torch.no_grad():
                target_n_view, valid = depth_to_normals_via_rays(viewpoint_cam, median_depth)
                target_n_world = view_normals_to_world(viewpoint_cam, target_n_view)
        if dn_active:
            shape_map = render_normal_field(viewpoint_cam, gaussians, pipe, shape=True)
            cos_DN = (shape_map * target_n_world).sum(dim=0)
            if valid.any():
                L_DN = (1.0 - cos_DN[valid]).mean()
                loss = loss + opt.depth_normal_weight * L_DN
        if normals_active:
            normal_map = render_normal_field(viewpoint_cam, gaussians, pipe)   # (3,H,W) world
            cos_N = (normal_map * target_n_world).sum(dim=0)
            if valid.any():
                L_N = (1.0 - cos_N[valid]).mean()
                loss = loss + opt.normal_weight * L_N
            # online per-Gaussian error for wrapping densification
            with torch.no_grad():
                gidx = render_pkg["gidx"]                        # (H,W) int32, -1 = none
                err = (1.0 - cos_N.detach()).clamp(min=0) * valid.float()
                sel = gidx >= 0
                if sel.any():
                    ids = gidx[sel].long().flatten()
                    gaussians.normal_error_accum.index_add_(0, ids, err[sel].flatten())
                    gaussians.normal_error_count.index_add_(0, ids, torch.ones_like(ids, dtype=torch.float))

        # --- GaussianWrapping multiview NCC+geo consistency (Session F2) ---
        # Uses SHAPE normals (render_normal_field(..., shape=True)), decoupled
        # from L_N's learned-normal/normal_from_iter schedule so multiview can
        # be active as early as iter 7000 regardless of when L_N/L_DN start.
        # Reference-side depth is the rasterizer's natively differentiable
        # median (Session G implicit-function backward with the opacity
        # relief valve — NOT 1/expected_invdepth, which is no surface
        # depth and collapsed training; see sessF2_mv_r1_fix post-mortem).
        # Neighbor-side sampled depth stays the no-grad median-depth
        # measurement. If the L_DN pass above already rendered a shape map
        # this iteration, reuse it instead of rendering a second time.
        L_MV = torch.zeros((), device="cuda")
        ncc_loss_val = torch.zeros((), device="cuda")
        geo_loss_val = torch.zeros((), device="cuda")
        if multiview_active:
            nearest_ids = multiview_nearest_cameras.get(vind, {"nearest_id": []})["nearest_id"]
            if len(nearest_ids) > 0:
                neighbor_idx = nearest_ids[randint(0, len(nearest_ids) - 1)]
                neighbor_cam = scene.getTrainCameras()[neighbor_idx]

                neighbor_render_pkg = render(neighbor_cam, gaussians, pipe, background)
                # Reference-side depth per --multiview_ref_depth (see
                # arguments/__init__.py for the three methods' history):
                # "median" = Session G natively differentiable median (implicit
                # backward + opacity relief valve, GW-faithful default).
                if opt.multiview_ref_depth == "median":
                    ref_depth = median_depth
                    ref_depth_valid = median_depth > 0
                elif opt.multiview_ref_depth == "st":
                    # detach() so the surrogate is the ONLY gradient path
                    # (median is natively differentiable since Session G).
                    ref_depth, ref_depth_valid = straight_through_median_depth(
                        median_depth.detach(), render_pkg["gidx"],
                        gaussians.get_xyz, viewpoint_cam.world_view_transform,
                    )
                elif opt.multiview_ref_depth == "invdepth":
                    ref_depth, ref_depth_valid = expected_depth_from_invdepth(render_pkg["depth"])
                else:
                    raise ValueError(f"unknown multiview_ref_depth {opt.multiview_ref_depth!r}")
                geo_loss_val, geo_mask, geo_weights = geo_loss(
                    viewpoint_cam, neighbor_cam, ref_depth, neighbor_render_pkg,
                    ref_depth_valid=ref_depth_valid,
                    pixel_noise_th=opt.multiview_pixel_noise_th,
                    znear_relative=opt.multiview_znear_relative,
                    scene_radius=scene.cameras_extent,
                )

                if geo_mask.any():
                    if shape_map is None:
                        shape_map = render_normal_field(viewpoint_cam, gaussians, pipe, shape=True)
                    normal_map_view = world_to_view_normals(viewpoint_cam, shape_map)  # (3,H,W)
                    Hc, Wc = viewpoint_cam.image_height, viewpoint_cam.image_width
                    ys_mv, xs_mv = torch.meshgrid(
                        torch.arange(Hc, device="cuda"), torch.arange(Wc, device="cuda"), indexing="ij"
                    )
                    valid_idx = geo_mask.view(-1).nonzero(as_tuple=True)[0]

                    depths_sel = ref_depth.view(-1)[valid_idx]
                    normals_sel = normal_map_view.reshape(3, -1)[:, valid_idx].transpose(0, 1).contiguous()
                    pixels_sel = torch.stack(
                        [xs_mv.reshape(-1)[valid_idx], ys_mv.reshape(-1)[valid_idx]], dim=-1
                    ).int()
                    weights_sel = geo_weights.view(-1)[valid_idx]

                    ray_dirs_ref = get_ray_dirs_view(viewpoint_cam).permute(1, 2, 0).contiguous()
                    R_rn, T_rn = ref_to_neighbor_RT(viewpoint_cam, neighbor_cam)
                    image_r = viewpoint_cam.gray_image.squeeze(0).cuda()
                    image_n = neighbor_cam.gray_image.squeeze(0).cuda()

                    ncc, ncc_valid = eq_warp_patch_ncc(
                        depths_sel, normals_sel, pixels_sel, ray_dirs_ref, R_rn, T_rn,
                        image_r, image_n, neighbor_cam.render_model,
                        neighbor_cam.focal_x, neighbor_cam.focal_y,
                        neighbor_cam.principal_x, neighbor_cam.principal_y,
                        opt.multiview_patch_size,
                    )
                    # 1 - correlation, weighted by the same geo-consistency
                    # weights, poor-match rejection (ncc_term>=0.9) -- mirrors
                    # GaussianWrapping's own PatchMatch.__call__ masking
                    # (multiview_gggs.py:291-298).
                    ncc_term = torch.clamp(1.0 - ncc, 0.0, 2.0)
                    ncc_mask = (ncc_term < 0.9) & ncc_valid
                    if ncc_mask.any():
                        ncc_loss_val = (ncc_term * weights_sel)[ncc_mask].mean()

                L_MV = opt.multiview_ncc_weight * ncc_loss_val + opt.multiview_geo_weight * geo_loss_val
                loss = loss + L_MV

        # --- periodic normal/depth dumps (outside the L_N gate so the
        # multiview phase, active from iter 7000, is visually monitorable
        # long before normals start at 20k; dumps the SHAPE normal field —
        # the one multiview regularizes — and adds the learned-normal render
        # once L_N is active) ---
        if iteration % 500 == 0 and (normals_active or multiview_active):
            dump_dir = os.path.join(scene.model_path, "normal_dumps")
            os.makedirs(dump_dir, exist_ok=True)
            if normals_active:
                render_sv = (normal_map.detach() * 0.5 + 0.5).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                render_sv = (render_sv * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(dump_dir, f'iter_{iteration:06d}_render.png'), render_sv[:, :, [2, 1, 0]])
            if target_n_world is None:
                with torch.no_grad():
                    target_n_view, _dump_valid = depth_to_normals_via_rays(viewpoint_cam, median_depth)
                    target_n_world = view_normals_to_world(viewpoint_cam, target_n_view)
            if shape_map is None:
                with torch.no_grad():
                    shape_map = render_normal_field(viewpoint_cam, gaussians, pipe, shape=True)
            shape_sv = (shape_map.detach() * 0.5 + 0.5).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            shape_sv = (shape_sv * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(dump_dir, f'iter_{iteration:06d}_shape.png'), shape_sv[:, :, [2, 1, 0]])
            target_sv = (target_n_world.detach() * 0.5 + 0.5).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            target_sv = (target_sv * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(dump_dir, f'iter_{iteration:06d}_target.png'), target_sv[:, :, [2, 1, 0]])
            depth_sv = median_depth.detach().squeeze(0)
            depth_sv = (depth_sv / depth_sv.max().clamp(min=1e-8)).clamp(0, 1).cpu().numpy()
            depth_sv = (depth_sv * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(dump_dir, f'iter_{iteration:06d}_depth.png'), depth_sv)

        # --- GW gaussian flattening loss (F2 completion; default OFF) ---
        L_flatten = torch.zeros((), device="cuda")
        flatten_active = opt.flatten_weight > 0 and iteration >= opt.flatten_from_iter
        if flatten_active:
            L_flatten = (gaussians.get_scaling.min(dim=-1).values / gaussians.spatial_lr_scale).mean()
            loss = loss + opt.flatten_weight * L_flatten

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            if normals_active:
                ema_LN_for_log = 0.4 * L_N.item() + 0.6 * ema_LN_for_log
            if dn_active:
                ema_LDN_for_log = 0.4 * L_DN.item() + 0.6 * ema_LDN_for_log
            if multiview_active:
                ema_ncc_for_log = 0.4 * ncc_loss_val.item() + 0.6 * ema_ncc_for_log
                ema_geo_for_log = 0.4 * geo_loss_val.item() + 0.6 * ema_geo_for_log
            if flatten_active:
                ema_flatten_for_log = 0.4 * L_flatten.item() + 0.6 * ema_flatten_for_log

            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"}
                if normals_active:
                    postfix["L_N"] = f"{ema_LN_for_log:.{7}f}"
                if dn_active:
                    postfix["L_DN"] = f"{ema_LDN_for_log:.{7}f}"
                if multiview_active:
                    postfix["ncc"] = f"{ema_ncc_for_log:.{7}f}"
                    postfix["geo"] = f"{ema_geo_for_log:.{7}f}"
                if flatten_active:
                    postfix["flat"] = f"{ema_flatten_for_log:.{7}f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer and normals_active:
                tb_writer.add_scalar('train_loss/L_N', L_N.item(), iteration)
            if tb_writer and dn_active:
                tb_writer.add_scalar('train_loss/L_DN', L_DN.item(), iteration)
            if tb_writer and multiview_active:
                tb_writer.add_scalar('train_loss/ncc_loss', ncc_loss_val.item(), iteration)
                tb_writer.add_scalar('train_loss/geo_loss', geo_loss_val.item(), iteration)
                tb_writer.add_scalar('train_loss/L_MV', L_MV.item(), iteration)
            if tb_writer and flatten_active:
                tb_writer.add_scalar('train_loss/L_flatten', L_flatten.item(), iteration)
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), dataset.train_test_exp, valid_mask)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                under_cap = opt.max_gaussians <= 0 or gaussians.get_xyz.shape[0] < opt.max_gaussians
                if under_cap and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Wrapping densification (GW schedule: discrete rounds after normal
            # convergence, decoupled from the standard densification epochs).
            under_cap = opt.max_gaussians <= 0 or gaussians.get_xyz.shape[0] < opt.max_gaussians
            if (under_cap
                    and iteration >= opt.densify_and_wrap_from_iter
                    and iteration <= opt.densify_and_wrap_until_iter
                    and iteration % opt.densify_and_wrap_interval == 0):
                n_wrapped = gaussians.densify_and_wrap(opt.normal_error_threshold, opt.wrap_quantile)
                print("[WRAP] iter {} cloned {}".format(iteration, n_wrapped))
                if tb_writer:
                    tb_writer.add_scalar('wrapping/n_cloned', n_wrapped, iteration)
                gaussians.reset_normal_error_stats()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp, valid_mask):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    if valid_mask is not None:
                        image[match_mask_to_image(valid_mask, image) == 0] = 0.0
                    gt_image = torch.clamp(viewpoint.sampled_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[500, 1200, 2000, 2800, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7_000, 7500, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 18000, 21000, 24000, 27000, 29000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--mask_path", type=str, default = None)
    parser.add_argument("--sibr_mask_refcam", type=str, default = None)
    parser.add_argument("--sample_step", type=float, default=0.002)
    parser.add_argument("--fov_mod", type=float, default = None)
    parser.add_argument("--render_model", type=str, default='BEAP',
                        help="Rendering/training projection mode: BEAP (default), KB, EQ, or PH")
    parser.add_argument("--raymap_path", type=str, default=None,
                        help="Path to a .npy per-pixel ray-direction map (required for KB/EQ training)")
    parser.add_argument("--focal_scaling", type=float, default=1.0,
                        help="Scale factor applied to the focal length (KB/PH modes)")
    parser.add_argument("--distortion_scaling", type=float, default=1.0,
                        help="Scale factor applied to distortion coefficients (KB mode; 0 = EQ)")
    parser.add_argument("--mirror_shift", type=float, default=0.0,
                        help="Mirror-model shift parameter xi for omnidirectional mapping (KB mode)")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, \
             args.fov_mod,  args.sample_step, args.mask_path, args.sibr_mask_refcam, \
             args.render_model, args.raymap_path, args.focal_scaling, args.distortion_scaling, args.mirror_shift)

    # All done
    print("\nTraining complete.")
