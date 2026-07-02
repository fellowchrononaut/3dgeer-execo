/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>
#include <cstdint>
#include <cuda_runtime.h>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* viewmatrix,
			const float* mirror_transformed_tan_theta,
			const float* mirror_transformed_tan_phi,
			const float* tan_theta,
			const float* tan_phi,
			const float* cam_pos,
			const float focal_x, float focal_y, 
			const float principal_x, float principal_y,
			const float* distortion_coeffs,
			const float* raymap,
			float* xmap,
			float* ymap,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* kernel_times,
			float* out_color,
			float* depth,
			float* out_median_depth,
			int*   out_gidx,
			bool antialiasing,
			int mode,
			int* radii = nullptr,
			int* ranges = nullptr,
			float near_threshold = 0.2f,
			bool debug = false,
			int asso_mode = 0);

		// Session E2: exact ray-integrated occupancy at arbitrary query points.
		// Reconstructs GeometryState/BinningState/ImageState from the buffers a
		// prior forward() call produced (SAME pattern as backward() below), then
		// launches FORWARD::integrate. Forward-only: no gradients, no state
		// mutation of the buffers.
		static void integratePoints(
			const int P,
			const int width, int height,
			const int mode,
			const float focal_x, float focal_y,
			const float* tan_theta,
			const float* tan_phi,
			const float* raymap,
			char* geom_buffer,
			const int R,
			char* binning_buffer,
			char* img_buffer,
			const int Q,
			const int* q_pix_id,
			const float* q_tval,
			const uint2* q_ranges,
			const uint32_t* q_point_order,
			float* out_alpha_integrated,
			bool debug = false);

		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* tan_theta,
			const float* tan_phi,
			const float* raymap,
			const int mode,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* viewmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			const float* dL_invdepths,
			float* dL_dmean2D,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dinvdepth,
			float* dL_dmean3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			float* dL_dsigmaInv,
			bool antialiasing,
			bool debug);
	};
};

#endif
