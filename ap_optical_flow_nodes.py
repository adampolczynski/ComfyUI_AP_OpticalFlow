import torch
import torch.nn.functional as F
import os
from datetime import datetime

try:
	from torchvision.models.optical_flow import (
		raft_large,
		raft_small,
		Raft_Large_Weights,
		Raft_Small_Weights,
	)
	_RAFT_IMPORT_ERROR = None
except Exception as e:
	_RAFT_IMPORT_ERROR = e

try:
	import comfy.model_management as model_management
except Exception:
	model_management = None

try:
	import folder_paths
except Exception:
	folder_paths = None


_MODEL_CACHE = {}
_INDEXER_STATE = {}
_INDEXER_RUN_BY_NODE = {}


def _get_output_directory():
	if folder_paths is not None:
		try:
			return folder_paths.get_output_directory()
		except Exception:
			pass
	return os.path.join(os.getcwd(), "output")


def _resolve_load_path(path_str):
	path_str = str(path_str).strip()
	if not path_str:
		raise ValueError("file_path is empty")

	if os.path.isabs(path_str) and os.path.exists(path_str):
		return path_str

	candidates = []
	if folder_paths is not None:
		try:
			candidates.append(os.path.join(folder_paths.get_output_directory(), path_str))
		except Exception:
			pass
		try:
			candidates.append(os.path.join(folder_paths.get_input_directory(), path_str))
		except Exception:
			pass

	candidates.append(os.path.join(os.getcwd(), path_str))

	for c in candidates:
		if os.path.exists(c):
			return c

	raise FileNotFoundError(f"Could not resolve flow file path: {path_str}")


def _make_save_path(filename_prefix, overwrite):
	base_dir = _get_output_directory()
	os.makedirs(base_dir, exist_ok=True)

	prefix = str(filename_prefix).strip().replace("\\", "/").lstrip("/")
	if not prefix:
		prefix = "AP_OpticalFlow/flow"

	if not prefix.endswith(".pt"):
		prefix = prefix + ".pt"

	full_path = os.path.normpath(os.path.join(base_dir, prefix))
	os.makedirs(os.path.dirname(full_path), exist_ok=True)

	if overwrite:
		return full_path

	if not os.path.exists(full_path):
		return full_path

	stem, ext = os.path.splitext(full_path)
	tag = datetime.now().strftime("%Y%m%d_%H%M%S")
	idx = 1
	while True:
		candidate = f"{stem}_{tag}_{idx:03d}{ext}"
		if not os.path.exists(candidate):
			return candidate
		idx += 1


def _normalize_flow_data_for_save(flow_data):
	if isinstance(flow_data, dict):
		out = {}
		for k, v in flow_data.items():
			if torch.is_tensor(v):
				out[k] = v.detach().cpu()
			else:
				out[k] = v
		return out

	# Allow saving raw flow tensor as convenience.
	return {"flow_ab": _as_bhw2(flow_data).detach().cpu()}


def _normalize_loaded_flow_data(data):
	if isinstance(data, dict) and "flow_data" in data and isinstance(data["flow_data"], dict):
		data = data["flow_data"]

	if isinstance(data, dict):
		out = {}
		for k, v in data.items():
			if torch.is_tensor(v):
				out[k] = v.detach().cpu().float()
			else:
				out[k] = v
		if "flow_ab" in out:
			out["flow_ab"] = _as_bhw2(out["flow_ab"]).detach().cpu().float()
		if "flow_ba" in out:
			out["flow_ba"] = _as_bhw2(out["flow_ba"]).detach().cpu().float()
		return out

	return {"flow_ab": _as_bhw2(data).detach().cpu().float()}


def _get_device():
	if model_management is not None:
		try:
			device = model_management.get_torch_device()
			return device if isinstance(device, torch.device) else torch.device(device)
		except Exception:
			pass
	if torch.cuda.is_available():
		return torch.device("cuda")
	if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def _ensure_bhwc(tensor):
	if not torch.is_tensor(tensor):
		raise TypeError("Expected torch.Tensor input")
	if tensor.ndim == 3:
		tensor = tensor.unsqueeze(0)
	if tensor.ndim != 4:
		raise ValueError(f"Expected [B,H,W,C], got shape {tuple(tensor.shape)}")
	return tensor.float().clamp(0.0, 1.0)


def _ensure_mask_bhw(mask):
	if not torch.is_tensor(mask):
		raise TypeError("Expected torch.Tensor mask")
	if mask.ndim == 2:
		mask = mask.unsqueeze(0)
	elif mask.ndim == 4 and mask.shape[-1] == 1:
		mask = mask[..., 0]
	elif mask.ndim == 4 and mask.shape[1] == 1:
		mask = mask[:, 0]
	if mask.ndim != 3:
		raise ValueError(f"Expected MASK [B,H,W], got shape {tuple(mask.shape)}")
	return mask.float().clamp(0.0, 1.0)


def _to_bchw(image_bhwc):
	return image_bhwc.permute(0, 3, 1, 2).contiguous()


def _to_bhwc(image_bchw):
	return image_bchw.permute(0, 2, 3, 1).contiguous()


def _match_batch(a, b):
	ba = a.shape[0]
	bb = b.shape[0]
	if ba == bb:
		return a, b
	if ba == 1:
		return a.repeat(bb, *([1] * (a.ndim - 1))), b
	if bb == 1:
		return a, b.repeat(ba, *([1] * (b.ndim - 1)))
	raise ValueError(f"Batch mismatch: {ba} vs {bb}")


def _resize_mask(mask_bhw, target_h, target_w):
	if mask_bhw.shape[1] == target_h and mask_bhw.shape[2] == target_w:
		return mask_bhw
	out = F.interpolate(mask_bhw.unsqueeze(1), size=(target_h, target_w), mode="bilinear", align_corners=False)
	return out.squeeze(1).clamp(0.0, 1.0)


def _to_raft_rgb_bchw(image_bhwc):
	channels = image_bhwc.shape[-1]
	if channels == 1:
		image_bhwc = image_bhwc.repeat(1, 1, 1, 3)
	elif channels >= 3:
		image_bhwc = image_bhwc[..., :3]
	else:
		raise ValueError("Input IMAGE must have at least 1 channel")
	return _to_bchw(image_bhwc)


def _resize_images_for_raft(img1_bchw, img2_bchw, max_side):
	_, _, h, w = img1_bchw.shape
	if max_side <= 0 or max(h, w) <= max_side:
		return img1_bchw, img2_bchw, (h, w), (h, w)

	scale = max_side / float(max(h, w))
	new_h = max(16, int(round(h * scale)))
	new_w = max(16, int(round(w * scale)))
	# RAFT commonly expects dimensions divisible by 8.
	new_h = ((new_h + 7) // 8) * 8
	new_w = ((new_w + 7) // 8) * 8

	i1 = F.interpolate(img1_bchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
	i2 = F.interpolate(img2_bchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
	return i1, i2, (h, w), (new_h, new_w)


def _resize_flow_b2hw(flow_b2hw, target_h, target_w):
	_, _, h, w = flow_b2hw.shape
	if (h, w) == (target_h, target_w):
		return flow_b2hw

	sx = target_w / float(w)
	sy = target_h / float(h)
	out = F.interpolate(flow_b2hw, size=(target_h, target_w), mode="bilinear", align_corners=True)
	out[:, 0] *= sx
	out[:, 1] *= sy
	return out


def _hsv_to_rgb(h, s, v):
	# Vectorized HSV->RGB conversion where h in [0,1].
	def f(n):
		k = (n + h * 6.0) % 6.0
		return v - v * s * torch.clamp(torch.minimum(torch.minimum(k, 4.0 - k), torch.ones_like(k)), 0.0, 1.0)

	r = f(5.0)
	g = f(3.0)
	b = f(1.0)
	return torch.stack([r, g, b], dim=-1)


def _flow_to_color(flow_bhw2, percentile=0.98):
	u = flow_bhw2[..., 0]
	v = flow_bhw2[..., 1]
	magnitude = torch.sqrt(u * u + v * v + 1e-8)
	angle = torch.atan2(v, u)

	flat = magnitude.flatten(1)
	n = flat.shape[1]
	k = max(1, min(n, int(round(n * percentile))))
	denom = torch.kthvalue(flat, k, dim=1).values.view(-1, 1, 1).clamp_min(1e-6)

	value = (magnitude / denom).clamp(0.0, 1.0)
	hue = ((angle + torch.pi) / (2.0 * torch.pi)) % 1.0
	saturation = torch.ones_like(hue)

	return _hsv_to_rgb(hue, saturation, value).clamp(0.0, 1.0)


def _as_bhw2(flow):
	if not torch.is_tensor(flow):
		raise TypeError("Flow must be a torch.Tensor")
	if flow.ndim != 4:
		raise ValueError(f"Expected rank-4 flow, got rank {flow.ndim}")
	if flow.shape[-1] == 2:
		return flow.float()
	if flow.shape[1] == 2:
		return flow.permute(0, 2, 3, 1).contiguous().float()
	raise ValueError(f"Unrecognized flow shape: {tuple(flow.shape)}")


def _pick_flow_from_data(flow_data, direction):
	key = "flow_ab" if direction == "ab" else "flow_ba"
	if isinstance(flow_data, dict):
		if key not in flow_data:
			raise KeyError(f"flow_data missing key: {key}")
		return _as_bhw2(flow_data[key])
	return _as_bhw2(flow_data)


def _select_batch_entry(tensor, index):
	if tensor.shape[0] <= 1:
		return tensor
	idx = int(index) % int(tensor.shape[0])
	return tensor[idx : idx + 1]


def _select_flow_data_by_index(flow_data, frame_index):
	idx = int(frame_index)
	if isinstance(flow_data, dict):
		out = {}
		for k, v in flow_data.items():
			if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] > 1:
				out[k] = _select_batch_entry(v, idx)
			else:
				out[k] = v
		return out

	flow = _as_bhw2(flow_data)
	return _select_batch_entry(flow, idx)


def _align_image_flow_batches(img, flow, batch_mode="auto", current_frame_index=None, flow_skip=0):
	bi = int(img.shape[0])
	bf = int(flow.shape[0])
	offset = max(0, int(flow_skip))
	if bi == bf:
		return img, flow

	if batch_mode == "by_index":
		if bi != 1:
			raise ValueError("batch_mode='by_index' expects IMAGE batch size 1")
		if bf == 1:
			return img, flow
		idx = offset if current_frame_index is None else int(current_frame_index) + offset
		return img, _select_batch_entry(flow, idx)

	if batch_mode == "repeat_image":
		if bi == 1 and bf > 1:
			return img.repeat(bf, *([1] * (img.ndim - 1))), flow
		if bf == 1 and bi > 1:
			return img, flow.repeat(bi, *([1] * (flow.ndim - 1)))
		return _match_batch(img, flow)

	# auto mode: prefer index-based behavior for loop pipelines if index provided,
	# otherwise use safe one-item fallback to avoid accidental image broadcasting.
	if bi == 1 and bf > 1:
		if current_frame_index is not None:
			return img, _select_batch_entry(flow, int(current_frame_index) + offset)
		return img, _select_batch_entry(flow, offset)
	if bf == 1 and bi > 1:
		return img, flow.repeat(bi, *([1] * (flow.ndim - 1)))
	return _match_batch(img, flow)


def _get_raft_model(variant, device, use_fp16):
	if _RAFT_IMPORT_ERROR is not None:
		raise RuntimeError(
			"torchvision RAFT import failed. Install a compatible torchvision build.\n"
			f"Import error: {_RAFT_IMPORT_ERROR}"
		)

	key = (variant, str(device), bool(use_fp16))
	if key in _MODEL_CACHE:
		return _MODEL_CACHE[key]

	if variant == "large":
		weights = Raft_Large_Weights.DEFAULT
		model = raft_large(weights=weights, progress=False)
	else:
		weights = Raft_Small_Weights.DEFAULT
		model = raft_small(weights=weights, progress=False)

	model = model.eval().to(device)
	transforms = weights.transforms()
	_MODEL_CACHE[key] = (model, transforms)
	return model, transforms


def _estimate_flow(model, transforms, img1_bchw, img2_bchw, device, use_fp16):
	img1 = img1_bchw.to(device).contiguous()
	img2 = img2_bchw.to(device).contiguous()
	img1, img2 = transforms(img1, img2)
	img1 = img1.contiguous()
	img2 = img2.contiguous()

	amp_enabled = bool(use_fp16 and device.type == "cuda")
	with torch.no_grad():
		try:
			with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
				preds = model(img1, img2)
		except RuntimeError as e:
			# Some CUDA/cuDNN combinations fail in RAFT internals with non-supported
			# kernels. Retry with cudnn disabled and float32 contiguous tensors.
			if "CUDNN_STATUS_NOT_SUPPORTED" not in str(e):
				raise
			img1_f32 = img1.float().contiguous()
			img2_f32 = img2.float().contiguous()
			with torch.backends.cudnn.flags(enabled=False):
				preds = model(img1_f32, img2_f32)

	flow = preds[-1] if isinstance(preds, (list, tuple)) else preds
	return flow.float()


def _warp_with_flow(img_bchw, flow_bhw2, interpolation="bilinear", padding_mode="border"):
	b, _, h, w = img_bchw.shape
	device = img_bchw.device
	dtype = img_bchw.dtype

	yy, xx = torch.meshgrid(
		torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
		torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
		indexing="ij",
	)
	base = torch.stack((xx, yy), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)

	flow = flow_bhw2.to(device=device, dtype=dtype)
	flow_norm = torch.empty_like(flow)
	flow_norm[..., 0] = (2.0 * flow[..., 0]) / max(w - 1, 1)
	flow_norm[..., 1] = (2.0 * flow[..., 1]) / max(h - 1, 1)

	grid = base + flow_norm
	warped = F.grid_sample(
		img_bchw,
		grid,
		mode=interpolation,
		padding_mode=padding_mode,
		align_corners=True,
	)

	valid = (
		(grid[..., 0] >= -1.0)
		& (grid[..., 0] <= 1.0)
		& (grid[..., 1] >= -1.0)
		& (grid[..., 1] <= 1.0)
	).float()
	return warped, valid


class APGetRAFTOpticalFlow:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"image_a": ("IMAGE",),
				"image_b": ("IMAGE",),
				"model_size": (["small", "large"], {"default": "large"}),
				"compute_backward": ("BOOLEAN", {"default": True}),
				"max_side": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 8}),
				"use_fp16": ("BOOLEAN", {"default": True}),
			}
		}

	RETURN_TYPES = ("IMAGE", "AP_FLOW")
	RETURN_NAMES = ("flow_visualization", "flow_data")
	FUNCTION = "compute"
	CATEGORY = "AP_OpticalFlow"

	def compute(self, image_a, image_b, model_size="large", compute_backward=True, max_side=1024, use_fp16=True):
		a = _ensure_bhwc(image_a)
		b = _ensure_bhwc(image_b)

		if b.shape[1] != a.shape[1] or b.shape[2] != a.shape[2]:
			b_bchw = _to_bchw(b)
			b_bchw = F.interpolate(b_bchw, size=(a.shape[1], a.shape[2]), mode="bilinear", align_corners=False)
			b = _to_bhwc(b_bchw)

		a, b = _match_batch(a, b)
		h0, w0 = a.shape[1], a.shape[2]

		a_raft = _to_raft_rgb_bchw(a)
		b_raft = _to_raft_rgb_bchw(b)
		a_raft, b_raft, _, _ = _resize_images_for_raft(a_raft, b_raft, int(max_side))

		device = _get_device()
		model, transforms = _get_raft_model(model_size, device, use_fp16)

		flow_ab = _estimate_flow(model, transforms, a_raft, b_raft, device, use_fp16)
		if compute_backward:
			flow_ba = _estimate_flow(model, transforms, b_raft, a_raft, device, use_fp16)
		else:
			flow_ba = -flow_ab

		flow_ab = _resize_flow_b2hw(flow_ab, h0, w0)
		flow_ba = _resize_flow_b2hw(flow_ba, h0, w0)

		flow_ab_bhw2 = flow_ab.permute(0, 2, 3, 1).contiguous().cpu()
		flow_ba_bhw2 = flow_ba.permute(0, 2, 3, 1).contiguous().cpu()

		flow_vis = _flow_to_color(flow_ab_bhw2)
		flow_data = {
			"flow_ab": flow_ab_bhw2,
			"flow_ba": flow_ba_bhw2,
			"height": h0,
			"width": w0,
			"model": model_size,
		}

		return flow_vis, flow_data


class APApplyRAFTOpticalFlow:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"images": ("IMAGE",),
				"flow_data": ("AP_FLOW",),
				"flow_direction": (["ab", "ba"], {"default": "ab"}),
				"batch_mode": (["auto", "by_index", "repeat_image"], {"default": "auto"}),
				"flow_skip": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
				"frames_skip": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
				"strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.01}),
				"invert_flow": ("BOOLEAN", {"default": False}),
				"interpolation": (["bilinear", "nearest", "bicubic"], {"default": "bilinear"}),
				"padding_mode": (["border", "zeros", "reflection"], {"default": "border"}),
			},
			"optional": {
				"current_frame_index": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
			},
		}

	RETURN_TYPES = ("IMAGE", "MASK")
	RETURN_NAMES = ("warped_images", "valid_mask")
	FUNCTION = "apply"
	CATEGORY = "AP_OpticalFlow"

	def apply(
		self,
		images,
		flow_data,
		flow_direction="ab",
		batch_mode="auto",
		flow_skip=0,
		frames_skip=0,
		strength=1.0,
		invert_flow=False,
		interpolation="bilinear",
		padding_mode="border",
		current_frame_index=None,
	):
		img = _ensure_bhwc(images)

		if current_frame_index is not None and int(current_frame_index) < max(0, int(frames_skip)):
			valid = torch.ones((img.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype, device=img.device)
			return img.cpu(), valid.cpu()

		flow = _pick_flow_from_data(flow_data, flow_direction)
		img, flow = _align_image_flow_batches(
			img,
			flow,
			batch_mode=batch_mode,
			current_frame_index=current_frame_index,
			flow_skip=flow_skip,
		)

		h, w = img.shape[1], img.shape[2]
		if flow.shape[1] != h or flow.shape[2] != w:
			flow_b2hw = flow.permute(0, 3, 1, 2).contiguous()
			flow_b2hw = _resize_flow_b2hw(flow_b2hw, h, w)
			flow = flow_b2hw.permute(0, 2, 3, 1).contiguous()

		if invert_flow:
			flow = -flow
		flow = flow * float(strength)

		device = _get_device()
		warped_bchw, valid = _warp_with_flow(
			_to_bchw(img).to(device),
			flow.to(device),
			interpolation=interpolation,
			padding_mode=padding_mode,
		)

		warped = _to_bhwc(warped_bchw).clamp(0.0, 1.0).cpu()
		valid = valid.clamp(0.0, 1.0).cpu()
		return warped, valid


class APFlowOcclusionMask:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"flow_data": ("AP_FLOW",),
				"flow_direction": (["ab", "ba"], {"default": "ab"}),
				"batch_mode": (["auto", "by_index"], {"default": "auto"}),
				"flow_skip": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
				"frames_skip": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
				"abs_epsilon": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 50.0, "step": 0.01}),
				"rel_epsilon": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.001}),
				"dilate_occlusion": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1}),
			},
			"optional": {
				"current_frame_index": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
			},
		}

	RETURN_TYPES = ("MASK", "MASK", "IMAGE")
	RETURN_NAMES = ("valid_mask", "occlusion_mask", "confidence_visualization")
	FUNCTION = "compute"
	CATEGORY = "AP_OpticalFlow"

	def compute(
		self,
		flow_data,
		flow_direction="ab",
		batch_mode="auto",
		flow_skip=0,
		frames_skip=0,
		abs_epsilon=1.0,
		rel_epsilon=0.05,
		dilate_occlusion=0,
		current_frame_index=None,
	):
		if current_frame_index is not None and int(current_frame_index) < max(0, int(frames_skip)):
			if isinstance(flow_data, dict):
				probe = flow_data.get("flow_ab", None)
				if probe is None:
					probe = flow_data.get("flow_ba", None)
				if probe is None:
					raise ValueError("flow_data must contain flow_ab or flow_ba")
				probe = _as_bhw2(probe)
				b, h, w = probe.shape[0], probe.shape[1], probe.shape[2]
			else:
				probe = _as_bhw2(flow_data)
				b, h, w = probe.shape[0], probe.shape[1], probe.shape[2]

			valid = torch.ones((b, h, w), dtype=probe.dtype)
			occ = torch.zeros((b, h, w), dtype=probe.dtype)
			conf = torch.ones((b, h, w, 3), dtype=probe.dtype)
			return valid.cpu(), occ.cpu(), conf.cpu()

		if not isinstance(flow_data, dict):
			raise TypeError("APFlowOcclusionMask expects AP_FLOW dict from APGetRAFTOpticalFlow")

		if batch_mode == "by_index" or (batch_mode == "auto" and current_frame_index is not None):
			idx = (0 if current_frame_index is None else int(current_frame_index)) + max(0, int(flow_skip))
			flow_data = _select_flow_data_by_index(flow_data, idx)

		fwd = _pick_flow_from_data(flow_data, flow_direction)
		bwd = _pick_flow_from_data(flow_data, "ba" if flow_direction == "ab" else "ab")
		fwd, bwd = _align_image_flow_batches(fwd, bwd, batch_mode="auto", current_frame_index=current_frame_index)

		h, w = fwd.shape[1], fwd.shape[2]
		if bwd.shape[1] != h or bwd.shape[2] != w:
			bwd_b2hw = bwd.permute(0, 3, 1, 2).contiguous()
			bwd_b2hw = _resize_flow_b2hw(bwd_b2hw, h, w)
			bwd = bwd_b2hw.permute(0, 2, 3, 1).contiguous()

		device = _get_device()
		fwd_d = fwd.to(device)
		bwd_d = bwd.to(device)

		bwd_b2hw = bwd_d.permute(0, 3, 1, 2).contiguous()
		warped_bwd_b2hw, grid_valid = _warp_with_flow(bwd_b2hw, fwd_d, interpolation="bilinear", padding_mode="zeros")
		warped_bwd = warped_bwd_b2hw.permute(0, 2, 3, 1).contiguous()

		cycle = fwd_d + warped_bwd
		cycle_err = torch.sqrt(torch.sum(cycle * cycle, dim=-1) + 1e-8)
		fwd_mag = torch.sqrt(torch.sum(fwd_d * fwd_d, dim=-1) + 1e-8)
		bwd_mag = torch.sqrt(torch.sum(warped_bwd * warped_bwd, dim=-1) + 1e-8)

		threshold = abs_epsilon + rel_epsilon * (fwd_mag + bwd_mag)
		consistent = (cycle_err <= threshold).float() * grid_valid
		occlusion = (1.0 - consistent).clamp(0.0, 1.0)

		if dilate_occlusion > 0:
			k = int(dilate_occlusion) * 2 + 1
			occlusion = F.max_pool2d(occlusion.unsqueeze(1), kernel_size=k, stride=1, padding=dilate_occlusion).squeeze(1)
			consistent = (1.0 - occlusion).clamp(0.0, 1.0) * grid_valid

		confidence = torch.exp(-cycle_err / (threshold + 1e-6)) * grid_valid
		conf_vis = confidence.unsqueeze(-1).repeat(1, 1, 1, 3).clamp(0.0, 1.0)

		return consistent.cpu(), occlusion.cpu(), conf_vis.cpu()


class APApplyRAFTOpticalFlowMasked:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"images": ("IMAGE",),
				"mask": ("MASK",),
				"flow_data": ("AP_FLOW",),
				"flow_direction": (["ab", "ba"], {"default": "ab"}),
				"batch_mode": (["auto", "by_index", "repeat_image"], {"default": "auto"}),
				"flow_skip": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
				"frames_skip": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
				"strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.01}),
				"invert_flow": ("BOOLEAN", {"default": False}),
				"invert_mask": ("BOOLEAN", {"default": False}),
				"mask_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
				"mask_feather": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
				"interpolation": (["bilinear", "nearest", "bicubic"], {"default": "bilinear"}),
				"padding_mode": (["border", "zeros", "reflection"], {"default": "border"}),
			},
			"optional": {
				"current_frame_index": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
			},
		}

	RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK")
	RETURN_NAMES = ("output_images", "warped_images", "warped_mask", "valid_mask")
	FUNCTION = "apply_masked"
	CATEGORY = "AP_OpticalFlow"

	def apply_masked(
		self,
		images,
		mask,
		flow_data,
		flow_direction="ab",
		batch_mode="auto",
		flow_skip=0,
		frames_skip=0,
		strength=1.0,
		invert_flow=False,
		invert_mask=False,
		mask_strength=1.0,
		mask_feather=0,
		interpolation="bilinear",
		padding_mode="border",
		current_frame_index=None,
	):
		img = _ensure_bhwc(images)
		m = _ensure_mask_bhw(mask)

		if current_frame_index is not None and int(current_frame_index) < max(0, int(frames_skip)):
			h, w = img.shape[1], img.shape[2]
			m = _resize_mask(m, h, w)
			valid = torch.ones((img.shape[0], h, w), dtype=img.dtype, device=img.device)
			return img.cpu(), img.cpu(), m.cpu(), valid.cpu()

		flow = _pick_flow_from_data(flow_data, flow_direction)

		img, flow = _align_image_flow_batches(
			img,
			flow,
			batch_mode=batch_mode,
			current_frame_index=current_frame_index,
			flow_skip=flow_skip,
		)
		img, m = _match_batch(img, m)

		h, w = img.shape[1], img.shape[2]
		if flow.shape[1] != h or flow.shape[2] != w:
			flow_b2hw = flow.permute(0, 3, 1, 2).contiguous()
			flow_b2hw = _resize_flow_b2hw(flow_b2hw, h, w)
			flow = flow_b2hw.permute(0, 2, 3, 1).contiguous()
		m = _resize_mask(m, h, w)

		if invert_flow:
			flow = -flow
		flow = flow * float(strength)

		if invert_mask:
			m = 1.0 - m

		if mask_feather > 0:
			k = int(mask_feather) * 2 + 1
			m = F.avg_pool2d(m.unsqueeze(1), kernel_size=k, stride=1, padding=mask_feather).squeeze(1)

		m = (m * float(mask_strength)).clamp(0.0, 1.0)

		device = _get_device()
		img_d = img.to(device)
		flow_d = flow.to(device)
		m_d = m.to(device)

		warped_bchw, valid = _warp_with_flow(
			_to_bchw(img_d),
			flow_d,
			interpolation=interpolation,
			padding_mode=padding_mode,
		)
		warped = _to_bhwc(warped_bchw).clamp(0.0, 1.0)

		warped_mask_bchw, _ = _warp_with_flow(
			m_d.unsqueeze(1),
			flow_d,
			interpolation="bilinear",
			padding_mode="zeros",
		)
		warped_mask = warped_mask_bchw.squeeze(1).clamp(0.0, 1.0)

		alpha = (warped_mask * m_d).clamp(0.0, 1.0).unsqueeze(-1)
		out = (warped * alpha) + (img_d * (1.0 - alpha))
		out = out.clamp(0.0, 1.0)

		valid_applied = (valid * alpha.squeeze(-1)).clamp(0.0, 1.0)
		return out.cpu(), warped.cpu(), warped_mask.cpu(), valid_applied.cpu()


class APWarpImageAndMaskByRAFTFlow:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"images": ("IMAGE",),
				"mask": ("MASK",),
				"flow_data": ("AP_FLOW",),
				"flow_direction": (["ab", "ba"], {"default": "ab"}),
				"batch_mode": (["auto", "by_index", "repeat_image"], {"default": "auto"}),
				"flow_skip": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
				"frames_skip": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
				"strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.01}),
				"invert_flow": ("BOOLEAN", {"default": False}),
				"mask_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
				"interpolation": (["bilinear", "nearest", "bicubic"], {"default": "bilinear"}),
				"padding_mode": (["border", "zeros", "reflection"], {"default": "border"}),
			},
			"optional": {
				"current_frame_index": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
			},
		}

	RETURN_TYPES = ("IMAGE", "MASK", "MASK")
	RETURN_NAMES = ("warped_image", "warped_mask", "valid_mask")
	FUNCTION = "warp_pair"
	CATEGORY = "AP_OpticalFlow"

	def warp_pair(
		self,
		images,
		mask,
		flow_data,
		flow_direction="ab",
		batch_mode="auto",
		flow_skip=0,
		frames_skip=0,
		strength=1.0,
		invert_flow=False,
		mask_threshold=0.0,
		interpolation="bilinear",
		padding_mode="border",
		current_frame_index=None,
	):
		img = _ensure_bhwc(images)
		m = _ensure_mask_bhw(mask)

		if current_frame_index is not None and int(current_frame_index) < max(0, int(frames_skip)):
			h, w = img.shape[1], img.shape[2]
			m = _resize_mask(m, h, w)
			if mask_threshold > 0.0:
				m = (m >= float(mask_threshold)).float()
			valid = torch.ones((img.shape[0], h, w), dtype=img.dtype, device=img.device)
			return img.cpu(), m.cpu(), valid.cpu()

		flow = _pick_flow_from_data(flow_data, flow_direction)

		img, flow = _align_image_flow_batches(
			img,
			flow,
			batch_mode=batch_mode,
			current_frame_index=current_frame_index,
			flow_skip=flow_skip,
		)
		img, m = _match_batch(img, m)

		h, w = img.shape[1], img.shape[2]
		if flow.shape[1] != h or flow.shape[2] != w:
			flow_b2hw = flow.permute(0, 3, 1, 2).contiguous()
			flow_b2hw = _resize_flow_b2hw(flow_b2hw, h, w)
			flow = flow_b2hw.permute(0, 2, 3, 1).contiguous()
		m = _resize_mask(m, h, w)

		if invert_flow:
			flow = -flow
		flow = flow * float(strength)

		device = _get_device()
		img_d = img.to(device)
		m_d = m.to(device)
		flow_d = flow.to(device)

		warped_img_bchw, valid = _warp_with_flow(
			_to_bchw(img_d),
			flow_d,
			interpolation=interpolation,
			padding_mode=padding_mode,
		)
		warped_mask_bchw, _ = _warp_with_flow(
			m_d.unsqueeze(1),
			flow_d,
			interpolation="bilinear",
			padding_mode="zeros",
		)

		warped_img = _to_bhwc(warped_img_bchw).clamp(0.0, 1.0)
		warped_mask = warped_mask_bchw.squeeze(1).clamp(0.0, 1.0)

		if mask_threshold > 0.0:
			warped_mask = (warped_mask >= float(mask_threshold)).float()

		return warped_img.cpu(), warped_mask.cpu(), valid.clamp(0.0, 1.0).cpu()


class APFlowComposite:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"original_images": ("IMAGE",),
				"warped_images": ("IMAGE",),
				"valid_mask": ("MASK",),
				"occlusion_mask": ("MASK",),
				"blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
				"feather": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
				"invert_occlusion": ("BOOLEAN", {"default": False}),
			}
		}

	RETURN_TYPES = ("IMAGE", "MASK")
	RETURN_NAMES = ("composited_images", "composite_alpha")
	FUNCTION = "composite"
	CATEGORY = "AP_OpticalFlow"

	def composite(
		self,
		original_images,
		warped_images,
		valid_mask,
		occlusion_mask,
		blend_strength=1.0,
		feather=0,
		invert_occlusion=False,
	):
		orig = _ensure_bhwc(original_images)
		warp = _ensure_bhwc(warped_images)
		vm = _ensure_mask_bhw(valid_mask)
		occ = _ensure_mask_bhw(occlusion_mask)

		orig, warp = _match_batch(orig, warp)
		orig, vm = _match_batch(orig, vm)
		orig, occ = _match_batch(orig, occ)

		h, w = orig.shape[1], orig.shape[2]
		vm = _resize_mask(vm, h, w)
		occ = _resize_mask(occ, h, w)

		if invert_occlusion:
			occ = 1.0 - occ

		alpha = (vm * (1.0 - occ)).clamp(0.0, 1.0)
		if feather > 0:
			k = int(feather) * 2 + 1
			alpha = F.avg_pool2d(alpha.unsqueeze(1), kernel_size=k, stride=1, padding=feather).squeeze(1)

		alpha = (alpha * float(blend_strength)).clamp(0.0, 1.0)

		alpha_img = alpha.unsqueeze(-1)
		out = (warp * alpha_img) + (orig * (1.0 - alpha_img))
		out = out.clamp(0.0, 1.0)

		return out.cpu(), alpha.cpu()


class APIndexer:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"images": ("IMAGE",),
			},
			"hidden": {
				"unique_id": "UNIQUE_ID",
				"prompt": "PROMPT",
			},
		}

	RETURN_TYPES = ("IMAGE", "INT")
	RETURN_NAMES = ("images", "current_frame_index")
	FUNCTION = "index"
	CATEGORY = "AP_OpticalFlow"

	def index(self, images, unique_id=None, prompt=None):
		img = _ensure_bhwc(images)
		batch_size = int(img.shape[0])

		node_key = str(unique_id) if unique_id is not None else "__default_node__"
		run_key = str(id(prompt)) if prompt is not None else "__default_run__"

		# Automatically reset per prompt execution.
		if _INDEXER_RUN_BY_NODE.get(node_key) != run_key:
			_INDEXER_RUN_BY_NODE[node_key] = run_key
			_INDEXER_STATE[node_key] = 0

		current_index = int(_INDEXER_STATE.get(node_key, 0))
		_INDEXER_STATE[node_key] = current_index + max(1, batch_size)

		return img, current_index


class APSelectFlowByIndex:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"flow_data": ("AP_FLOW",),
				"frame_index": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
			}
		}

	RETURN_TYPES = ("AP_FLOW", "IMAGE")
	RETURN_NAMES = ("flow_data", "flow_visualization")
	FUNCTION = "select"
	CATEGORY = "AP_OpticalFlow"

	def select(self, flow_data, frame_index=0):
		selected = _select_flow_data_by_index(flow_data, frame_index)

		vis_flow = None
		if isinstance(selected, dict):
			vis_flow = selected.get("flow_ab", None)
			if vis_flow is None:
				vis_flow = selected.get("flow_ba", None)
		else:
			vis_flow = selected

		if vis_flow is None:
			raise ValueError("Selected flow data does not contain flow_ab or flow_ba")

		flow_vis = _flow_to_color(_as_bhw2(vis_flow).detach().cpu().float())
		return selected, flow_vis


class APSaveOpticalFlow:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"flow_data": ("AP_FLOW",),
				"filename_prefix": ("STRING", {"default": "AP_OpticalFlow/flow"}),
				"overwrite": ("BOOLEAN", {"default": False}),
			}
		}

	RETURN_TYPES = ("AP_FLOW", "STRING")
	RETURN_NAMES = ("flow_data", "saved_path")
	FUNCTION = "save"
	CATEGORY = "AP_OpticalFlow"

	def save(self, flow_data, filename_prefix="AP_OpticalFlow/flow", overwrite=False):
		save_path = _make_save_path(filename_prefix, bool(overwrite))
		payload = {
			"ap_optical_flow": 1,
			"saved_at": datetime.now().isoformat(),
			"flow_data": _normalize_flow_data_for_save(flow_data),
		}
		torch.save(payload, save_path)
		return flow_data, save_path


class APLoadOpticalFlow:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"file_path": ("STRING", {"default": "AP_OpticalFlow/flow.pt"}),
			}
		}

	RETURN_TYPES = ("AP_FLOW", "IMAGE")
	RETURN_NAMES = ("flow_data", "flow_visualization")
	FUNCTION = "load"
	CATEGORY = "AP_OpticalFlow"

	def load(self, file_path):
		resolved_path = _resolve_load_path(file_path)
		data = torch.load(resolved_path, map_location="cpu")
		flow_data = _normalize_loaded_flow_data(data)

		vis_flow = flow_data.get("flow_ab", None)
		if vis_flow is None:
			vis_flow = flow_data.get("flow_ba", None)
		if vis_flow is None:
			raise ValueError("Loaded flow data does not contain flow_ab or flow_ba")

		flow_vis = _flow_to_color(_as_bhw2(vis_flow).detach().cpu().float())
		return flow_data, flow_vis


class AP_ImageMaskInpaintCrop:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"images": ("IMAGE",),
				"masks": ("MASK",),
				"padding": ("INT", {"default": 32, "min": 0, "max": 2048, "step": 1}),
				"mask_threshold": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.001}),
				"out_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
				"out_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
				"upscale_only": ("BOOLEAN", {"default": False}),
				"interpolation": (["bilinear", "bicubic", "nearest"], {"default": "bilinear"}),
			}
		}

	RETURN_TYPES = ("IMAGE", "MASK", "AP_STITCH")
	RETURN_NAMES = ("cropped_images", "cropped_masks", "stitch_data")
	FUNCTION = "crop"
	CATEGORY = "AP_OpticalFlow"

	def _resize_image(self, img_bhwc, out_h, out_w, interpolation):
		x = _to_bchw(img_bhwc.unsqueeze(0))
		mode = interpolation
		if mode == "nearest":
			y = F.interpolate(x, size=(out_h, out_w), mode=mode)
		else:
			y = F.interpolate(x, size=(out_h, out_w), mode=mode, align_corners=False)
		return _to_bhwc(y)[0]

	def _resize_mask(self, mask_hw, out_h, out_w):
		x = mask_hw.unsqueeze(0).unsqueeze(0)
		y = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)
		return y[0, 0].clamp(0.0, 1.0)

	def crop(
		self,
		images,
		masks,
		padding=32,
		mask_threshold=0.001,
		out_width=0,
		out_height=0,
		upscale_only=False,
		interpolation="bilinear",
	):
		img = _ensure_bhwc(images)
		msk = _ensure_mask_bhw(masks)
		img, msk = _match_batch(img, msk)

		b, h, w, c = img.shape
		padding = max(0, int(padding))
		thr = float(mask_threshold)
		ow = int(out_width)
		oh = int(out_height)

		crop_imgs = []
		crop_msks = []
		meta = []
		mask_proc_list = []

		max_h = 1
		max_w = 1

		for i in range(b):
			m = msk[i]
			bin_mask = m > thr
			if torch.any(bin_mask):
				ys, xs = torch.where(bin_mask)
				y0 = max(0, int(ys.min().item()) - padding)
				y1 = min(h, int(ys.max().item()) + 1 + padding)
				x0 = max(0, int(xs.min().item()) - padding)
				x1 = min(w, int(xs.max().item()) + 1 + padding)
			else:
				y0, y1, x0, x1 = 0, h, 0, w

			crop_img = img[i, y0:y1, x0:x1, :]
			crop_m = m[y0:y1, x0:x1]

			bbox_h = int(y1 - y0)
			bbox_w = int(x1 - x0)

			resize_requested = (ow > 0 and oh > 0)
			do_resize = resize_requested
			if resize_requested and upscale_only:
				do_resize = (bbox_w < ow) or (bbox_h < oh)

			if do_resize:
				proc_img = self._resize_image(crop_img, oh, ow, interpolation)
				proc_m = self._resize_mask(crop_m, oh, ow)
			else:
				proc_img = crop_img
				proc_m = crop_m

			vh = int(proc_img.shape[0])
			vw = int(proc_img.shape[1])
			max_h = max(max_h, vh)
			max_w = max(max_w, vw)

			crop_imgs.append(proc_img)
			crop_msks.append(proc_m)
			mask_proc_list.append(proc_m.detach().cpu())

			meta.append(
				{
					"y0": int(y0),
					"y1": int(y1),
					"x0": int(x0),
					"x1": int(x1),
					"bbox_h": int(bbox_h),
					"bbox_w": int(bbox_w),
					"valid_h": int(vh),
					"valid_w": int(vw),
					"resized": bool(do_resize),
				}
			)

		pad_imgs = []
		pad_msks = []
		for i in range(b):
			ci = crop_imgs[i]
			cm = crop_msks[i]
			vh, vw = int(ci.shape[0]), int(ci.shape[1])

			canvas_i = torch.zeros((max_h, max_w, c), dtype=ci.dtype, device=ci.device)
			canvas_m = torch.zeros((max_h, max_w), dtype=cm.dtype, device=cm.device)
			canvas_i[:vh, :vw, :] = ci
			canvas_m[:vh, :vw] = cm

			pad_imgs.append(canvas_i)
			pad_msks.append(canvas_m)

		out_img = torch.stack(pad_imgs, dim=0)
		out_msk = torch.stack(pad_msks, dim=0)

		stitch_data = {
			"version": 1,
			"batch": int(b),
			"source_h": int(h),
			"source_w": int(w),
			"meta": meta,
			"mask_proc": mask_proc_list,
		}

		return out_img, out_msk, stitch_data


class AP_ImageMaskStitch:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"original_images": ("IMAGE",),
				"inpainted_images": ("IMAGE",),
				"stitch_data": ("AP_STITCH",),
				"blend_with_mask": ("BOOLEAN", {"default": True}),
				"feather": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
				"interpolation": (["bilinear", "bicubic", "nearest"], {"default": "bilinear"}),
			}
		}

	RETURN_TYPES = ("IMAGE",)
	RETURN_NAMES = ("stitched_images",)
	FUNCTION = "stitch"
	CATEGORY = "AP_OpticalFlow"

	def _resize_patch(self, patch_bhwc, out_h, out_w, interpolation):
		x = _to_bchw(patch_bhwc.unsqueeze(0))
		if interpolation == "nearest":
			y = F.interpolate(x, size=(out_h, out_w), mode=interpolation)
		else:
			y = F.interpolate(x, size=(out_h, out_w), mode=interpolation, align_corners=False)
		return _to_bhwc(y)[0]

	def _resize_mask(self, mask_hw, out_h, out_w):
		x = mask_hw.unsqueeze(0).unsqueeze(0)
		y = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)
		return y[0, 0].clamp(0.0, 1.0)

	def stitch(self, original_images, inpainted_images, stitch_data, blend_with_mask=True, feather=0, interpolation="bilinear"):
		orig = _ensure_bhwc(original_images)
		inp = _ensure_bhwc(inpainted_images)

		if not isinstance(stitch_data, dict) or "meta" not in stitch_data:
			raise TypeError("stitch_data must be AP_STITCH output from AP_ImageMaskInpaintCrop")

		meta = stitch_data.get("meta", [])
		mask_proc_list = stitch_data.get("mask_proc", [])
		n = len(meta)
		if n <= 0:
			return orig

		if orig.shape[0] != n:
			if orig.shape[0] == 1:
				orig = orig.repeat(n, 1, 1, 1)
			else:
				raise ValueError(f"original_images batch {orig.shape[0]} does not match stitch_data batch {n}")

		if inp.shape[0] != n:
			if inp.shape[0] == 1:
				inp = inp.repeat(n, 1, 1, 1)
			else:
				raise ValueError(f"inpainted_images batch {inp.shape[0]} does not match stitch_data batch {n}")

		out_list = []
		for i in range(n):
			m = meta[i]
			y0, y1 = int(m["y0"]), int(m["y1"])
			x0, x1 = int(m["x0"]), int(m["x1"])
			bbox_h, bbox_w = int(m["bbox_h"]), int(m["bbox_w"])
			valid_h, valid_w = int(m["valid_h"]), int(m["valid_w"])

			orig_i = orig[i].clone()
			patch = inp[i, :valid_h, :valid_w, :]

			if patch.shape[0] != bbox_h or patch.shape[1] != bbox_w:
				patch = self._resize_patch(patch, bbox_h, bbox_w, interpolation)

			region = orig_i[y0:y1, x0:x1, :]

			if blend_with_mask and i < len(mask_proc_list):
				alpha = mask_proc_list[i].to(device=patch.device, dtype=patch.dtype)
				alpha = alpha[:valid_h, :valid_w]
				if alpha.shape[0] != bbox_h or alpha.shape[1] != bbox_w:
					alpha = self._resize_mask(alpha, bbox_h, bbox_w)

				if feather > 0:
					k = int(feather) * 2 + 1
					alpha = F.avg_pool2d(alpha.unsqueeze(0).unsqueeze(0), kernel_size=k, stride=1, padding=feather)[0, 0]

				alpha = alpha.clamp(0.0, 1.0).unsqueeze(-1)
				orig_i[y0:y1, x0:x1, :] = (patch * alpha) + (region * (1.0 - alpha))
			else:
				orig_i[y0:y1, x0:x1, :] = patch

			out_list.append(orig_i)

		return (torch.stack(out_list, dim=0),)


NODE_CLASS_MAPPINGS = {
	"APGetRAFTOpticalFlow": APGetRAFTOpticalFlow,
	"APApplyRAFTOpticalFlow": APApplyRAFTOpticalFlow,
	"APFlowOcclusionMask": APFlowOcclusionMask,
	"APApplyRAFTOpticalFlowMasked": APApplyRAFTOpticalFlowMasked,
	"APWarpImageAndMaskByRAFTFlow": APWarpImageAndMaskByRAFTFlow,
	"APFlowComposite": APFlowComposite,
	"APIndexer": APIndexer,
	"APSelectFlowByIndex": APSelectFlowByIndex,
	"APSaveOpticalFlow": APSaveOpticalFlow,
	"APLoadOpticalFlow": APLoadOpticalFlow,
	"AP_ImageMaskInpaintCrop": AP_ImageMaskInpaintCrop,
	"AP_ImageMaskStitch": AP_ImageMaskStitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"APGetRAFTOpticalFlow": "AP Get RAFT Optical Flow",
	"APApplyRAFTOpticalFlow": "AP Apply RAFT Optical Flow",
	"APFlowOcclusionMask": "AP Flow Occlusion Mask",
	"APApplyRAFTOpticalFlowMasked": "AP Apply RAFT Optical Flow (Masked)",
	"APWarpImageAndMaskByRAFTFlow": "AP Warp Image + Mask by RAFT Flow",
	"APFlowComposite": "AP Flow Composite",
	"APIndexer": "AP Indexer",
	"APSelectFlowByIndex": "AP Select Flow By Index",
	"APSaveOpticalFlow": "AP Save Optical Flow",
	"APLoadOpticalFlow": "AP Load Optical Flow",
	"AP_ImageMaskInpaintCrop": "AP Image Mask Inpaint Crop",
	"AP_ImageMaskStitch": "AP Image Mask Stitch",
}
