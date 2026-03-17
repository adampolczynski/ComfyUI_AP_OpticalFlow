import torch
import torch.nn.functional as F
import os
import gc
from datetime import datetime
import numpy as np
from PIL import Image

try:
	from comfy_execution.graph_utils import GraphBuilder, is_link
except Exception:
	GraphBuilder = None

	def is_link(value):
		if not isinstance(value, (list, tuple)):
			return False
		if len(value) != 2:
			return False
		return isinstance(value[0], (str, int)) and isinstance(value[1], int)

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

try:
	import comfy.utils as comfy_utils
except Exception:
	comfy_utils = None


_MODEL_CACHE = {}
_INDEXER_STATE = {}
_INDEXER_RUN_BY_NODE = {}


def _empty_device_cache():
	if model_management is not None:
		try:
			model_management.soft_empty_cache()
			return
		except Exception:
			pass

	if torch.cuda.is_available():
		try:
			torch.cuda.empty_cache()
		except Exception:
			pass

	if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
		try:
			torch.mps.empty_cache()
		except Exception:
			pass


def _clear_model_cache():
	for _, item in list(_MODEL_CACHE.items()):
		try:
			model = item[0]
			model.to("cpu")
		except Exception:
			pass
	_MODEL_CACHE.clear()
	gc.collect()
	_empty_device_cache()


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


def _bridge_mask_to_rgb(mask_bhw):
	m = _ensure_mask_bhw(mask_bhw).clamp(0.0, 1.0)
	return m.unsqueeze(-1).repeat(1, 1, 1, 3)


def _bridge_tensor_to_pil(image_hwc):
	arr = image_hwc.detach().cpu().numpy()
	if arr.ndim == 2:
		return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L")
	if arr.ndim != 3:
		raise ValueError(f"Expected image with 3 dims [H,W,C], got {arr.ndim}")

	c = arr.shape[2]
	if c == 1:
		gray = np.clip(arr[:, :, 0] * 255.0, 0, 255).astype(np.uint8)
		return Image.fromarray(gray, mode="L")
	if c == 2:
		pad = np.zeros((arr.shape[0], arr.shape[1], 1), dtype=arr.dtype)
		arr = np.concatenate([arr, pad], axis=2)
		rgb = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
		return Image.fromarray(rgb[:, :, :3], mode="RGB")

	rgb = np.clip(arr[:, :, :3] * 255.0, 0, 255).astype(np.uint8)
	return Image.fromarray(rgb, mode="RGB")


def _bridge_combine_image_and_mask(image_bhwc, mask_bhw, overlay_strength=1.0):
	img = _ensure_bhwc(image_bhwc)
	msk = _ensure_mask_bhw(mask_bhw)
	img, msk = _match_batch(img, msk)

	alpha = (msk.unsqueeze(-1).clamp(0.0, 1.0) * float(overlay_strength)).clamp(0.0, 1.0)
	white = torch.ones_like(img)
	return (img * (1.0 - alpha) + white * alpha).clamp(0.0, 1.0)


def _bridge_save_images_and_masks(
	images,
	masks,
	filename_prefix,
	output_dir,
	output_type="output",
	compress_level=4,
	write_images=True,
	write_masks=True,
	save_combined=False,
):
	if images is None and masks is None:
		raise ValueError("Provide IMAGE and/or MASK input")

	if folder_paths is None:
		raise RuntimeError("folder_paths is unavailable; cannot save preview/output images")

	img_batch = _ensure_bhwc(images) if images is not None else None
	mask_batch = _ensure_mask_bhw(masks) if masks is not None else None

	if img_batch is not None:
		height = int(img_batch.shape[1])
		width = int(img_batch.shape[2])
	else:
		height = int(mask_batch.shape[1])
		width = int(mask_batch.shape[2])

	full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
		str(filename_prefix),
		output_dir,
		width,
		height,
	)

	results = []

	if bool(write_images) and img_batch is not None:
		for batch_number, image in enumerate(img_batch):
			filename_with_batch = filename.replace("%batch_num%", str(batch_number))
			file = f"{filename_with_batch}_{counter:05}_img_.png"
			_bridge_tensor_to_pil(image).save(
				os.path.join(full_output_folder, file),
				compress_level=int(max(0, min(9, compress_level))),
			)
			results.append({"filename": file, "subfolder": subfolder, "type": output_type})
			counter += 1

	if bool(write_masks) and mask_batch is not None:
		mask_rgb_batch = _bridge_mask_to_rgb(mask_batch)
		for batch_number, image in enumerate(mask_rgb_batch):
			filename_with_batch = filename.replace("%batch_num%", str(batch_number))
			file = f"{filename_with_batch}_{counter:05}_mask_.png"
			_bridge_tensor_to_pil(image).save(
				os.path.join(full_output_folder, file),
				compress_level=int(max(0, min(9, compress_level))),
			)
			results.append({"filename": file, "subfolder": subfolder, "type": output_type})
			counter += 1

	if bool(save_combined) and img_batch is not None and mask_batch is not None:
		combined_batch = _bridge_combine_image_and_mask(img_batch, mask_batch)
		for batch_number, image in enumerate(combined_batch):
			filename_with_batch = filename.replace("%batch_num%", str(batch_number))
			file = f"{filename_with_batch}_{counter:05}_imgmask_.png"
			_bridge_tensor_to_pil(image).save(
				os.path.join(full_output_folder, file),
				compress_level=int(max(0, min(9, compress_level))),
			)
			results.append({"filename": file, "subfolder": subfolder, "type": output_type})
			counter += 1

	return results


def _bridge_passthrough(images, masks, node_name):
	if images is None and masks is None:
		raise ValueError(f"{node_name} requires IMAGE and/or MASK input")

	img_out = _ensure_bhwc(images).clone() if images is not None else None
	mask_out = _ensure_mask_bhw(masks).clone() if masks is not None else None

	if img_out is None:
		img_out = _bridge_mask_to_rgb(mask_out)

	if mask_out is None:
		mask_out = torch.zeros(
			(img_out.shape[0], img_out.shape[1], img_out.shape[2]),
			dtype=img_out.dtype,
			device=img_out.device,
		)

	return img_out, mask_out


def _bridge_emit_preview_batch(images=None, masks=None, unique_id=None, preview_mode="image"):
	if comfy_utils is None:
		return

	mode = str(preview_mode)
	preview_images = []
	if mode == "mask":
		if masks is None:
			raise ValueError("AP Bridge Preview Batch preview_mode='mask' requires MASK input")
		for mask_rgb in _bridge_mask_to_rgb(_ensure_mask_bhw(masks)):
			preview_images.append(_bridge_tensor_to_pil(mask_rgb))
	elif mode == "composite":
		if images is None or masks is None:
			raise ValueError("AP Bridge Preview Batch preview_mode='composite' requires both IMAGE and MASK inputs")
		for image in _bridge_combine_image_and_mask(images, masks):
			preview_images.append(_bridge_tensor_to_pil(image))
	else:
		if images is None:
			raise ValueError("AP Bridge Preview Batch preview_mode='image' requires IMAGE input")
		for image in _ensure_bhwc(images):
			preview_images.append(_bridge_tensor_to_pil(image))

	if not preview_images:
		return

	node_id = str(unique_id) if unique_id is not None else None
	pbar = comfy_utils.ProgressBar(len(preview_images), node_id=node_id)
	for i, pil_img in enumerate(preview_images):
		pbar.update_absolute(i + 1, len(preview_images), ("PNG", pil_img, None))


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


def _materialize_flow_data(flow_data):
	if isinstance(flow_data, dict) and str(flow_data.get("__storage__", "")).lower() == "disk":
		file_path = flow_data.get("file_path", "")
		resolved = _resolve_load_path(file_path)
		data = torch.load(resolved, map_location="cpu")
		loaded = _normalize_loaded_flow_data(data)
		loaded["source_file_path"] = resolved
		return loaded
	return flow_data


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


def _ensure_latent(latent):
	if not isinstance(latent, dict):
		raise TypeError("Expected LATENT dict")
	if "samples" not in latent:
		raise KeyError("LATENT dict missing 'samples'")
	samples = latent["samples"]
	if not torch.is_tensor(samples) or samples.ndim != 4:
		raise ValueError(f"LATENT samples must be [B,C,H,W], got {tuple(getattr(samples, 'shape', []))}")
	return latent, samples


def _copy_latent_with_samples(latent, samples):
	out = dict(latent)
	out["samples"] = samples
	return out


def _latent_dict_to_cpu(latent):
	out = {}
	for k, v in latent.items():
		if torch.is_tensor(v):
			out[k] = v.detach().cpu()
		else:
			out[k] = v
	return out


def _safe_int(value, default=0):
	try:
		return int(value)
	except Exception:
		return int(default)


def _normalize_loop_index(index, total):
	total_i = max(1, _safe_int(total, 1))
	idx = _safe_int(index, 0)
	if idx < 0:
		idx = 0
	if idx >= total_i:
		idx = total_i - 1
	return idx


def _parse_index_list(indexes_csv):
	text = str(indexes_csv if indexes_csv is not None else "").strip()
	if not text:
		return []
	out = []
	for token in text.split(","):
		t = token.strip()
		if not t:
			continue
		try:
			out.append(int(t))
		except Exception:
			continue
	return out


def _slice_image_batch(images, index):
	img = _ensure_bhwc(images)
	idx = _normalize_loop_index(index, img.shape[0])
	return img[idx : idx + 1].clone()


def _slice_mask_batch(mask, index):
	m = _ensure_mask_bhw(mask)
	idx = _normalize_loop_index(index, m.shape[0])
	return m[idx : idx + 1].clone()


def _slice_latent_batch(latent, index):
	latent_in, samples = _ensure_latent(latent)
	b = int(samples.shape[0])
	idx = _normalize_loop_index(index, b)
	out = {}
	for k, v in latent_in.items():
		if torch.is_tensor(v):
			if v.ndim >= 1 and int(v.shape[0]) == b:
				out[k] = v[idx : idx + 1].clone()
			else:
				out[k] = v.clone()
		else:
			out[k] = v
	return out


def _slice_additional_data_for_index(additional_data, index, total_hint=None):
	if additional_data is None:
		return None

	if isinstance(additional_data, dict) and "samples" in additional_data and torch.is_tensor(additional_data["samples"]):
		return _slice_latent_batch(additional_data, index)

	if torch.is_tensor(additional_data):
		t = additional_data
		if t.ndim >= 1:
			if (total_hint is not None and int(t.shape[0]) == int(total_hint)) or int(t.shape[0]) > 1:
				idx = _normalize_loop_index(index, int(t.shape[0]))
				return t[idx : idx + 1]
		return t

	if isinstance(additional_data, (list, tuple)):
		if len(additional_data) == 0:
			return None
		idx = _normalize_loop_index(index, len(additional_data))
		return additional_data[idx]

	if isinstance(additional_data, dict):
		out = {}
		had_slice = False
		for key, value in additional_data.items():
			if torch.is_tensor(value) and value.ndim >= 1:
				if (total_hint is not None and int(value.shape[0]) == int(total_hint)) or int(value.shape[0]) > 1:
					vidx = _normalize_loop_index(index, int(value.shape[0]))
					out[key] = value[vidx : vidx + 1]
					had_slice = True
				else:
					out[key] = value
			elif isinstance(value, (list, tuple)) and len(value) > 0:
				if (total_hint is not None and len(value) == int(total_hint)) or len(value) > 1:
					vidx = _normalize_loop_index(index, len(value))
					out[key] = value[vidx]
					had_slice = True
				else:
					out[key] = value
			else:
				out[key] = value
		return out if had_slice else additional_data

	return additional_data


def _accumulate_additional_data(accumulator, value, index, total_items):
	if value is None:
		return accumulator

	total = max(1, _safe_int(total_items, 1))
	idx = _normalize_loop_index(index, total)

	if isinstance(value, dict) and "samples" in value and torch.is_tensor(value["samples"]):
		entry = _slice_latent_batch(value, 0)
		if not (isinstance(accumulator, dict) and "samples" in accumulator and torch.is_tensor(accumulator["samples"])):
			accumulator = _init_latent_accumulator(entry, total)
		_store_latent_accumulator(accumulator, entry, idx)
		return accumulator

	if torch.is_tensor(value):
		entry = value
		if entry.ndim >= 1 and int(entry.shape[0]) != 1:
			eidx = _normalize_loop_index(idx, int(entry.shape[0]))
			entry = entry[eidx : eidx + 1]

		if entry.ndim >= 1 and int(entry.shape[0]) == 1:
			if not (torch.is_tensor(accumulator) and accumulator.ndim >= 1 and int(accumulator.shape[0]) == total and accumulator.shape[1:] == entry.shape[1:]):
				new_acc = torch.zeros((total, *entry.shape[1:]), dtype=entry.dtype, device=entry.device)
				if torch.is_tensor(accumulator) and accumulator.ndim >= 1 and accumulator.shape[1:] == entry.shape[1:]:
					n = min(total, int(accumulator.shape[0]))
					new_acc[:n] = accumulator[:n]
				accumulator = new_acc
			accumulator[idx : idx + 1] = entry
			return accumulator

	if not isinstance(accumulator, list):
		new_acc = [None for _ in range(total)]
		if isinstance(accumulator, tuple):
			for i, item in enumerate(accumulator[:total]):
				new_acc[i] = item
		elif isinstance(accumulator, list):
			for i, item in enumerate(accumulator[:total]):
				new_acc[i] = item
		accumulator = new_acc
	elif len(accumulator) < total:
		accumulator = list(accumulator) + [None for _ in range(total - len(accumulator))]

	accumulator[idx] = value
	return accumulator


def _move_any_to_cpu(value):
	if torch.is_tensor(value):
		return value.detach().cpu()

	if isinstance(value, dict):
		if "samples" in value and torch.is_tensor(value["samples"]):
			return _latent_dict_to_cpu(value)
		return {k: _move_any_to_cpu(v) for k, v in value.items()}

	if isinstance(value, list):
		return [_move_any_to_cpu(v) for v in value]

	if isinstance(value, tuple):
		return tuple(_move_any_to_cpu(v) for v in value)

	return value


def _make_default_mask_from_image(images):
	img = _ensure_bhwc(images)
	return torch.ones((img.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype, device=img.device)


def _make_default_mask_from_latent(samples):
	return torch.ones((samples.shape[0], samples.shape[2], samples.shape[3]), dtype=samples.dtype, device=samples.device)


def _ensure_mask_for_image_batch(mask, images):
	img = _ensure_bhwc(images)
	if mask is None:
		return _make_default_mask_from_image(img)
	m = _ensure_mask_bhw(mask)
	img, m = _match_batch(img, m)
	m = _resize_mask(m, img.shape[1], img.shape[2])
	return m


def _ensure_mask_for_latent_batch(mask, latent_samples):
	s = latent_samples
	if mask is None:
		return _make_default_mask_from_latent(s)
	m = _ensure_mask_bhw(mask)
	if m.shape[0] != s.shape[0]:
		if m.shape[0] == 1 and s.shape[0] > 1:
			m = m.repeat(s.shape[0], 1, 1)
		elif s.shape[0] == 1 and m.shape[0] > 1:
			m = m[:1]
		else:
			raise ValueError(f"Batch mismatch: latent={s.shape[0]} mask={m.shape[0]}")
	m = _resize_mask(m, s.shape[2], s.shape[3])
	return m


def _prepend_history_batch(history, new_entry, max_items):
	n = max(0, min(5, _safe_int(max_items, 0)))
	if n <= 0:
		return None
	entry = new_entry
	if entry.shape[0] != 1:
		entry = entry[:1]
	if history is None:
		return entry[:n]
	h = history
	if h.shape[0] > 0:
		if h.shape[1:] != entry.shape[1:]:
			h = None
	if h is None:
		return entry[:n]
	out = torch.cat([entry, h], dim=0)
	return out[:n]


def _history_slot_or_zero(history, slot, like_entry):
	s = _safe_int(slot, 0)
	if history is None or history.shape[0] <= s:
		return torch.zeros_like(like_entry[:1])
	return history[s : s + 1]


def _zero_latent_like(latent_like):
	lat, samples = _ensure_latent(latent_like)
	return _copy_latent_with_samples(lat, torch.zeros_like(samples[:1]))


def _init_latent_accumulator(latent_template, total_items):
	entry = _slice_latent_batch(latent_template, 0)
	total = max(1, _safe_int(total_items, 1))
	out = {}
	for k, v in entry.items():
		if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == 1:
			out[k] = torch.zeros((total, *v.shape[1:]), dtype=v.dtype, device=v.device)
		else:
			out[k] = v
	return out


def _store_latent_accumulator(acc_latent, latent_entry, index):
	idx = _safe_int(index, 0)
	for k, v in latent_entry.items():
		if not torch.is_tensor(v):
			continue
		if k not in acc_latent or not torch.is_tensor(acc_latent[k]):
			continue
		a = acc_latent[k]
		if a.ndim < 1 or v.ndim < 1:
			continue
		if v.shape[0] != 1 or a.shape[0] <= idx:
			continue
		if a.shape[1:] != v.shape[1:]:
			continue
		a[idx : idx + 1] = v


def _prepend_latent_history(history_latent, latent_entry, max_items):
	n = max(0, min(5, _safe_int(max_items, 0)))
	if n <= 0:
		return None
	entry = _slice_latent_batch(latent_entry, 0)
	if history_latent is None:
		return entry

	out = {}
	keys = set(history_latent.keys()) | set(entry.keys())
	for k in keys:
		new_v = entry.get(k, None)
		old_v = history_latent.get(k, None)
		if torch.is_tensor(new_v) and new_v.ndim >= 1 and new_v.shape[0] == 1:
			if torch.is_tensor(old_v) and old_v.ndim >= 1 and old_v.shape[1:] == new_v.shape[1:]:
				cat = torch.cat([new_v, old_v], dim=0)
			else:
				cat = new_v
			out[k] = cat[:n]
		elif k in entry:
			out[k] = entry[k]
		else:
			out[k] = old_v
	return out


def _latent_history_slot_or_zero(history_latent, slot, latent_like):
	s = _safe_int(slot, 0)
	if history_latent is None:
		return _zero_latent_like(latent_like)
	if "samples" not in history_latent:
		return _zero_latent_like(latent_like)
	hs = history_latent["samples"]
	if not torch.is_tensor(hs) or hs.ndim != 4 or hs.shape[0] <= s:
		return _zero_latent_like(latent_like)
	return _slice_latent_batch(history_latent, s)


def _init_image_accumulator(existing, total_items, sample):
	total = max(1, _safe_int(total_items, 1))
	if existing is None:
		return torch.zeros((total, sample.shape[1], sample.shape[2], sample.shape[3]), dtype=sample.dtype, device=sample.device)

	acc = _ensure_bhwc(existing).clone()
	if acc.shape[0] == total and acc.shape[1:] == sample.shape[1:]:
		return acc

	out = torch.zeros((total, sample.shape[1], sample.shape[2], sample.shape[3]), dtype=sample.dtype, device=sample.device)
	n = min(total, acc.shape[0])
	if acc.shape[1:] == sample.shape[1:]:
		out[:n] = acc[:n]
	return out


def _store_image_accumulator(acc, image_entry, index):
	idx = _safe_int(index, 0)
	if acc.shape[0] <= idx:
		return
	img = _ensure_bhwc(image_entry)
	if img.shape[0] != 1:
		img = img[:1]
	if acc.shape[1:] != img.shape[1:]:
		return
	acc[idx : idx + 1] = img


def _init_mask_accumulator(existing, total_items, sample_mask):
	total = max(1, _safe_int(total_items, 1))
	if existing is None:
		return torch.zeros((total, sample_mask.shape[1], sample_mask.shape[2]), dtype=sample_mask.dtype, device=sample_mask.device)

	acc = _ensure_mask_bhw(existing).clone()
	if acc.shape[0] == total and acc.shape[1:] == sample_mask.shape[1:]:
		return acc

	out = torch.zeros((total, sample_mask.shape[1], sample_mask.shape[2]), dtype=sample_mask.dtype, device=sample_mask.device)
	n = min(total, acc.shape[0])
	if acc.shape[1:] == sample_mask.shape[1:]:
		out[:n] = acc[:n]
	return out


def _store_mask_accumulator(acc, mask_entry, index):
	idx = _safe_int(index, 0)
	if acc.shape[0] <= idx:
		return
	m = _ensure_mask_bhw(mask_entry)
	if m.shape[0] != 1:
		m = m[:1]
	if acc.shape[1:] != m.shape[1:]:
		return
	acc[idx : idx + 1] = m


def _select_custom_image_frame(custom_frames, custom_indices_csv, iteration_index):
	if custom_frames is None:
		return None
	indices = _parse_index_list(custom_indices_csv)
	if not indices:
		return None
	frames = _ensure_bhwc(custom_frames)
	target = _safe_int(iteration_index, 0)
	matches = []
	for i, frame_idx in enumerate(indices):
		if frame_idx == target and i < frames.shape[0]:
			matches.append(frames[i : i + 1].clone())
	if not matches:
		return None
	return torch.cat(matches, dim=0)


def _select_custom_latent_frame(custom_latents, custom_indices_csv, iteration_index):
	if custom_latents is None:
		return None
	indices = _parse_index_list(custom_indices_csv)
	if not indices:
		return None
	target = _safe_int(iteration_index, 0)
	matches = []
	for i, frame_idx in enumerate(indices):
		if frame_idx == target:
			matches.append(_slice_latent_batch(custom_latents, i))
	if not matches:
		return None

	out = {}
	keys = set()
	for m in matches:
		keys.update(m.keys())

	for k in keys:
		values = [m.get(k) for m in matches if k in m]
		if values and all(torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == 1 for v in values):
			out[k] = torch.cat(values, dim=0)
		elif values:
			out[k] = values[0]

	return out


def _require_graph_builder():
	if GraphBuilder is None:
		raise RuntimeError("Comfy loop expansion requires comfy_execution.graph_utils.GraphBuilder")


def _explore_loop_dependencies(node_id, dynprompt, upstream):
	node_info = dynprompt.get_node(node_id)
	if "inputs" not in node_info:
		return
	for _, value in node_info["inputs"].items():
		if is_link(value):
			parent_id = value[0]
			if parent_id not in upstream:
				upstream[parent_id] = []
				_explore_loop_dependencies(parent_id, dynprompt, upstream)
			upstream[parent_id].append(node_id)


def _collect_loop_contained(node_id, upstream, contained):
	if node_id not in upstream:
		return
	for child_id in upstream[node_id]:
		if child_id not in contained:
			contained[child_id] = True
			_collect_loop_contained(child_id, upstream, contained)


def _collect_loop_upstream(node_id, dynprompt, contained, visited):
	if node_id in visited or node_id in contained:
		return
	visited[node_id] = True
	node_info = dynprompt.get_node(node_id)
	if not isinstance(node_info, dict):
		return
	contained[node_id] = True
	for _, value in node_info.get("inputs", {}).items():
		if is_link(value):
			_collect_loop_upstream(value[0], dynprompt, contained, visited)


def _linked_parents(node_info):
	if not isinstance(node_info, dict):
		return []
	parents = []
	for _, value in node_info.get("inputs", {}).items():
		if is_link(value):
			parents.append(str(value[0]))
	return parents


def _is_boundary_passthrough_node(node_info):
	if not isinstance(node_info, dict):
		return False

	parents = _linked_parents(node_info)
	if not parents:
		return False

	# Passthrough/proxy nodes should forward from a single upstream source.
	if len(set(parents)) > 1:
		return False

	class_type = str(node_info.get("class_type", "")).strip().lower()
	proxy_markers = (
		"reroute",
		"subgraph",
		"graphinput",
		"graph_input",
		"graphoutput",
		"graph_output",
		"passthrough",
		"pass_through",
		"identity",
		"proxy",
		"relay",
	)
	if any(marker in class_type for marker in proxy_markers):
		return True

	return class_type in {"primitivenode", "primitive"}


def _expand_processed_input_boundaries(seed_node_ids, dynprompt, contained):
	# In subgraph workflows, close inputs may point to internal proxy/input nodes.
	# Walk through those contained proxies and pull in any outside parents they reference.
	if not seed_node_ids:
		return 0

	# This is a boundary repair pass, so only traverse from nodes already in
	# the loop core and only through proxy/passthrough nodes.
	queue = [str(x) for x in seed_node_ids if x is not None and str(x) in contained]
	seen = {}
	upstream_seen = {}
	added = 0

	while queue:
		node_id = queue.pop()
		if node_id in seen:
			continue
		seen[node_id] = True

		node_info = dynprompt.get_node(node_id)
		if not isinstance(node_info, dict):
			continue
		if not _is_boundary_passthrough_node(node_info):
			continue

		for parent_id in _linked_parents(node_info):
			if parent_id in contained:
				queue.append(parent_id)
				continue

			before = len(contained)
			_collect_loop_upstream(parent_id, dynprompt, contained, upstream_seen)
			if len(contained) > before:
				added += len(contained) - before

	return added


def _loop_token_open_node(loop_token):
	if not isinstance(loop_token, dict):
		return None
	open_node = loop_token.get("open_node")
	if open_node is None:
		return None
	return str(open_node)


def _linked_input_parent_id(dynprompt, node_id, input_name):
	try:
		node_info = dynprompt.get_node(node_id)
	except Exception:
		return None
	if not isinstance(node_info, dict):
		return None
	inputs = node_info.get("inputs", {})
	v = inputs.get(input_name)
	if is_link(v):
		return str(v[0])
	return None


def _loop_debug(message):
	print(f"[AP_OpticalFlow][LoopDebug] {message}")


def _loop_token_state(loop_token):
	if not isinstance(loop_token, dict):
		raise ValueError("loop_token must be a dict emitted by AP Loop Open")
	idx = _safe_int(loop_token.get("iteration_index", 0), 0)
	total = _safe_int(loop_token.get("iteration_count", 0), 0)
	hist = _safe_int(loop_token.get("history_count", 3), 3)
	return idx, total, hist


def _build_loop_recurse(flow_control, dynprompt, unique_id, close_overrides=None, open_overrides=None):
	_require_graph_builder()
	if dynprompt is None:
		raise RuntimeError("Loop close requires DYNPROMPT hidden input")
	if unique_id is None:
		raise RuntimeError("Loop close requires UNIQUE_ID hidden input")

	iter_idx = _safe_int(flow_control.get("iteration_index", -1), -1) if isinstance(flow_control, dict) else -1
	iter_total = _safe_int(flow_control.get("iteration_count", 0), 0) if isinstance(flow_control, dict) else 0
	_loop_debug(
		f"RecurseStart close_node={unique_id} token_iter={iter_idx} token_total={iter_total}"
	)

	# Prefer the live link feeding loop_token into this close node.
	# Token payload IDs can become stale across recursive ephemeral clones.
	live_open_node = _linked_input_parent_id(dynprompt, unique_id, "loop_token")
	token_open_node = _loop_token_open_node(flow_control)
	open_node = live_open_node if live_open_node is not None else token_open_node
	if open_node is None:
		raise RuntimeError("Loop close requires loop_token with open node reference")
	_loop_debug(
		"RecurseOpenResolve "
		f"live_open={live_open_node} token_open={token_open_node} selected_open={open_node}"
	)
	if live_open_node is not None and token_open_node is not None and live_open_node != token_open_node:
		if iter_idx <= 1:
			print(
				"[AP_OpticalFlow] Loop notice: token open_node differs from live link "
				f"({token_open_node} -> {live_open_node}); using live link."
			)
	upstream = {}
	_explore_loop_dependencies(unique_id, dynprompt, upstream)

	contained = {}
	_collect_loop_contained(open_node, upstream, contained)
	contained[unique_id] = True
	contained[open_node] = True
	_loop_debug(
		f"RecurseContainedInitial size={len(contained)} open_node={open_node} close_node={unique_id}"
	)

	close_node = dynprompt.get_node(unique_id)
	external_processed_inputs = []
	processed_input_seeds = []
	if isinstance(close_node, dict):
		close_inputs = close_node.get("inputs", {})
		for k in ("processed_image", "processed_latent"):
			v = close_inputs.get(k)
			if is_link(v):
				processed_input_seeds.append(str(v[0]))
				if v[0] not in contained:
					external_processed_inputs.append((k, v[0]))

	# When processed inputs are outside the initial open->close containment,
	# include their upstream branch so recursion can rebuild the processing path.
	if external_processed_inputs:
		visited = {}
		for _, source_id in external_processed_inputs:
			_collect_loop_upstream(source_id, dynprompt, contained, visited)
		_loop_debug(
			"RecurseExternalInputs "
			f"count={len(external_processed_inputs)} contained_size_after_expand={len(contained)}"
		)

	boundary_added = _expand_processed_input_boundaries(processed_input_seeds, dynprompt, contained)
	if boundary_added > 0:
		_loop_debug(
			"RecurseBoundaryExpansion "
			f"seeds={len(processed_input_seeds)} added={boundary_added} contained_size_after_expand={len(contained)}"
		)

	if iter_idx == 0 and external_processed_inputs:
		names = ", ".join(name for name, _ in external_processed_inputs)
		print(
			"[AP_OpticalFlow] Loop notice: "
			f"{names} linked outside initial loop core; auto-including upstream processing branch for recursion."
		)

	graph = GraphBuilder()
	for node_id in contained:
		original = dynprompt.get_node(node_id)
		clone_id = "Recurse" if node_id == unique_id else node_id
		node = graph.node(original["class_type"], clone_id)
		node.set_override_display_id(node_id)

	for node_id in contained:
		original = dynprompt.get_node(node_id)
		clone_id = "Recurse" if node_id == unique_id else node_id
		node = graph.lookup_node(clone_id)
		if node is None:
			continue
		for key, value in original.get("inputs", {}).items():
			if is_link(value) and value[0] in contained:
				parent = graph.lookup_node(value[0])
				if parent is not None:
					node.set_input(key, parent.out(value[1]))
			else:
				node.set_input(key, value)

	recurse = graph.lookup_node("Recurse")
	if recurse is None:
		raise RuntimeError("Failed to clone loop close node")

	if close_overrides:
		for key, value in close_overrides.items():
			recurse.set_input(key, value)

	if open_overrides:
		new_open = graph.lookup_node(open_node)
		if new_open is not None:
			for key, value in open_overrides.items():
				new_open.set_input(key, value)

	_loop_debug(
		f"RecurseFinalize clone_nodes={len(contained)} close_clone=Recurse open_clone={open_node}"
	)

	return recurse, graph.finalize()


def _align_image_sequence(frames):
	items = []
	for frame in frames:
		if frame is None:
			continue
		items.append(_ensure_bhwc(frame))
	if not items:
		raise ValueError("At least one IMAGE input is required")

	ref_h, ref_w = items[0].shape[1], items[0].shape[2]
	target_b = max(int(x.shape[0]) for x in items)

	aligned = []
	for x in items:
		y = x
		if y.shape[1] != ref_h or y.shape[2] != ref_w:
			y = _to_bhwc(F.interpolate(_to_bchw(y), size=(ref_h, ref_w), mode="bilinear", align_corners=False))
		if y.shape[0] == 1 and target_b > 1:
			y = y.repeat(target_b, 1, 1, 1)
		elif y.shape[0] != target_b:
			raise ValueError(f"Batch mismatch in temporal image blend inputs: expected 1 or {target_b}, got {y.shape[0]}")
		aligned.append(y)
	return aligned


def _align_latent_sequence(latents):
	entries = []
	samples = []
	for lat in latents:
		if lat is None:
			continue
		entry, s = _ensure_latent(lat)
		entries.append(entry)
		samples.append(s)

	if not samples:
		raise ValueError("At least one LATENT input is required")

	ref_c, ref_h, ref_w = samples[0].shape[1], samples[0].shape[2], samples[0].shape[3]
	target_b = max(int(s.shape[0]) for s in samples)

	aligned = []
	for s in samples:
		x = s
		if x.shape[1] != ref_c:
			raise ValueError(f"Latent channel mismatch: expected {ref_c}, got {x.shape[1]}")
		if x.shape[2] != ref_h or x.shape[3] != ref_w:
			x = F.interpolate(x, size=(ref_h, ref_w), mode="bilinear", align_corners=False)
		if x.shape[0] == 1 and target_b > 1:
			x = x.repeat(target_b, 1, 1, 1)
		elif x.shape[0] != target_b:
			raise ValueError(f"Batch mismatch in temporal latent blend inputs: expected 1 or {target_b}, got {x.shape[0]}")
		aligned.append(x)

	return entries[0], aligned


def _make_temporal_base_weights(num_items, recency_decay):
	n = max(1, _safe_int(num_items, 1))
	decay = max(0.0, float(recency_decay))
	idx = torch.arange(n, dtype=torch.float32)
	weights = torch.exp(-decay * idx)
	return weights


def _temporal_reduce_stack(
	stack,
	blend_mode,
	recency_decay=0.35,
	trim_ratio=0.2,
	similarity_sigma=0.1,
	robust_delta=0.08,
	channel_dim=-1,
	first_frame_boost=1.0,
):
	# stack: [T, B, ...]
	if stack.ndim < 3:
		raise ValueError("Temporal stack must be rank >= 3")

	t = int(stack.shape[0])
	if t == 1:
		return stack[0]

	mode = str(blend_mode)
	device = stack.device
	dtype = stack.dtype

	base_w = _make_temporal_base_weights(t, recency_decay).to(device=device, dtype=dtype)
	first_boost = max(1.0, float(first_frame_boost))
	if first_boost > 1.0:
		base_w[0] = base_w[0] * first_boost
	shape = [t] + [1] * (stack.ndim - 1)
	base_w_broadcast = base_w.view(*shape)

	if mode == "median":
		return torch.median(stack, dim=0).values

	if mode == "trimmed_mean":
		trim = int(round(float(trim_ratio) * t))
		trim = max(0, min(trim, (t - 1) // 2))
		if trim <= 0:
			return torch.mean(stack, dim=0)
		sorted_stack = torch.sort(stack, dim=0).values
		return torch.mean(sorted_stack[trim : t - trim], dim=0)

	if mode == "similarity_weighted":
		reference = stack[0:1]
		sigma = max(1e-5, float(similarity_sigma))
		dist = torch.mean((stack - reference) ** 2, dim=channel_dim, keepdim=True)
		sim = torch.exp(-dist / (sigma * sigma))
		weights = base_w_broadcast * sim
		return torch.sum(stack * weights, dim=0) / (torch.sum(weights, dim=0) + 1e-6)

	if mode == "robust_huber":
		delta = max(1e-5, float(robust_delta))
		base_mean = torch.sum(stack * base_w_broadcast, dim=0) / (torch.sum(base_w_broadcast, dim=0) + 1e-6)
		residual = torch.mean(torch.abs(stack - base_mean.unsqueeze(0)), dim=channel_dim, keepdim=True)
		huber = torch.where(residual <= delta, torch.ones_like(residual), delta / (residual + 1e-6))
		weights = base_w_broadcast * huber
		return torch.sum(stack * weights, dim=0) / (torch.sum(weights, dim=0) + 1e-6)

	# weighted_mean
	return torch.sum(stack * base_w_broadcast, dim=0) / (torch.sum(base_w_broadcast, dim=0) + 1e-6)


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
	flow_data = _materialize_flow_data(flow_data)
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
	flow_data = _materialize_flow_data(flow_data)
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
			raise ValueError("batch_mode='by_index' expects input batch size 1")
		if bf == 1:
			return img, flow
		idx = offset if current_frame_index is None else int(current_frame_index) + offset
		return img, _select_batch_entry(flow, idx)

	if batch_mode in ("repeat_image", "repeat_latent"):
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


def _build_raft_model(variant):
	if variant == "large":
		weights = Raft_Large_Weights.DEFAULT
		model = raft_large(weights=weights, progress=False)
	else:
		weights = Raft_Small_Weights.DEFAULT
		model = raft_small(weights=weights, progress=False)

	model = model.eval()
	transforms = weights.transforms()
	return model, transforms


def _get_raft_model(variant, device, use_fp16, model_residency="unload_after_use"):
	if _RAFT_IMPORT_ERROR is not None:
		raise RuntimeError(
			"torchvision RAFT import failed. Install a compatible torchvision build.\n"
			f"Import error: {_RAFT_IMPORT_ERROR}"
		)

	mode = str(model_residency)

	if mode == "cache_on_gpu":
		key = ("gpu", variant, str(device), bool(use_fp16))
		if key in _MODEL_CACHE:
			return _MODEL_CACHE[key]
		model, transforms = _build_raft_model(variant)
		model = model.to(device)
		_MODEL_CACHE[key] = (model, transforms)
		return model, transforms

	if mode == "cache_on_cpu":
		key = ("cpu", variant, bool(use_fp16))
		if key not in _MODEL_CACHE:
			model, transforms = _build_raft_model(variant)
			model = model.to("cpu")
			_MODEL_CACHE[key] = (model, transforms)
		model, transforms = _MODEL_CACHE[key]
		model = model.to(device)
		return model, transforms

	# unload_after_use (default): never keep the model in cache.
	model, transforms = _build_raft_model(variant)
	model = model.to(device)
	return model, transforms


def _release_raft_model(model, model_residency="unload_after_use"):
	mode = str(model_residency)
	if mode == "cache_on_cpu":
		try:
			model.to("cpu")
		except Exception:
			pass
		gc.collect()
		_empty_device_cache()
		return

	if mode == "unload_after_use":
		try:
			del model
		except Exception:
			pass
		gc.collect()
		_empty_device_cache()


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
				"model_residency": (["unload_after_use", "cache_on_cpu", "cache_on_gpu"], {"default": "unload_after_use"}),
				"compute_device": (["auto", "cuda", "cpu"], {"default": "auto"}),
				"compute_mode": (["sequential", "batched"], {"default": "sequential"}),
				"flow_offload": (["cpu_ram", "disk_storage"], {"default": "cpu_ram"}),
				"disk_filename_prefix": ("STRING", {"default": "AP_OpticalFlow/flow_auto"}),
				"disk_overwrite": ("BOOLEAN", {"default": False}),
				"clear_cached_models_first": ("BOOLEAN", {"default": False}),
				"compute_backward": ("BOOLEAN", {"default": True}),
				"max_side": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 8}),
				"use_fp16": ("BOOLEAN", {"default": True}),
			}
		}

	RETURN_TYPES = ("IMAGE", "AP_FLOW")
	RETURN_NAMES = ("flow_visualization", "flow_data")
	FUNCTION = "compute"
	CATEGORY = "AP_OpticalFlow"

	def compute(
		self,
		image_a,
		image_b,
		model_size="large",
		model_residency="unload_after_use",
		compute_device="auto",
		compute_mode="sequential",
		flow_offload="cpu_ram",
		disk_filename_prefix="AP_OpticalFlow/flow_auto",
		disk_overwrite=False,
		clear_cached_models_first=False,
		compute_backward=True,
		max_side=1024,
		use_fp16=True,
	):
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

		if bool(clear_cached_models_first):
			_clear_model_cache()

		if compute_device == "cpu":
			device = torch.device("cpu")
		elif compute_device == "cuda":
			device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
		else:
			device = _get_device()

		if model_residency == "unload_after_use":
			# Make sure no old cached model keeps VRAM hostage when unload mode is selected.
			_clear_model_cache()

		model, transforms = _get_raft_model(model_size, device, use_fp16, model_residency=model_residency)

		try:
			if compute_mode == "batched":
				flow_ab = _estimate_flow(model, transforms, a_raft, b_raft, device, use_fp16)
				if compute_backward:
					flow_ba = _estimate_flow(model, transforms, b_raft, a_raft, device, use_fp16)
				else:
					flow_ba = -flow_ab

				flow_ab = _resize_flow_b2hw(flow_ab, h0, w0)
				flow_ba = _resize_flow_b2hw(flow_ba, h0, w0)

				flow_ab_bhw2 = flow_ab.permute(0, 2, 3, 1).contiguous().cpu()
				flow_ba_bhw2 = flow_ba.permute(0, 2, 3, 1).contiguous().cpu()
			else:
				flow_ab_list = []
				flow_ba_list = []

				for i in range(a_raft.shape[0]):
					a_i = a_raft[i : i + 1]
					b_i = b_raft[i : i + 1]

					f_ab_i = _estimate_flow(model, transforms, a_i, b_i, device, use_fp16)
					if compute_backward:
						f_ba_i = _estimate_flow(model, transforms, b_i, a_i, device, use_fp16)
					else:
						f_ba_i = -f_ab_i

					f_ab_i = _resize_flow_b2hw(f_ab_i, h0, w0).permute(0, 2, 3, 1).contiguous().cpu()
					f_ba_i = _resize_flow_b2hw(f_ba_i, h0, w0).permute(0, 2, 3, 1).contiguous().cpu()

					flow_ab_list.append(f_ab_i)
					flow_ba_list.append(f_ba_i)

					if device.type == "cuda":
						_empty_device_cache()

				flow_ab_bhw2 = torch.cat(flow_ab_list, dim=0)
				flow_ba_bhw2 = torch.cat(flow_ba_list, dim=0)
		finally:
			_release_raft_model(model, model_residency=model_residency)

		flow_vis = _flow_to_color(flow_ab_bhw2)

		if flow_offload == "disk_storage":
			temp_flow_data = {
				"flow_ab": flow_ab_bhw2,
				"flow_ba": flow_ba_bhw2,
				"height": h0,
				"width": w0,
				"model": model_size,
			}
			save_path = _make_save_path(disk_filename_prefix, bool(disk_overwrite))
			payload = {
				"ap_optical_flow": 1,
				"saved_at": datetime.now().isoformat(),
				"flow_data": _normalize_flow_data_for_save(temp_flow_data),
			}
			torch.save(payload, save_path)
			flow_data = {
				"__storage__": "disk",
				"file_path": save_path,
				"height": h0,
				"width": w0,
				"model": model_size,
			}
		else:
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
		flow_data = _materialize_flow_data(flow_data)

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


class APApplyRAFTOpticalFlowLatent:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"latent": ("LATENT",),
				"flow_data": ("AP_FLOW",),
				"flow_direction": (["ab", "ba"], {"default": "ab"}),
				"batch_mode": (["auto", "by_index", "repeat_latent"], {"default": "auto"}),
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

	RETURN_TYPES = ("LATENT", "MASK")
	RETURN_NAMES = ("warped_latent", "valid_mask")
	FUNCTION = "apply"
	CATEGORY = "AP_OpticalFlow"

	def apply(
		self,
		latent,
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
		latent_in, samples = _ensure_latent(latent)

		if current_frame_index is not None and int(current_frame_index) < max(0, int(frames_skip)):
			valid = torch.ones((samples.shape[0], samples.shape[2], samples.shape[3]), dtype=samples.dtype, device=samples.device)
			return _copy_latent_with_samples(latent_in, samples.cpu()), valid.cpu()

		flow = _pick_flow_from_data(flow_data, flow_direction)
		samples, flow = _align_image_flow_batches(
			samples,
			flow,
			batch_mode=batch_mode,
			current_frame_index=current_frame_index,
			flow_skip=flow_skip,
		)

		lh, lw = samples.shape[2], samples.shape[3]
		if flow.shape[1] != lh or flow.shape[2] != lw:
			flow_b2hw = flow.permute(0, 3, 1, 2).contiguous()
			flow_b2hw = _resize_flow_b2hw(flow_b2hw, lh, lw)
			flow = flow_b2hw.permute(0, 2, 3, 1).contiguous()

		if invert_flow:
			flow = -flow
		flow = flow * float(strength)

		device = _get_device()
		samples_d = samples.to(device)
		orig_dtype = samples_d.dtype
		if device.type == "cpu" and samples_d.dtype in (torch.float16, torch.bfloat16):
			samples_d = samples_d.float()

		warped, valid = _warp_with_flow(
			samples_d,
			flow.to(device),
			interpolation=interpolation,
			padding_mode=padding_mode,
		)

		if warped.dtype != orig_dtype:
			warped = warped.to(orig_dtype)

		out_latent = _copy_latent_with_samples(latent_in, warped.cpu())
		return out_latent, valid.clamp(0.0, 1.0).cpu()


class APApplyRAFTOpticalFlowLatentMasked:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"latent": ("LATENT",),
				"mask": ("MASK",),
				"flow_data": ("AP_FLOW",),
				"flow_direction": (["ab", "ba"], {"default": "ab"}),
				"batch_mode": (["auto", "by_index", "repeat_latent"], {"default": "auto"}),
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

	RETURN_TYPES = ("LATENT", "LATENT", "MASK", "MASK")
	RETURN_NAMES = ("output_latent", "warped_latent", "warped_mask", "valid_mask")
	FUNCTION = "apply_masked"
	CATEGORY = "AP_OpticalFlow"

	def apply_masked(
		self,
		latent,
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
		latent_in, samples = _ensure_latent(latent)
		m = _ensure_mask_bhw(mask)

		if current_frame_index is not None and int(current_frame_index) < max(0, int(frames_skip)):
			m = _resize_mask(m, samples.shape[2], samples.shape[3])
			valid = torch.ones((samples.shape[0], samples.shape[2], samples.shape[3]), dtype=samples.dtype, device=samples.device)
			lat_cpu = samples.cpu()
			return _copy_latent_with_samples(latent_in, lat_cpu), _copy_latent_with_samples(latent_in, lat_cpu), m.cpu(), valid.cpu()

		flow = _pick_flow_from_data(flow_data, flow_direction)
		samples, flow = _align_image_flow_batches(
			samples,
			flow,
			batch_mode=batch_mode,
			current_frame_index=current_frame_index,
			flow_skip=flow_skip,
		)
		samples, m = _match_batch(samples, m)

		lh, lw = samples.shape[2], samples.shape[3]
		if flow.shape[1] != lh or flow.shape[2] != lw:
			flow_b2hw = flow.permute(0, 3, 1, 2).contiguous()
			flow_b2hw = _resize_flow_b2hw(flow_b2hw, lh, lw)
			flow = flow_b2hw.permute(0, 2, 3, 1).contiguous()
		m = _resize_mask(m, lh, lw)

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
		samples_d = samples.to(device)
		orig_dtype = samples_d.dtype
		if device.type == "cpu" and samples_d.dtype in (torch.float16, torch.bfloat16):
			samples_d = samples_d.float()
		flow_d = flow.to(device)
		m_d = m.to(device)

		warped, valid = _warp_with_flow(
			samples_d,
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
		warped_mask = warped_mask_bchw.squeeze(1).clamp(0.0, 1.0)

		alpha = (warped_mask * m_d).clamp(0.0, 1.0).unsqueeze(1)
		out = (warped * alpha) + (samples_d * (1.0 - alpha))

		if warped.dtype != orig_dtype:
			warped = warped.to(orig_dtype)
		if out.dtype != orig_dtype:
			out = out.to(orig_dtype)

		valid_applied = (valid * alpha.squeeze(1)).clamp(0.0, 1.0)

		out_latent = _copy_latent_with_samples(latent_in, out.cpu())
		warped_latent = _copy_latent_with_samples(latent_in, warped.cpu())
		return out_latent, warped_latent, warped_mask.cpu(), valid_applied.cpu()


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
				"alpha_mode": (["flow_confidence", "flow_confidence_x_mask", "mask_only"], {"default": "flow_confidence"}),
				"mask_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
				"use_difference_gate": ("BOOLEAN", {"default": False}),
				"difference_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
				"difference_feather": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
				"blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
				"feather": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
				"invert_occlusion": ("BOOLEAN", {"default": False}),
			},
			"optional": {
				"effect_mask": ("MASK",),
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
		alpha_mode="flow_confidence",
		mask_threshold=0.0,
		use_difference_gate=False,
		difference_threshold=0.01,
		difference_feather=0,
		blend_strength=1.0,
		feather=0,
		invert_occlusion=False,
		effect_mask=None,
	):
		orig = _ensure_bhwc(original_images)
		warp = _ensure_bhwc(warped_images)
		vm = _ensure_mask_bhw(valid_mask)
		occ = _ensure_mask_bhw(occlusion_mask)
		em = _ensure_mask_bhw(effect_mask) if effect_mask is not None else None

		orig, warp = _match_batch(orig, warp)
		orig, vm = _match_batch(orig, vm)
		orig, occ = _match_batch(orig, occ)
		if em is not None:
			orig, em = _match_batch(orig, em)

		h, w = orig.shape[1], orig.shape[2]
		vm = _resize_mask(vm, h, w)
		occ = _resize_mask(occ, h, w)
		if em is not None:
			em = _resize_mask(em, h, w)
			if float(mask_threshold) > 0.0:
				em = (em >= float(mask_threshold)).float()

		if invert_occlusion:
			occ = 1.0 - occ

		alpha_flow = (vm * (1.0 - occ)).clamp(0.0, 1.0)

		mode = str(alpha_mode)
		if mode == "mask_only":
			if em is None:
				alpha = alpha_flow
			else:
				alpha = em.clamp(0.0, 1.0)
		elif mode == "flow_confidence_x_mask":
			if em is None:
				alpha = alpha_flow
			else:
				alpha = (alpha_flow * em).clamp(0.0, 1.0)
		else:
			alpha = alpha_flow

		if bool(use_difference_gate):
			diff = torch.mean(torch.abs(warp - orig), dim=-1)
			diff_mask = (diff >= float(difference_threshold)).float()
			if int(difference_feather) > 0:
				k2 = int(difference_feather) * 2 + 1
				diff_mask = F.avg_pool2d(diff_mask.unsqueeze(1), kernel_size=k2, stride=1, padding=difference_feather).squeeze(1)
			alpha = (alpha * diff_mask).clamp(0.0, 1.0)

		if feather > 0:
			k = int(feather) * 2 + 1
			alpha = F.avg_pool2d(alpha.unsqueeze(1), kernel_size=k, stride=1, padding=feather).squeeze(1)

		alpha = (alpha * float(blend_strength)).clamp(0.0, 1.0)
		# Ensure compositing never leaks outside the explicit effect mask.
		if em is not None:
			alpha = (alpha * em).clamp(0.0, 1.0)

		alpha_img = alpha.unsqueeze(-1)
		out = (warp * alpha_img) + (orig * (1.0 - alpha_img))
		out = out.clamp(0.0, 1.0)

		return out.cpu(), alpha.cpu()


class APWarpMaskedCompositeOcclusion:
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
				"abs_epsilon": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 50.0, "step": 0.01}),
				"rel_epsilon": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.001}),
				"dilate_occlusion": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1}),
				"blend_images": ("BOOLEAN", {"default": True}),
				"alpha_mode": (["flow_confidence", "flow_confidence_x_mask", "mask_only"], {"default": "flow_confidence_x_mask"}),
				"mask_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
				"use_difference_gate": ("BOOLEAN", {"default": False}),
				"difference_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
				"difference_feather": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
				"blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
				"feather": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
				"invert_occlusion": ("BOOLEAN", {"default": False}),
			},
			"optional": {
				"current_frame_index": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
			},
		}

	RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK", "MASK", "MASK")
	RETURN_NAMES = (
		"output_images",
		"warped_images",
		"warped_mask",
		"valid_mask",
		"occlusion_mask",
		"composite_alpha",
	)
	FUNCTION = "run"
	CATEGORY = "AP_OpticalFlow"

	def run(
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
		abs_epsilon=1.0,
		rel_epsilon=0.05,
		dilate_occlusion=0,
		blend_images=True,
		alpha_mode="flow_confidence_x_mask",
		mask_threshold=0.0,
		use_difference_gate=False,
		difference_threshold=0.01,
		difference_feather=0,
		blend_strength=1.0,
		feather=0,
		invert_occlusion=False,
		current_frame_index=None,
	):
		masked_node = APApplyRAFTOpticalFlowMasked()
		_, warped_images, warped_mask, valid_mask = masked_node.apply_masked(
			images=images,
			mask=mask,
			flow_data=flow_data,
			flow_direction=flow_direction,
			batch_mode=batch_mode,
			flow_skip=flow_skip,
			frames_skip=frames_skip,
			strength=strength,
			invert_flow=invert_flow,
			invert_mask=invert_mask,
			mask_strength=mask_strength,
			mask_feather=mask_feather,
			interpolation=interpolation,
			padding_mode=padding_mode,
			current_frame_index=current_frame_index,
		)

		occ_batch_mode = "by_index" if str(batch_mode) == "by_index" else "auto"
		occ_node = APFlowOcclusionMask()
		_, occlusion_mask, _ = occ_node.compute(
			flow_data=flow_data,
			flow_direction=flow_direction,
			batch_mode=occ_batch_mode,
			flow_skip=flow_skip,
			frames_skip=frames_skip,
			abs_epsilon=abs_epsilon,
			rel_epsilon=rel_epsilon,
			dilate_occlusion=dilate_occlusion,
			current_frame_index=current_frame_index,
		)

		# Match the same effective mask logic as APApplyRAFTOpticalFlowMasked so
		# compositing happens strictly inside the configured mask footprint.
		source_effect_mask = _ensure_mask_bhw(mask)
		if bool(invert_mask):
			source_effect_mask = 1.0 - source_effect_mask
		if int(mask_feather) > 0:
			k = int(mask_feather) * 2 + 1
			source_effect_mask = F.avg_pool2d(
				source_effect_mask.unsqueeze(1),
				kernel_size=k,
				stride=1,
				padding=int(mask_feather),
			).squeeze(1)
		source_effect_mask = (source_effect_mask * float(mask_strength)).clamp(0.0, 1.0)

		warped_mask_bhw = _ensure_mask_bhw(warped_mask)
		source_effect_mask = _resize_mask(source_effect_mask, warped_mask_bhw.shape[1], warped_mask_bhw.shape[2])
		source_effect_mask, warped_mask_bhw = _match_batch(source_effect_mask, warped_mask_bhw)
		blend_effect_mask = (source_effect_mask * warped_mask_bhw).clamp(0.0, 1.0)

		if bool(blend_images):
			comp_node = APFlowComposite()
			output_images, composite_alpha = comp_node.composite(
				original_images=images,
				warped_images=warped_images,
				valid_mask=valid_mask,
				occlusion_mask=occlusion_mask,
				alpha_mode=alpha_mode,
				mask_threshold=mask_threshold,
				use_difference_gate=use_difference_gate,
				difference_threshold=difference_threshold,
				difference_feather=difference_feather,
				blend_strength=blend_strength,
				feather=feather,
				invert_occlusion=invert_occlusion,
				effect_mask=blend_effect_mask,
			)
		else:
			# No compositing requested: return pure warped result, not a mixed image.
			output_images = warped_images
			composite_alpha = torch.zeros_like(_ensure_mask_bhw(valid_mask))

		return output_images, warped_images, warped_mask, valid_mask, occlusion_mask, composite_alpha


class APImageLoopOpen:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"source_images": ("IMAGE",),
				"history_count": ("INT", {"default": 3, "min": 0, "max": 5, "step": 1}),
				"return_first_when_no_previous_available": ("BOOLEAN", {"default": False}),
				"apply_custom_replacement": ("BOOLEAN", {"default": False}),
				"custom_frame_index_map": ("STRING", {"default": ""}),
			},
			"optional": {
				"source_masks": ("MASK",),
				"custom_frames": ("IMAGE",),
				"additional_data": ("*",),
			},
			"hidden": {
				"iteration_index": ("INT", {"default": 0}),
				"all_processed_images": ("IMAGE",),
				"all_processed_masks": ("MASK",),
				"processed_history_images": ("IMAGE",),
				"processed_history_masks": ("MASK",),
				"unique_id": "UNIQUE_ID",
			},
		}

	RETURN_TYPES = (
		"FLOW_CONTROL",
		"IMAGE",
		"MASK",
		"IMAGE",
		"IMAGE",
		"MASK",
		"IMAGE",
		"IMAGE",
		"IMAGE",
		"IMAGE",
		"IMAGE",
		"MASK",
		"MASK",
		"MASK",
		"MASK",
		"MASK",
		"IMAGE",
		"*",
		"INT",
	)
	RETURN_NAMES = (
		"loop_token",
		"original_current_image",
		"original_current_mask",
		"original_first_image",
		"original_previous_image",
		"original_previous_mask",
		"processed_previous_image_1",
		"processed_previous_image_2",
		"processed_previous_image_3",
		"processed_previous_image_4",
		"processed_previous_image_5",
		"processed_previous_mask_1",
		"processed_previous_mask_2",
		"processed_previous_mask_3",
		"processed_previous_mask_4",
		"processed_previous_mask_5",
		"custom_current_frame",
		"additional_data_current",
		"iteration_index",
	)
	FUNCTION = "loop_open"
	CATEGORY = "AP_OpticalFlow"

	def loop_open(
		self,
		source_images,
		history_count=3,
		return_first_when_no_previous_available=False,
		apply_custom_replacement=False,
		custom_frame_index_map="",
		source_masks=None,
		custom_frames=None,
		additional_data=None,
		iteration_index=0,
		all_processed_images=None,
		all_processed_masks=None,
		processed_history_images=None,
		processed_history_masks=None,
		unique_id=None,
	):
		img = _ensure_bhwc(source_images).clone()
		total = int(img.shape[0])
		idx = _normalize_loop_index(iteration_index, total)

		m = _ensure_mask_for_image_batch(source_masks, img).clone()

		current_image = _slice_image_batch(img, idx).clone()
		current_mask = _slice_mask_batch(m, idx).clone()
		first_image = _slice_image_batch(img, 0).clone()
		first_mask = _slice_mask_batch(m, 0).clone()
		previous_image = _slice_image_batch(img, max(0, idx - 1)).clone()
		previous_mask = _slice_mask_batch(m, max(0, idx - 1)).clone()

		custom_frames_current = _select_custom_image_frame(custom_frames, custom_frame_index_map, idx)
		if custom_frames_current is not None:
			if custom_frames_current.shape[1] != current_image.shape[1] or custom_frames_current.shape[2] != current_image.shape[2]:
				custom_frames_current = _to_bhwc(
					F.interpolate(
						_to_bchw(custom_frames_current),
						size=(current_image.shape[1], current_image.shape[2]),
						mode="bilinear",
						align_corners=False,
					)
				)
			if apply_custom_replacement and custom_frames_current.shape[0] > 0:
				current_image = custom_frames_current[0:1].clone()
		else:
			custom_frames_current = torch.zeros_like(current_image)

		current_additional_data = _slice_additional_data_for_index(additional_data, idx, total_hint=total)

		hlen = max(0, min(5, _safe_int(history_count, 3)))
		acc_i = _ensure_bhwc(all_processed_images) if all_processed_images is not None else None
		acc_m = _ensure_mask_bhw(all_processed_masks) if all_processed_masks is not None else None
		hist_i = _ensure_bhwc(processed_history_images) if processed_history_images is not None else None
		hist_m = _ensure_mask_bhw(processed_history_masks) if processed_history_masks is not None else None
		if idx > 0 and hlen > 0:
			expected_prev_idx = idx - 1
			has_prev_from_acc = acc_i is not None and acc_i.shape[0] > expected_prev_idx
			has_prev_from_hist = hist_i is not None and hist_i.shape[0] > 0
			if not (has_prev_from_acc or has_prev_from_hist):
				raise RuntimeError(
					"APImageLoopOpen expected previous processed image at "
					f"iteration {idx}, but none was provided. "
					"Ensure loop_close.processed_image is connected to the actual processed output."
				)
		_loop_debug(
			"ImageLoopOpen "
			f"uid={unique_id} idx={idx}/{max(total - 1, 0)} total={total} history={hlen} "
			f"source_batch={int(img.shape[0])} custom_frames={'yes' if custom_frames is not None else 'no'}"
		)

		prev_images = []
		prev_masks = []
		prev1_source = "source"
		for i in range(5):
			prev_abs_idx = idx - (i + 1)
			has_acc_image = prev_abs_idx >= 0 and acc_i is not None and acc_i.shape[0] > prev_abs_idx
			has_acc_mask = prev_abs_idx >= 0 and acc_m is not None and acc_m.shape[0] > prev_abs_idx
			has_hist_image = i < hlen and hist_i is not None and hist_i.shape[0] > i
			has_hist_mask = i < hlen and hist_m is not None and hist_m.shape[0] > i
			if has_acc_image:
				prev_images.append(_slice_image_batch(acc_i, prev_abs_idx).clone())
				if i == 0:
					prev1_source = "accumulator"
			elif has_hist_image:
				prev_images.append(hist_i[i : i + 1].clone())
				if i == 0:
					prev1_source = "history"
			elif return_first_when_no_previous_available:
				prev_images.append(first_image.clone())
				if i == 0:
					prev1_source = "first"
			else:
				source_hist_idx = max(0, idx - (i + 1))
				prev_images.append(_slice_image_batch(img, source_hist_idx).clone())
				if i == 0:
					prev1_source = "source"

			if has_acc_mask:
				prev_masks.append(_slice_mask_batch(acc_m, prev_abs_idx).clone())
			elif has_hist_mask:
				prev_masks.append(hist_m[i : i + 1].clone())
			elif return_first_when_no_previous_available:
				prev_masks.append(first_mask.clone())
			else:
				source_hist_idx = max(0, idx - (i + 1))
				prev_masks.append(_slice_mask_batch(m, source_hist_idx).clone())

		if idx <= 1:
			_loop_debug(
				"ImageLoopOpenHistory "
				f"uid={unique_id} idx={idx} has_hist={'yes' if hist_i is not None and hist_i.shape[0] > 0 else 'no'} "
				f"prev1_source={prev1_source}"
			)

		if return_first_when_no_previous_available and idx <= 0:
			previous_image = first_image.clone()
			previous_mask = first_mask.clone()

		loop_token = {
			"open_node": str(unique_id) if unique_id is not None else None,
			"loop_type": "image",
			"iteration_index": idx,
			"iteration_count": total,
			"history_count": hlen,
		}
		_loop_debug(
			"ImageLoopOpenToken "
			f"uid={unique_id} open_node={loop_token['open_node']} idx={idx} total={total}"
		)

		return (
			loop_token,
			current_image,
			current_mask,
			first_image,
			previous_image,
			previous_mask,
			prev_images[0],
			prev_images[1],
			prev_images[2],
			prev_images[3],
			prev_images[4],
			prev_masks[0],
			prev_masks[1],
			prev_masks[2],
			prev_masks[3],
			prev_masks[4],
			custom_frames_current,
			current_additional_data,
			idx,
		)


class APImageLoopClose:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"loop_token": ("FLOW_CONTROL",),
				"processed_image": ("IMAGE",),
			},
			"optional": {
				"processed_mask": ("MASK",),
				"additional_data": ("*",),
			},
			"hidden": {
				"dynprompt": "DYNPROMPT",
				"unique_id": "UNIQUE_ID",
				"all_processed_images": ("IMAGE",),
				"all_processed_masks": ("MASK",),
				"all_processed_additional_data": ("*",),
				"processed_history_images": ("IMAGE",),
				"processed_history_masks": ("MASK",),
			},
		}

	RETURN_TYPES = ("IMAGE", "MASK", "*")
	RETURN_NAMES = ("processed_images", "processed_masks", "processed_additional_data")
	FUNCTION = "loop_close"
	CATEGORY = "AP_OpticalFlow"

	def loop_close(
		self,
		loop_token,
		processed_image,
		processed_mask=None,
		additional_data=None,
		dynprompt=None,
		unique_id=None,
		all_processed_images=None,
		all_processed_masks=None,
		all_processed_additional_data=None,
		processed_history_images=None,
		processed_history_masks=None,
	):
		token_idx, token_total, token_hist = _loop_token_state(loop_token)
		current_live_open = _linked_input_parent_id(dynprompt, unique_id, "loop_token") if dynprompt is not None and unique_id is not None else None

		processed_batch = _ensure_bhwc(processed_image)
		processed_batch_len = int(processed_batch.shape[0])

		total = _safe_int(token_total, 0)
		if total <= 0 and all_processed_images is not None:
			total = int(_ensure_bhwc(all_processed_images).shape[0])
		total = max(1, total if total > 0 else processed_batch_len)
		idx = _normalize_loop_index(token_idx, total)

		proc_idx = _normalize_loop_index(idx, processed_batch_len) if processed_batch_len > 1 else 0
		proc = _slice_image_batch(processed_batch, proc_idx)
		pmask = _slice_mask_batch(_ensure_mask_for_image_batch(processed_mask, processed_batch), proc_idx)
		_loop_debug(
			"ImageLoopClose "
			f"uid={unique_id} open_node={current_live_open} token_idx={token_idx} token_total={token_total} "
			f"resolved_idx={idx} resolved_total={total} processed_batch={processed_batch_len} proc_idx={proc_idx}"
		)

		acc_images = _init_image_accumulator(all_processed_images, total, proc)
		acc_masks = _init_mask_accumulator(all_processed_masks, total, pmask)
		acc_additional = _accumulate_additional_data(all_processed_additional_data, additional_data, idx, total)
		_store_image_accumulator(acc_images, proc, idx)
		_store_mask_accumulator(acc_masks, pmask, idx)

		hlen = max(0, min(5, _safe_int(token_hist, 3)))

		hist_i = _prepend_history_batch(
			_ensure_bhwc(processed_history_images) if processed_history_images is not None else None,
			proc,
			hlen,
		)
		hist_m = _prepend_history_batch(
			_ensure_mask_bhw(processed_history_masks) if processed_history_masks is not None else None,
			pmask,
			hlen,
		)

		if idx >= total - 1:
			_loop_debug(
				"ImageLoopCloseFinish "
				f"uid={unique_id} idx={idx} total={total} returning_accumulated_batch={int(acc_images.shape[0])}"
			)
			return acc_images.cpu(), acc_masks.cpu(), _move_any_to_cpu(acc_additional)

		next_idx = idx + 1
		next_loop_token = dict(loop_token)
		live_open_node = current_live_open
		if live_open_node is not None:
			next_loop_token["open_node"] = live_open_node
		next_loop_token["iteration_index"] = next_idx
		next_loop_token["iteration_count"] = total
		next_loop_token["history_count"] = hlen
		_loop_debug(
			"ImageLoopCloseRecurse "
			f"uid={unique_id} next_idx={next_idx} total={total} next_open={next_loop_token.get('open_node')}"
		)

		recurse, expand = _build_loop_recurse(
			next_loop_token,
			dynprompt,
			unique_id,
			close_overrides={
				"all_processed_images": acc_images,
				"all_processed_masks": acc_masks,
				"all_processed_additional_data": acc_additional,
				"processed_history_images": hist_i,
				"processed_history_masks": hist_m,
			},
			open_overrides={
				"iteration_index": next_idx,
				"all_processed_images": acc_images,
				"all_processed_masks": acc_masks,
				"processed_history_images": hist_i,
				"processed_history_masks": hist_m,
			},
		)

		return {
			"result": (recurse.out(0), recurse.out(1), recurse.out(2)),
			"expand": expand,
		}


class APLatentLoopOpen:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"source_latents": ("LATENT",),
				"history_count": ("INT", {"default": 3, "min": 0, "max": 5, "step": 1}),
				"return_first_when_no_previous_available": ("BOOLEAN", {"default": False}),
				"apply_custom_replacement": ("BOOLEAN", {"default": False}),
				"custom_frame_index_map": ("STRING", {"default": ""}),
			},
			"optional": {
				"source_masks": ("MASK",),
				"custom_latents": ("LATENT",),
				"additional_data": ("*",),
			},
			"hidden": {
				"iteration_index": ("INT", {"default": 0}),
				"all_processed_latents": ("LATENT",),
				"all_processed_masks": ("MASK",),
				"processed_history_latents": ("LATENT",),
				"processed_history_masks": ("MASK",),
				"unique_id": "UNIQUE_ID",
			},
		}

	RETURN_TYPES = (
		"FLOW_CONTROL",
		"LATENT",
		"MASK",
		"LATENT",
		"LATENT",
		"MASK",
		"LATENT",
		"LATENT",
		"LATENT",
		"LATENT",
		"LATENT",
		"MASK",
		"MASK",
		"MASK",
		"MASK",
		"MASK",
		"LATENT",
		"*",
		"INT",
	)
	RETURN_NAMES = (
		"loop_token",
		"original_current_latent",
		"original_current_mask",
		"original_first_latent",
		"original_previous_latent",
		"original_previous_mask",
		"processed_previous_latent_1",
		"processed_previous_latent_2",
		"processed_previous_latent_3",
		"processed_previous_latent_4",
		"processed_previous_latent_5",
		"processed_previous_mask_1",
		"processed_previous_mask_2",
		"processed_previous_mask_3",
		"processed_previous_mask_4",
		"processed_previous_mask_5",
		"custom_current_latent",
		"additional_data_current",
		"iteration_index",
	)
	FUNCTION = "loop_open"
	CATEGORY = "AP_OpticalFlow"

	def loop_open(
		self,
		source_latents,
		history_count=3,
		return_first_when_no_previous_available=False,
		apply_custom_replacement=False,
		custom_frame_index_map="",
		source_masks=None,
		custom_latents=None,
		additional_data=None,
		iteration_index=0,
		all_processed_latents=None,
		all_processed_masks=None,
		processed_history_latents=None,
		processed_history_masks=None,
		unique_id=None,
	):
		lat_in, samples = _ensure_latent(source_latents)
		lat_in = _latent_dict_to_cpu(lat_in)
		total = int(samples.shape[0])
		idx = _normalize_loop_index(iteration_index, total)

		m = _ensure_mask_for_latent_batch(source_masks, samples).clone()

		current_latent = _slice_latent_batch(lat_in, idx)
		current_mask = _slice_mask_batch(m, idx).clone()
		first_latent = _slice_latent_batch(lat_in, 0)
		first_mask = _slice_mask_batch(m, 0).clone()
		previous_latent = _slice_latent_batch(lat_in, max(0, idx - 1))
		previous_mask = _slice_mask_batch(m, max(0, idx - 1)).clone()

		custom_latent = _select_custom_latent_frame(custom_latents, custom_frame_index_map, idx)
		if custom_latent is not None:
			_, cs = _ensure_latent(custom_latent)
			_, cur_s = _ensure_latent(current_latent)
			if cs.shape[1] != cur_s.shape[1]:
				raise ValueError(
					f"custom_latents channel mismatch: expected {cur_s.shape[1]}, got {cs.shape[1]}"
				)
			if cs.shape[2] != cur_s.shape[2] or cs.shape[3] != cur_s.shape[3]:
				cs = F.interpolate(cs, size=(cur_s.shape[2], cur_s.shape[3]), mode="bilinear", align_corners=False)
				custom_latent = _copy_latent_with_samples(custom_latent, cs)
			if apply_custom_replacement and cs.shape[0] > 0:
				current_latent = _slice_latent_batch(custom_latent, 0)
		else:
			custom_latent = _zero_latent_like(current_latent)

		current_additional_data = _slice_additional_data_for_index(additional_data, idx, total_hint=total)

		hlen = max(0, min(5, _safe_int(history_count, 3)))
		acc_l = all_processed_latents if all_processed_latents is not None else None
		acc_m = _ensure_mask_bhw(all_processed_masks) if all_processed_masks is not None else None
		hist_l = processed_history_latents if processed_history_latents is not None else None
		hist_m = _ensure_mask_bhw(processed_history_masks) if processed_history_masks is not None else None
		if idx > 0 and hlen > 0:
			expected_prev_idx = idx - 1
			has_prev_from_acc = (
				acc_l is not None
				and "samples" in acc_l
				and torch.is_tensor(acc_l["samples"])
				and acc_l["samples"].ndim == 4
				and acc_l["samples"].shape[0] > expected_prev_idx
			)
			has_prev_from_hist = (
				hist_l is not None
				and "samples" in hist_l
				and torch.is_tensor(hist_l["samples"])
				and hist_l["samples"].ndim == 4
				and hist_l["samples"].shape[0] > 0
			)
			if not (has_prev_from_acc or has_prev_from_hist):
				raise RuntimeError(
					"APLatentLoopOpen expected previous processed latent at "
					f"iteration {idx}, but none was provided. "
					"Ensure loop_close.processed_latent is connected to the actual processed output."
				)
		_loop_debug(
			"LatentLoopOpen "
			f"uid={unique_id} idx={idx}/{max(total - 1, 0)} total={total} history={hlen} "
			f"latent_batch={int(samples.shape[0])} custom_latents={'yes' if custom_latents is not None else 'no'}"
		)

		prev_latents = []
		prev_masks = []
		for i in range(5):
			prev_abs_idx = idx - (i + 1)
			has_acc_latent = (
				prev_abs_idx >= 0
				and acc_l is not None
				and "samples" in acc_l
				and torch.is_tensor(acc_l["samples"])
				and acc_l["samples"].ndim == 4
				and acc_l["samples"].shape[0] > prev_abs_idx
			)
			has_acc_mask = prev_abs_idx >= 0 and acc_m is not None and acc_m.shape[0] > prev_abs_idx
			has_hist_latent = (
				i < hlen
				and hist_l is not None
				and "samples" in hist_l
				and torch.is_tensor(hist_l["samples"])
				and hist_l["samples"].ndim == 4
				and hist_l["samples"].shape[0] > i
			)
			has_hist_mask = i < hlen and hist_m is not None and hist_m.shape[0] > i
			if has_acc_latent:
				prev_latents.append(_slice_latent_batch(acc_l, prev_abs_idx))
			elif has_hist_latent:
				prev_latents.append(_slice_latent_batch(hist_l, i))
			elif return_first_when_no_previous_available:
				prev_latents.append(_slice_latent_batch(first_latent, 0))
			else:
				prev_latents.append(_zero_latent_like(current_latent))

			if has_acc_mask:
				prev_masks.append(_slice_mask_batch(acc_m, prev_abs_idx).clone())
			elif has_hist_mask:
				prev_masks.append(hist_m[i : i + 1].clone())
			elif return_first_when_no_previous_available:
				prev_masks.append(first_mask.clone())
			else:
				prev_masks.append(torch.zeros_like(current_mask))

		if return_first_when_no_previous_available and idx <= 0:
			previous_latent = _slice_latent_batch(first_latent, 0)
			previous_mask = first_mask.clone()

		loop_token = {
			"open_node": str(unique_id) if unique_id is not None else None,
			"loop_type": "latent",
			"iteration_index": idx,
			"iteration_count": total,
			"history_count": hlen,
		}
		_loop_debug(
			"LatentLoopOpenToken "
			f"uid={unique_id} open_node={loop_token['open_node']} idx={idx} total={total}"
		)

		return (
			loop_token,
			current_latent,
			current_mask,
			first_latent,
			previous_latent,
			previous_mask,
			prev_latents[0],
			prev_latents[1],
			prev_latents[2],
			prev_latents[3],
			prev_latents[4],
			prev_masks[0],
			prev_masks[1],
			prev_masks[2],
			prev_masks[3],
			prev_masks[4],
			custom_latent,
			current_additional_data,
			idx,
		)


class APLatentLoopClose:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"loop_token": ("FLOW_CONTROL",),
				"processed_latent": ("LATENT",),
			},
			"optional": {
				"processed_mask": ("MASK",),
				"additional_data": ("*",),
			},
			"hidden": {
				"dynprompt": "DYNPROMPT",
				"unique_id": "UNIQUE_ID",
				"all_processed_latents": ("LATENT",),
				"all_processed_masks": ("MASK",),
				"all_processed_additional_data": ("*",),
				"processed_history_latents": ("LATENT",),
				"processed_history_masks": ("MASK",),
			},
		}

	RETURN_TYPES = ("LATENT", "MASK", "*")
	RETURN_NAMES = ("processed_latents", "processed_masks", "processed_additional_data")
	FUNCTION = "loop_close"
	CATEGORY = "AP_OpticalFlow"

	def loop_close(
		self,
		loop_token,
		processed_latent,
		processed_mask=None,
		additional_data=None,
		dynprompt=None,
		unique_id=None,
		all_processed_latents=None,
		all_processed_masks=None,
		all_processed_additional_data=None,
		processed_history_latents=None,
		processed_history_masks=None,
	):
		token_idx, token_total, token_hist = _loop_token_state(loop_token)
		current_live_open = _linked_input_parent_id(dynprompt, unique_id, "loop_token") if dynprompt is not None and unique_id is not None else None

		lat_entry_in, lat_samples_in = _ensure_latent(processed_latent)
		processed_batch_len = int(lat_samples_in.shape[0])

		total = _safe_int(token_total, 0)
		if total <= 0 and all_processed_latents is not None:
			_, acc_samples = _ensure_latent(all_processed_latents)
			total = int(acc_samples.shape[0])
		total = max(1, total if total > 0 else processed_batch_len)
		idx = _normalize_loop_index(token_idx, total)

		proc_idx = _normalize_loop_index(idx, processed_batch_len) if processed_batch_len > 1 else 0
		lat_entry = _slice_latent_batch(lat_entry_in, proc_idx)
		_, proc_samples = _ensure_latent(lat_entry)
		pmask = _slice_mask_batch(_ensure_mask_for_latent_batch(processed_mask, lat_samples_in), proc_idx)
		_loop_debug(
			"LatentLoopClose "
			f"uid={unique_id} open_node={current_live_open} token_idx={token_idx} token_total={token_total} "
			f"resolved_idx={idx} resolved_total={total} processed_batch={processed_batch_len} proc_idx={proc_idx}"
		)

		acc_lat = _init_latent_accumulator(all_processed_latents if all_processed_latents is not None else lat_entry, total)
		acc_mask = _init_mask_accumulator(all_processed_masks, total, pmask)
		acc_additional = _accumulate_additional_data(all_processed_additional_data, additional_data, idx, total)
		_store_latent_accumulator(acc_lat, lat_entry, idx)
		_store_mask_accumulator(acc_mask, pmask, idx)

		hlen = max(0, min(5, _safe_int(token_hist, 3)))

		hist_lat = _prepend_latent_history(processed_history_latents, lat_entry, hlen)
		hist_mask = _prepend_history_batch(
			_ensure_mask_bhw(processed_history_masks) if processed_history_masks is not None else None,
			pmask,
			hlen,
		)

		if idx >= total - 1:
			_loop_debug(
				"LatentLoopCloseFinish "
				f"uid={unique_id} idx={idx} total={total} returning_accumulated_batch={int(acc_mask.shape[0])}"
			)
			return _latent_dict_to_cpu(acc_lat), acc_mask.cpu(), _move_any_to_cpu(acc_additional)

		next_idx = idx + 1
		next_loop_token = dict(loop_token)
		live_open_node = current_live_open
		if live_open_node is not None:
			next_loop_token["open_node"] = live_open_node
		next_loop_token["iteration_index"] = next_idx
		next_loop_token["iteration_count"] = total
		next_loop_token["history_count"] = hlen
		_loop_debug(
			"LatentLoopCloseRecurse "
			f"uid={unique_id} next_idx={next_idx} total={total} next_open={next_loop_token.get('open_node')}"
		)

		recurse, expand = _build_loop_recurse(
			next_loop_token,
			dynprompt,
			unique_id,
			close_overrides={
				"all_processed_latents": acc_lat,
				"all_processed_masks": acc_mask,
				"all_processed_additional_data": acc_additional,
				"processed_history_latents": hist_lat,
				"processed_history_masks": hist_mask,
			},
			open_overrides={
				"iteration_index": next_idx,
				"all_processed_latents": acc_lat,
				"all_processed_masks": acc_mask,
				"processed_history_latents": hist_lat,
				"processed_history_masks": hist_mask,
			},
		)

		return {
			"result": (recurse.out(0), recurse.out(1), recurse.out(2)),
			"expand": expand,
		}


class APTemporalBlendImages:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"image_1": ("IMAGE",),
				"blend_mode": (
					["weighted_mean", "similarity_weighted", "median", "trimmed_mean", "robust_huber"],
					{"default": "similarity_weighted"},
				),
				"recency_decay": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 4.0, "step": 0.01}),
				"trim_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.49, "step": 0.01}),
				"similarity_sigma": ("FLOAT", {"default": 0.1, "min": 0.0001, "max": 1.0, "step": 0.0001}),
				"robust_delta": ("FLOAT", {"default": 0.08, "min": 0.0001, "max": 1.0, "step": 0.0001}),
				"detail_preservation": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
				"clamp_output": ("BOOLEAN", {"default": True}),
			},
			"optional": {
				"image_2": ("IMAGE",),
				"image_3": ("IMAGE",),
				"image_4": ("IMAGE",),
				"image_5": ("IMAGE",),
				"mask": ("MASK",),
			},
		}

	RETURN_TYPES = ("IMAGE", "MASK")
	RETURN_NAMES = ("blended_images", "blend_mask")
	FUNCTION = "blend"
	CATEGORY = "AP_OpticalFlow"

	def blend(
		self,
		image_1,
		blend_mode="similarity_weighted",
		recency_decay=0.35,
		trim_ratio=0.2,
		similarity_sigma=0.1,
		robust_delta=0.08,
		detail_preservation=0.2,
		clamp_output=True,
		image_2=None,
		image_3=None,
		image_4=None,
		image_5=None,
		mask=None,
	):
		frames = _align_image_sequence([image_1, image_2, image_3, image_4, image_5])
		current = frames[0]
		stack = torch.stack(frames, dim=0)

		blended = _temporal_reduce_stack(
			stack,
			blend_mode,
			recency_decay=recency_decay,
			trim_ratio=trim_ratio,
			similarity_sigma=similarity_sigma,
			robust_delta=robust_delta,
			channel_dim=-1,
		)

		detail = float(detail_preservation)
		detail = max(0.0, min(1.0, detail))
		blended = (current * detail) + (blended * (1.0 - detail))

		if mask is not None:
			blend_mask = _ensure_mask_for_image_batch(mask, current).clamp(0.0, 1.0)
			alpha = blend_mask.unsqueeze(-1)
			blended = (blended * alpha) + (current * (1.0 - alpha))
		else:
			blend_mask = torch.ones((current.shape[0], current.shape[1], current.shape[2]), dtype=current.dtype, device=current.device)

		if clamp_output:
			blended = blended.clamp(0.0, 1.0)

		return blended.cpu(), blend_mask.cpu()


class APTemporalBlendLatents:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"latent_1": ("LATENT",),
				"favor_previous_frame_when_few": ("BOOLEAN", {"default": False}),
				"blend_mode": (
					["weighted_mean", "similarity_weighted", "median", "trimmed_mean", "robust_huber"],
					{"default": "similarity_weighted"},
				),
				"recency_decay": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 4.0, "step": 0.01}),
				"trim_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.49, "step": 0.01}),
				"similarity_sigma": ("FLOAT", {"default": 0.1, "min": 0.0001, "max": 2.0, "step": 0.0001}),
				"robust_delta": ("FLOAT", {"default": 0.08, "min": 0.0001, "max": 2.0, "step": 0.0001}),
				"detail_preservation": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
			},
			"optional": {
				"latent_2": ("LATENT",),
				"latent_3": ("LATENT",),
				"latent_4": ("LATENT",),
				"latent_5": ("LATENT",),
				"mask": ("MASK",),
			},
		}

	RETURN_TYPES = ("LATENT", "MASK")
	RETURN_NAMES = ("blended_latent", "blend_mask")
	FUNCTION = "blend"
	CATEGORY = "AP_OpticalFlow"

	def blend(
		self,
		latent_1,
		favor_previous_frame_when_few=False,
		blend_mode="similarity_weighted",
		recency_decay=0.35,
		trim_ratio=0.2,
		similarity_sigma=0.1,
		robust_delta=0.08,
		detail_preservation=0.15,
		latent_2=None,
		latent_3=None,
		latent_4=None,
		latent_5=None,
		mask=None,
	):
		template, samples = _align_latent_sequence([latent_1, latent_2, latent_3, latent_4, latent_5])
		current = samples[0]
		stack = torch.stack(samples, dim=0)
		num_frames = int(stack.shape[0])

		first_frame_boost = 1.0
		if bool(favor_previous_frame_when_few) and 1 < num_frames <= 3:
			# Small temporal windows can lose continuity details; boost latent_1.
			first_frame_boost = 2.5

		blended = _temporal_reduce_stack(
			stack,
			blend_mode,
			recency_decay=recency_decay,
			trim_ratio=trim_ratio,
			similarity_sigma=similarity_sigma,
			robust_delta=robust_delta,
			channel_dim=2,
			first_frame_boost=first_frame_boost,
		)

		detail = float(detail_preservation)
		detail = max(0.0, min(1.0, detail))
		if bool(favor_previous_frame_when_few) and 1 < num_frames <= 3:
			# Ensure previous-frame detail is favored in 2-3 frame scenarios.
			min_detail = 0.75 if num_frames == 2 else 0.65
			detail = max(detail, min_detail)
		blended = (current * detail) + (blended * (1.0 - detail))

		if mask is not None:
			blend_mask = _ensure_mask_bhw(mask)
			if blend_mask.shape[0] != current.shape[0]:
				if blend_mask.shape[0] == 1 and current.shape[0] > 1:
					blend_mask = blend_mask.repeat(current.shape[0], 1, 1)
				elif current.shape[0] == 1 and blend_mask.shape[0] > 1:
					blend_mask = blend_mask[:1]
				else:
					raise ValueError(f"Batch mismatch: latent={current.shape[0]} mask={blend_mask.shape[0]}")

			if blend_mask.shape[1] != current.shape[2] or blend_mask.shape[2] != current.shape[3]:
				blend_mask = F.interpolate(
					blend_mask.unsqueeze(1),
					size=(current.shape[2], current.shape[3]),
					mode="nearest",
				).squeeze(1)

			blend_mask = blend_mask.clamp(0.0, 1.0)
			alpha = blend_mask.unsqueeze(1)
			blended = (blended * alpha) + (current * (1.0 - alpha))
		else:
			blend_mask = torch.ones((current.shape[0], current.shape[2], current.shape[3]), dtype=current.dtype, device=current.device)

		out_latent = _copy_latent_with_samples(template, blended)
		return _latent_dict_to_cpu(out_latent), blend_mask.cpu()


class APTemporalBlendLatentsSimple:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"current_latent": ("LATENT",),
				"previous_latent_1": ("LATENT",),
				"blend_percentage": ("FLOAT", {"default": 35.0, "min": 0.0, "max": 100.0, "step": 0.1}),
				"favor_prev_frame_when_few": ("BOOLEAN", {"default": False}),
			},
			"optional": {
				"previous_latent_2": ("LATENT",),
				"previous_latent_3": ("LATENT",),
				"previous_latent_4": ("LATENT",),
				"mask": ("MASK",),
			},
		}

	RETURN_TYPES = ("LATENT", "MASK")
	RETURN_NAMES = ("blended_latent", "blend_mask")
	FUNCTION = "blend"
	CATEGORY = "AP_OpticalFlow"

	def blend(
		self,
		current_latent,
		previous_latent_1,
		blend_percentage=35.0,
		favor_prev_frame_when_few=False,
		previous_latent_2=None,
		previous_latent_3=None,
		previous_latent_4=None,
		mask=None,
	):
		# Input order is explicit: current frame base + previous frame history.
		template, samples = _align_latent_sequence(
			[current_latent, previous_latent_1, previous_latent_2, previous_latent_3, previous_latent_4]
		)
		current = samples[0]
		previous_samples = samples[1:]

		if not previous_samples:
			out_latent = _copy_latent_with_samples(template, current)
			blend_mask = torch.zeros(
				(current.shape[0], current.shape[2], current.shape[3]),
				dtype=current.dtype,
				device=current.device,
			)
			return _latent_dict_to_cpu(out_latent), blend_mask.cpu()

		if len(previous_samples) == 1:
			previous_mix = previous_samples[0]
		else:
			prev_stack = torch.stack(previous_samples, dim=0)
			first_boost = 1.0
			if bool(favor_prev_frame_when_few) and len(previous_samples) <= 3:
				# Favor nearest previous frame when temporal context is small.
				first_boost = 2.5
			previous_mix = _temporal_reduce_stack(
				prev_stack,
				"weighted_mean",
				recency_decay=0.8,
				channel_dim=2,
				first_frame_boost=first_boost,
			)

		alpha_scalar = max(0.0, min(1.0, float(blend_percentage) / 100.0))

		if mask is not None:
			blend_mask = _ensure_mask_bhw(mask)
			if blend_mask.shape[0] != current.shape[0]:
				if blend_mask.shape[0] == 1 and current.shape[0] > 1:
					blend_mask = blend_mask.repeat(current.shape[0], 1, 1)
				elif current.shape[0] == 1 and blend_mask.shape[0] > 1:
					blend_mask = blend_mask[:1]
				else:
					raise ValueError(f"Batch mismatch: latent={current.shape[0]} mask={blend_mask.shape[0]}")

			if blend_mask.shape[1] != current.shape[2] or blend_mask.shape[2] != current.shape[3]:
				blend_mask = F.interpolate(
					blend_mask.unsqueeze(1),
					size=(current.shape[2], current.shape[3]),
					mode="nearest",
				).squeeze(1)
			blend_mask = blend_mask.clamp(0.0, 1.0)
		else:
			blend_mask = torch.ones(
				(current.shape[0], current.shape[2], current.shape[3]),
				dtype=current.dtype,
				device=current.device,
			)

		alpha = (blend_mask.unsqueeze(1) * alpha_scalar).clamp(0.0, 1.0)
		blended = (previous_mix * alpha) + (current * (1.0 - alpha))

		out_latent = _copy_latent_with_samples(template, blended)
		return _latent_dict_to_cpu(out_latent), blend_mask.cpu()


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
		flow_data = _materialize_flow_data(flow_data)
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


class APBridgeSave:
	def __init__(self):
		self.output_dir = _get_output_directory()

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"filename_prefix": ("STRING", {"default": "AP_Bridge/bridge"}),
				"save_images": ("BOOLEAN", {"default": True}),
				"save_masks": ("BOOLEAN", {"default": True}),
				"save_image_mask_together": ("BOOLEAN", {"default": False}),
				"compress_level": ("INT", {"default": 4, "min": 0, "max": 9, "step": 1}),
			},
			"optional": {
				"images": ("IMAGE",),
				"masks": ("MASK",),
			},
		}

	RETURN_TYPES = ("IMAGE", "MASK")
	RETURN_NAMES = ("images", "masks")
	FUNCTION = "bridge_save"
	CATEGORY = "AP_OpticalFlow"

	def bridge_save(
		self,
		filename_prefix="AP_Bridge/bridge",
		save_images=True,
		save_masks=True,
		save_image_mask_together=False,
		compress_level=4,
		images=None,
		masks=None,
	):
		img_out, mask_out = _bridge_passthrough(images, masks, "AP Bridge Save")

		images_to_save = images if bool(save_images) else None
		masks_to_save = masks if bool(save_masks) else None
		save_combined = bool(save_image_mask_together)
		if save_combined and (images is None or masks is None):
			raise ValueError("AP Bridge Save: save_image_mask_together requires both IMAGE and MASK inputs")

		source_images = images if (bool(save_images) or save_combined) else None
		source_masks = masks if (bool(save_masks) or save_combined) else None

		results = []
		if images_to_save is not None or masks_to_save is not None or save_combined:
			results = _bridge_save_images_and_masks(
				source_images,
				source_masks,
				filename_prefix,
				self.output_dir,
				output_type="output",
				compress_level=compress_level,
				write_images=bool(save_images),
				write_masks=bool(save_masks),
				save_combined=save_combined,
			)

		return {"ui": {"images": results}, "result": (img_out, mask_out)}


class APBridgePreviewBatch:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"preview_mode": (["image", "mask", "composite"], {"default": "image"}),
			},
			"optional": {
				"images": ("IMAGE",),
				"masks": ("MASK",),
			},
			"hidden": {
				"unique_id": "UNIQUE_ID",
			},
		}

	RETURN_TYPES = ("IMAGE", "MASK")
	RETURN_NAMES = ("images", "masks")
	FUNCTION = "preview_batch"
	CATEGORY = "AP_OpticalFlow"
	OUTPUT_NODE = True

	def preview_batch(self, preview_mode="image", images=None, masks=None, unique_id=None):
		# Preview bridge is pure passthrough and does not write any files.
		img_out, mask_out = _bridge_passthrough(images, masks, "AP Bridge Preview Batch")
		_bridge_emit_preview_batch(images=images, masks=masks, unique_id=unique_id, preview_mode=preview_mode)
		return img_out, mask_out


class AP_ImageMaskInpaintCrop:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"images": ("IMAGE",),
				"masks": ("MASK",),
				"padding": ("INT", {"default": 32, "min": 0, "max": 2048, "step": 1}),
				"crop_universal_box": ("BOOLEAN", {"default": False}),
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
		crop_universal_box=False,
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

		universal_bbox = None
		if bool(crop_universal_box):
			batch_bin = msk > thr
			if torch.any(batch_bin):
				idx = torch.where(batch_bin)
				ys = idx[1]
				xs = idx[2]
				y0u = max(0, int(ys.min().item()) - padding)
				y1u = min(h, int(ys.max().item()) + 1 + padding)
				x0u = max(0, int(xs.min().item()) - padding)
				x1u = min(w, int(xs.max().item()) + 1 + padding)
			else:
				y0u, y1u, x0u, x1u = 0, h, 0, w
			universal_bbox = (y0u, y1u, x0u, x1u)

		for i in range(b):
			m = msk[i]
			if universal_bbox is not None:
				y0, y1, x0, x1 = universal_bbox
			else:
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

			resize_requested = (ow > 0) or (oh > 0)
			target_w = int(bbox_w)
			target_h = int(bbox_h)
			resize_anchor = "none"
			scale = 1.0

			if resize_requested:
				if ow > 0 and oh > 0:
					target_h_from_w = max(1, int(round((float(ow) * float(bbox_h)) / float(max(1, bbox_w)))))
					target_w_from_h = max(1, int(round((float(oh) * float(bbox_w)) / float(max(1, bbox_h)))))

					mismatch_h = abs(float(target_h_from_w - oh)) / float(max(1, oh))
					mismatch_w = abs(float(target_w_from_h - ow)) / float(max(1, ow))

					if mismatch_h <= mismatch_w:
						resize_anchor = "width"
						target_w = int(ow)
						target_h = int(target_h_from_w)
						scale = float(target_w) / float(max(1, bbox_w))
					else:
						resize_anchor = "height"
						target_h = int(oh)
						target_w = int(target_w_from_h)
						scale = float(target_h) / float(max(1, bbox_h))
				elif ow > 0:
					resize_anchor = "width"
					target_w = int(ow)
					target_h = max(1, int(round((float(ow) * float(bbox_h)) / float(max(1, bbox_w)))))
					scale = float(target_w) / float(max(1, bbox_w))
				else:
					resize_anchor = "height"
					target_h = int(oh)
					target_w = max(1, int(round((float(oh) * float(bbox_w)) / float(max(1, bbox_h)))))
					scale = float(target_h) / float(max(1, bbox_h))

			do_resize = resize_requested
			if resize_requested and upscale_only:
				do_resize = scale > 1.0 + 1e-6

			if do_resize:
				proc_img = self._resize_image(crop_img, target_h, target_w, interpolation)
				proc_m = self._resize_mask(crop_m, target_h, target_w)
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
					"target_h": int(target_h),
					"target_w": int(target_w),
					"resize_anchor": resize_anchor,
					"resize_scale": float(scale),
					"resize_requested": bool(resize_requested),
					"valid_h": int(vh),
					"valid_w": int(vw),
					"resized": bool(do_resize),
				}
			)

		pad_imgs = []
		pad_msks = []
		out_h8 = int(((max_h + 7) // 8) * 8)
		out_w8 = int(((max_w + 7) // 8) * 8)
		for i in range(b):
			ci = crop_imgs[i]
			cm = crop_msks[i]
			vh, vw = int(ci.shape[0]), int(ci.shape[1])

			canvas_i = torch.zeros((out_h8, out_w8, c), dtype=ci.dtype, device=ci.device)
			canvas_m = torch.zeros((out_h8, out_w8), dtype=cm.dtype, device=cm.device)
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
			"requested_out_width": int(ow),
			"requested_out_height": int(oh),
			"resize_policy": "aspect_preserve",
			"output_h": int(out_h8),
			"output_w": int(out_w8),
			"output_multiple": 8,
			"crop_universal_box": bool(crop_universal_box),
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
	"APApplyRAFTOpticalFlowLatent": APApplyRAFTOpticalFlowLatent,
	"APApplyRAFTOpticalFlowLatentMasked": APApplyRAFTOpticalFlowLatentMasked,
	"APWarpImageAndMaskByRAFTFlow": APWarpImageAndMaskByRAFTFlow,
	"APFlowComposite": APFlowComposite,
	"APWarpMaskedCompositeOcclusion": APWarpMaskedCompositeOcclusion,
	"APImageLoopOpen": APImageLoopOpen,
	"APImageLoopClose": APImageLoopClose,
	"APLatentLoopOpen": APLatentLoopOpen,
	"APLatentLoopClose": APLatentLoopClose,
	"APTemporalBlendImages": APTemporalBlendImages,
	"APTemporalBlendLatents": APTemporalBlendLatents,
	"APTemporalBlendLatentsSimple": APTemporalBlendLatentsSimple,
	"APIndexer": APIndexer,
	"APSelectFlowByIndex": APSelectFlowByIndex,
	"APSaveOpticalFlow": APSaveOpticalFlow,
	"APLoadOpticalFlow": APLoadOpticalFlow,
	"APBridgeSave": APBridgeSave,
	"APBridgePreviewBatch": APBridgePreviewBatch,
	"AP_ImageMaskInpaintCrop": AP_ImageMaskInpaintCrop,
	"AP_ImageMaskStitch": AP_ImageMaskStitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"APGetRAFTOpticalFlow": "AP Get RAFT Optical Flow",
	"APApplyRAFTOpticalFlow": "AP Apply RAFT Optical Flow",
	"APFlowOcclusionMask": "AP Flow Occlusion Mask",
	"APApplyRAFTOpticalFlowMasked": "AP Apply RAFT Optical Flow (Masked)",
	"APApplyRAFTOpticalFlowLatent": "AP Apply RAFT Optical Flow (Latent)",
	"APApplyRAFTOpticalFlowLatentMasked": "AP Apply RAFT Optical Flow (Latent, Masked)",
	"APWarpImageAndMaskByRAFTFlow": "AP Warp Image + Mask by RAFT Flow",
	"APFlowComposite": "AP Flow Composite",
	"APWarpMaskedCompositeOcclusion": "AP Warp Masked Composite Blend (Occlusion)",
	"APImageLoopOpen": "AP Loop Open",
	"APImageLoopClose": "AP Loop Close",
	"APLatentLoopOpen": "AP Loop Open (Latent)",
	"APLatentLoopClose": "AP Loop Close (Latent)",
	"APTemporalBlendImages": "AP Temporal Blend Images",
	"APTemporalBlendLatents": "AP Temporal Blend Latents",
	"APTemporalBlendLatentsSimple": "AP Temporal Blend Latents Simple",
	"APIndexer": "AP Indexer",
	"APSelectFlowByIndex": "AP Select Flow By Index",
	"APSaveOpticalFlow": "AP Save Optical Flow",
	"APLoadOpticalFlow": "AP Load Optical Flow",
	"APBridgeSave": "AP Bridge Save",
	"APBridgePreviewBatch": "AP Bridge Preview Batch",
	"AP_ImageMaskInpaintCrop": "AP Image Mask Inpaint Crop",
	"AP_ImageMaskStitch": "AP Image Mask Stitch",
}
