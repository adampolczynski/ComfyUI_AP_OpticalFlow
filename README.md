# AP_OpticalFlow

`AP_OpticalFlow` is a RAFT-based optical flow node pack for ComfyUI focused on real workflow stability: correct input/output contracts, batch-safe behavior, and loop/index support.

## Motivation

I got tired of incomplete optical flow packages that did not support correct configurations or reliable inputs/outputs in real ComfyUI graphs.

This pack exists to make optical-flow workflows practical for production-style use: temporal consistency, masked warping, index-driven loops, and clean handoff between nodes.

## Recent Changes

- Added full index-aware loop support through `current_frame_index` on flow-application nodes.
- Added `flow_skip` and `frames_skip` controls to handle offset starts and delayed flow activation.
- Added explicit batch alignment modes for flow application:
	- `auto`
	- `by_index`
	- `repeat_image`
- Added `AP Indexer` (`APIndexer`) for persistent frame indexing in iterative pipelines.
- Added `AP Select Flow By Index` (`APSelectFlowByIndex`) to pick the correct flow entry for a frame.
- Added flow persistence nodes:
	- `AP Save Optical Flow` (`APSaveOpticalFlow`)
	- `AP Load Optical Flow` (`APLoadOpticalFlow`)
- Added frame-wise recursive loop nodes for image and latent pipelines:
	- `AP Loop Open` / `AP Loop Close`
	- `AP Loop Open (Latent)` / `AP Loop Close (Latent)`
- Added temporal-consistency blend nodes:
	- `AP Temporal Blend Images`
	- `AP Temporal Blend Latents`
- Added lockstep `additional_data` iteration support to image/latent loop nodes,
	allowing any parallel payload (for example LATENT with IMAGE loop).
- Added inpaint rectangle workflow nodes with batch support:
	- `AP ImageMask InpaintCrop` (`AP_ImageMaskInpaintCrop`)
	- `AP ImageMask Stitch` (`AP_ImageMaskStitch`)
- Added `AP_STITCH` metadata handoff type for reliable crop->inpaint->stitch roundtrips.
- Improved file path handling for load/save with output/input/current-dir resolution.
- Improved runtime robustness for some CUDA/cuDNN setups by retrying RAFT inference in float32 with cuDNN disabled when needed.

## Included Nodes

- `AP Get RAFT Optical Flow` (`APGetRAFTOpticalFlow`)
- `AP Apply RAFT Optical Flow` (`APApplyRAFTOpticalFlow`)
- `AP Flow Occlusion Mask` (`APFlowOcclusionMask`)
- `AP Apply RAFT Optical Flow (Masked)` (`APApplyRAFTOpticalFlowMasked`)
- `AP Apply RAFT Optical Flow (Latent)` (`APApplyRAFTOpticalFlowLatent`)
- `AP Apply RAFT Optical Flow (Latent, Masked)` (`APApplyRAFTOpticalFlowLatentMasked`)
- `AP Warp Image + Mask by RAFT Flow` (`APWarpImageAndMaskByRAFTFlow`)
- `AP Flow Composite` (`APFlowComposite`)
- `AP Loop Open` (`APImageLoopOpen`)
- `AP Loop Close` (`APImageLoopClose`)
- `AP Loop Open (Latent)` (`APLatentLoopOpen`)
- `AP Loop Close (Latent)` (`APLatentLoopClose`)
- `AP Temporal Blend Images` (`APTemporalBlendImages`)
- `AP Temporal Blend Latents` (`APTemporalBlendLatents`)
- `AP Indexer` (`APIndexer`)
- `AP Select Flow By Index` (`APSelectFlowByIndex`)
- `AP Save Optical Flow` (`APSaveOpticalFlow`)
- `AP Load Optical Flow` (`APLoadOpticalFlow`)
- `AP Image Mask Inpaint Crop` (`AP_ImageMaskInpaintCrop`)
- `AP Image Mask Stitch` (`AP_ImageMaskStitch`)

## Install

1. Put this folder in ComfyUI custom nodes (or symlink it):

```bash
cp -r tools/AP_OpticalFlow custom_nodes/AP_OpticalFlow
```

2. Install dependencies in your ComfyUI environment:

```bash
python -m pip install -r custom_nodes/AP_OpticalFlow/requirements.txt
```

3. Restart ComfyUI.

## ComfyUI Manager Support

This node pack includes Manager/registry metadata in `pyproject.toml`:

- `project.name = "comfyui-ap-optical-flow"`
- `tool.comfy.PublisherId = "adampolczynski"`
- `tool.comfy.DisplayName = "AP Optical Flow"`

If you want to install/update through ComfyUI Manager, use the repository URL:

```text
https://github.com/adampolczynski/ComfyUI_AP_OpticalFlow
```

Note: to make this appear in Manager's public install catalog, it also needs to be published in the Comfy registry / Manager node list.

## Quick Workflows

### A) Temporal warp and blend

1. `APGetRAFTOpticalFlow` with frame A and frame B.
2. `APFlowOcclusionMask` from `flow_data`.
3. `APApplyRAFTOpticalFlowMasked` (or `APApplyRAFTOpticalFlow`) to warp with flow.
4. `APFlowComposite` to blend warped result back using valid/occlusion masks.

Professional anti-blur compositing (recommended):
- Connect `warped_mask` output from `APApplyRAFTOpticalFlowMasked` to `APFlowComposite.effect_mask`.
- Set `alpha_mode = flow_confidence_x_mask`.
- Enable `use_difference_gate = true` with a low `difference_threshold` (for example `0.005 - 0.02`).
- This prevents blending untouched background/non-masked areas and keeps changes focused on actually warped regions.

![Temporal flow load/apply example](examples/optical_flow_load_apply.png)

### B) Loop/index pipeline

1. `APIndexer` to produce `current_frame_index`.
2. Feed `current_frame_index` into flow nodes that support it.
3. Use `flow_skip` and `frames_skip` to align flow timing with your loop start.
4. Optionally use `APSelectFlowByIndex` for explicit flow slicing.

### B.1) Save/Load flow cache

1. Use `AP Save Optical Flow` to write `flow_data` to `.pt` in your Comfy output path.
2. Reuse it later with `AP Load Optical Flow` to skip recomputing flow.

![Save optical flow example](examples/save_optical_flow.png)

### C) Inpaint crop/stitch pipeline

1. `AP Image Mask Inpaint Crop` to extract padded crop + crop mask + `AP_STITCH` data.
2. Run your inpaint model on the cropped image/mask.
3. `AP Image Mask Stitch` to place the inpainted crop back into the original frame.

![Inpaint crop and stitch example](examples/inpaint_crop.png)

### D) Latent warp pipeline (with optional masking)

1. Build flow with `APGetRAFTOpticalFlow` from neighboring frames.
2. Warp latent directly with `AP Apply RAFT Optical Flow (Latent)`.
3. For region-limited latent warping, use `AP Apply RAFT Optical Flow (Latent, Masked)` and provide your mask.
4. Use `flow_skip` / `frames_skip` / `current_frame_index` exactly like image-flow nodes for loop pipelines.

### E) Recursive loop pipeline (images and latents)

`AP Loop Open` / `AP Loop Close` are made for frame-by-frame processing with feedback:

`AP Loop Open` outputs:
- `current_image` / `current_mask`
- `first_image`
- `previous_image` / `previous_mask` (unprocessed source timeline)
- `prev_processed_1 .. prev_processed_5` and matching masks (history depth controlled by `history_length`)
- `custom_frame` (optional index-based override source)
- `current_additional_data` (optional wildcard payload iterated at the same index)

`AP Loop Close` takes your processed result and feeds it into the next iteration automatically.
It can also collect `additional_data` through the loop and output `processed_additional_data` at the end.

Custom frame replacement:
- Connect optional `custom_frames` and set `custom_frame_indices` (comma-separated, e.g. `0,12,48`).
- `custom_frame` output provides the mapped replacement frame for the current index.
- Enable `apply_custom_replacement=true` to force `current_image` to use this replacement.

Latents use the same pattern:
- `AP Loop Open (Latent)` / `AP Loop Close (Latent)`
- same iteration behavior
- same up-to-5 processed history concept
- avoids repeated VAE encode/decode in iterative latent workflows

`additional_data` usage (both image and latent loops):
- Connect any type to `additional_data` on Loop Open.
- Loop Open emits `current_additional_data` aligned with `iteration_index`.
- Connect your per-iteration processed payload back to Loop Close `additional_data`.
- Final Loop Close output includes `processed_additional_data` aggregated over all iterations.
- Practical example: image loop + parallel latent loop payload in one recursion.

### F) Temporal consistency blend nodes

`AP Temporal Blend Images` and `AP Temporal Blend Latents` accept up to 5 inputs and blend with temporal-robust modes:

- `weighted_mean`: stable baseline, fastest
- `similarity_weighted`: per-pixel/feature weighting against current frame; best default for flicker reduction
- `median`: strong outlier suppression (good for sporadic artifacts)
- `trimmed_mean`: robust against outliers while preserving smoother gradients than median
- `robust_huber`: adaptive robust weighting; helps in unstable inpaint regions

Mask-aware blending:
- Optional `mask` limits temporal blend to masked regions only.
- Outside mask, output stays at current frame/latent.

Recommended temporal-inpaint setup:
1. Use loop nodes to process one frame at a time.
2. Keep `prev_processed_1..N` connected as blend history inputs.
3. Start with `blend_mode=similarity_weighted` and `recency_decay=0.25 - 0.45`.
4. Use mask-limited blending to prevent background drift.
5. Increase `detail_preservation` when blend becomes too soft.

### G) Examples and tests

- Example workflows and dedicated test workflows are currently in progress.
- Existing screenshots in `examples/` show core usage patterns, and JSON test graphs will be expanded next.

## Recommended Settings

For better quality (faces/eyes):
- `model_size`: `large`
- `compute_backward`: `true`
- `max_side`: `1536` (or `2048` if VRAM allows)
- `use_fp16`: `false` for best precision, `true` for speed

For low VRAM / memory stability:
- `model_residency`: `unload_after_use` (prevents persistent VRAM allocation)
- `compute_mode`: `sequential` (processes batch one frame-pair at a time)
- `compute_device`: `cpu` (slowest, but minimizes GPU pressure)
- `clear_cached_models_first`: `true` if you suspect old cached models are still resident

For RAM/storage offloading of flow data:
- `flow_offload`: `cpu_ram` keeps flow tensors in system RAM (default)
- `flow_offload`: `disk_storage` writes flow tensors to `.pt` and passes a lightweight AP_FLOW handle
- `disk_filename_prefix`: choose output subpath/name for auto-saved flow files

For masked warping:
- `strength`: `0.6 - 1.0`
- `mask_feather`: `3 - 8`
- `mask_strength`: `0.8 - 1.0`

For stitch blending:
- `blend_with_mask`: `true`
- `feather`: start low and increase only when seam is visible

## Notes

- Flow direction matters: if motion looks reversed, switch `ab/ba` or toggle `invert_flow`.
- Occlusion handling is important for reducing ghosting and stretching.
- In `auto` batch mode, index-based behavior is preferred when `current_frame_index` is connected.
- For fast motion or heavy blur, flow quality can still degrade.
- If you had VRAM stuck from older runs, run one flow pass with `model_residency=unload_after_use` and `clear_cached_models_first=true`.

## Known Limitations

- Depends on `torchvision` RAFT availability and compatible Torch/Torchvision versions.
- Very large inputs can be slow and VRAM-heavy.
- Not a replacement for full multi-shot tracking systems in extreme scenes.

## License

This project is open source under the MIT License. See `LICENSE`.
