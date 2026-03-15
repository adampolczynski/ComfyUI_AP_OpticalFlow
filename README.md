# AP_OpticalFlow

I made this node pack because I needed better temporal consistency for face/eye edits in video.

`AP_OpticalFlow` adds RAFT-based optical flow nodes to ComfyUI so I can:
- compute flow between frames,
- visualize flow,
- warp images/masks with that flow,
- detect occlusion/confidence,
- and blend warped results back in a controlled way.

## What is included

- `AP Get RAFT Optical Flow` (`APGetRAFTOpticalFlow`)
- `AP Apply RAFT Optical Flow` (`APApplyRAFTOpticalFlow`)
- `AP Flow Occlusion Mask` (`APFlowOcclusionMask`)
- `AP Apply RAFT Optical Flow (Masked)` (`APApplyRAFTOpticalFlowMasked`)
- `AP Warp IMAGE+MASK by RAFT Flow` (`APWarpImageAndMaskByRAFTFlow`)
- `AP Flow Composite` (`APFlowComposite`)

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

## Quick workflow (the one I use most)

1. `APGetRAFTOpticalFlow` with frame A and frame B
2. `APFlowOcclusionMask` from `flow_data`
3. `APApplyRAFTOpticalFlowMasked` to warp only masked areas
4. `APFlowComposite` to blend warped result back using valid/occlusion masks

This gives cleaner temporal matching than naive per-frame edits.

## Recommended settings

For better quality (faces/eyes):
- `model_size`: `large`
- `compute_backward`: `true`
- `max_side`: `1536` (or `2048` if VRAM allows)
- `use_fp16`: `false` for best precision (slower), `true` for speed

For masked warping:
- `strength`: `0.6 - 1.0`
- `mask_feather`: `3 - 8`
- `mask_strength`: `0.8 - 1.0`

For compositing:
- `blend_strength`: `1.0`
- `feather`: `2 - 4`

## Notes

- Flow direction matters: if motion looks reversed, switch `ab/ba` or toggle `invert_flow`.
- Occlusion is important for reducing ghosting and stretching.
- For fast motion / motion blur, flow quality will still degrade.

## Known limitations

- Depends on `torchvision` RAFT availability.
- Very large inputs can be slow and VRAM-heavy.
- Not a replacement for full video tracking pipelines in extreme scenes.

## Why I made this

I wanted practical nodes for matching masked elements between neighboring frames and stabilizing edits through a shot, without building a full external pipeline every time.
