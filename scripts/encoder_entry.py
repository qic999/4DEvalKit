"""Run native model scripts with an optional geometry-only detector fusion path.

The native joint model runs its detector AFTER all tracker frames. Detector
segmentation masks are neither consumed by the 3D exporter nor fed back into
tracking. This path keeps detector box/score fusion and omits those unused
segmentation computations. It does not disable the Full detector branch.
"""
import importlib.util
import os
from pathlib import Path
import runpy
import sys
from types import MethodType


def install_depth_scale_sampling_fix():
    """Initialize the native depth sample when foreground has <=100k pixels.

    The existing branch initializes only confidence, then later reads an
    unbound sampled_depth. Other branches and their random sampling stay native.
    This local inference hook leaves the model repository on disk unchanged.
    """
    import inspect
    import textwrap
    from sam3.modeling.backbones.spatial.depthanythingv3.model import DepthAnything3
    original = DepthAnything3.get_scale
    source = textwrap.dedent(inspect.getsource(original))
    needle = '        depth_conf_sampled = depth_conf_ns\n'
    if 'sampled_depth = non_sky_depth\n' in source:
        return
    if source.count(needle) != 1:
        raise RuntimeError('Unsupported native depth scaling implementation')
    source = source.replace(needle, needle+'        sampled_depth = non_sky_depth\n')
    namespace = {}
    exec(compile(source, '<4deval-depth-scale-sampling-fix>', 'exec'), original.__globals__, namespace)
    DepthAnything3.get_scale = namespace['get_scale']
    print('[4deval] initialized depth samples for native <=100k foreground branch', flush=True)


def install_chunked_rope(chunk_size=4, inplace_k=False):
    """Bound RoPE's float32/complex temporaries without changing attention."""
    import torch
    from sam3.model import decoder
    original = decoder.apply_rotary_enc

    def rotate(xq, xk, freqs_cis, repeat_freqs_k=False):
        if (xq.ndim != 4 or xq.shape[0] <= chunk_size or xk.shape[0] != xq.shape[0]
                or torch.is_grad_enabled() and (xq.requires_grad or xk.requires_grad)):
            return original(xq, xk, freqs_cis, repeat_freqs_k)
        qout = kout = None
        for start in range(0, xq.shape[0], chunk_size):
            stop = start + chunk_size
            qr, kr = original(xq[start:stop], xk[start:stop], freqs_cis, repeat_freqs_k)
            if qout is None:
                # Native per-item strides are independent of batch size.
                qout = torch.empty_strided(xq.shape, qr.stride(), dtype=qr.dtype, device=qr.device)
                # functional_attention immediately writes rotated keys into
                # this same projected-key slice. Reuse it in inference to
                # avoid retaining a second full temporal-key tensor.
                kout = xk if inplace_k else torch.empty_strided(xk.shape, kr.stride(), dtype=kr.dtype, device=kr.device)
            qout[start:stop].copy_(qr)
            kout[start:stop].copy_(kr)
            del qr, kr
        return qout, kout

    decoder.apply_rotary_enc = rotate
    print(f'[4deval] RoPE object batch={chunk_size}, inplace_k={inplace_k}; all temporal keys and attention operations preserved', flush=True)


def install_large_interpolation():
    """Split independent batch items past the CUDA bilinear INT_MAX limit."""
    import torch
    from sam3.sam import mask_decoder
    original = mask_decoder.F

    class FunctionalProxy:
        def __getattr__(self, name):
            return getattr(original, name)

        def interpolate(self, input, *args, **kwargs):
            if (input.ndim == 4 and input.numel() >= 2**31 - 1
                    and kwargs.get('mode') == 'bilinear'):
                # Batch slices retain the input strides and memory format. No
                # image/object is omitted and the per-pixel arithmetic is native.
                chunks = max(1, (2**31 - 2) // input[0].numel())
                if chunks >= input.shape[0]:
                    return original.interpolate(input, *args, **kwargs)
                return torch.cat([original.interpolate(x, *args, **kwargs)
                                  for x in input.split(chunks, dim=0)], dim=0)
            return original.interpolate(input, *args, **kwargs)

    mask_decoder.F = FunctionalProxy()
    print('[4deval] large bilinear tensors split along independent object batch dimension', flush=True)


def install_tracker_mask_offload(model):
    """Copy completed frame masks to CPU, retaining native temporal memory.

    Hook after tracking, correction and memory encoding finish for each frame.
    The exported 3D fields, object pointers, memory features and memory selection
    scores stay untouched. Unlike the native generic VOS offload option, this
    preserves the spatial model's additional bbox/pose output fields.
    """
    import torch
    tracker = model.tracker
    if model.training or tracker.training:
        raise ValueError('Tracker mask offload is evaluation-only')
    if tracker.offload_output_to_cpu_for_eval or tracker.trim_past_non_cond_mem_for_eval:
        raise ValueError('Mask offload requires the untrimmed native spatial outputs')
    original_trim = tracker._trim_output_and_memory
    mask_keys = {
        'pred_masks', 'pred_masks_high_res', 'multistep_pred_masks',
        'multistep_pred_masks_high_res', 'multistep_pred_multimasks',
        'multistep_pred_multimasks_high_res',
    }

    def offload(self, **kwargs):
        current = original_trim(**kwargs)
        memo = {}

        def cpu(value):
            if torch.is_tensor(value):
                if id(value) not in memo:
                    memo[id(value)] = value.cpu()
                return memo[id(value)]
            if isinstance(value, list):
                return [cpu(x) for x in value]
            if isinstance(value, tuple):
                return tuple(cpu(x) for x in value)
            return value

        for key in mask_keys & current.keys():
            current[key] = cpu(current[key])
        return current

    tracker._trim_output_and_memory = MethodType(offload, tracker)
    print('[4deval] completed tracker masks offloaded to CPU; temporal memory, object coverage and bbox/pose outputs unchanged', flush=True)
    return model


def install_box_only_detector_fusion(model):
    import torch
    if model.training or not model.freeze_detector:
        raise ValueError('Box-only detector fusion is an evaluation-only optimization')
    if not getattr(model, 'enable_detector_fusion', False):
        return model

    def no_segmentation(self, **kwargs):
        return None

    def fuse_boxes(self, input, stage_outputs):
        if not stage_outputs:
            return
        device = next(self.tracker.parameters()).device
        backbone_out = {'img_batch_all_stages': input.img_batch}
        backbone_out.update(self.detector.backbone.forward_image(input.img_batch))
        backbone_out.update(self.detector.backbone.forward_text(input.find_text_batch, device=device))
        for frame_idx, stage_out in enumerate(stage_outputs):
            if not isinstance(stage_out, dict):
                continue
            if frame_idx >= len(input.find_inputs):
                break
            find_input = input.find_inputs[frame_idx]
            # All detector classification/regression operations are unchanged.
            raw = self.detector.forward_grounding(
                backbone_out=backbone_out, find_input=find_input, find_target=None,
                geometric_prompt=self._build_geo_prompt_from_find_input(find_input))
            det = {key: raw[key] for key in ('pred_logits', 'pred_boxes', 'pred_boxes_xyxy')}
            del raw
            # Reuse native top-candidate selection and threshold handling. This
            # 1x1 placeholder is discarded; it is never exported or fused.
            logits = det['pred_logits']
            det['pred_masks'] = logits.new_zeros((*logits.shape[:2], 1, 1))
            _, boxes, scores = self._extract_detector_stage_prediction(det)
            self._fuse_detector_into_bbox_scores(stage_out, boxes, scores)
            del det, logits, boxes, scores

    model.detector._run_segmentation_heads = MethodType(no_segmentation, model.detector)
    model._run_detector_fusion = MethodType(fuse_boxes, model)
    print('[4deval] box-only detector fusion: preserve classification, box regression and bbox-score fusion; skip unused detector masks', flush=True)
    return model


def main():
    if len(sys.argv) < 2 or sys.argv[1].startswith('-'):
        os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
    script = Path(sys.argv[1]).resolve()
    sys.argv = [str(script), *sys.argv[2:]]
    sys.path.insert(0, str(script.parent))
    box_only = os.environ.get('FOURDEVAL_BOX_ONLY_FUSION') == '1'
    mask_offload = os.environ.get('FOURDEVAL_TRACKER_MASK_OFFLOAD') == '1'
    large_interpolation = os.environ.get('FOURDEVAL_LARGE_INTERPOLATION') == '1'
    chunked_rope = os.environ.get('FOURDEVAL_CHUNKED_ROPE') == '1'
    if script.name != 'scene_inference.py' or not (box_only or mask_offload or large_interpolation or chunked_rope):
        runpy.run_path(str(script), run_name='__main__')
        return
    spec = importlib.util.spec_from_file_location('fourdeval_native_scene', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if large_interpolation:
        install_large_interpolation()
    if chunked_rope:
        install_chunked_rope(inplace_k=os.environ.get('FOURDEVAL_INPLACE_ROPE') == '1')
    original_make_model = module.make_model

    def make_model(args):
        model = original_make_model(args)
        if mask_offload and args.model_profile == 'full' and not box_only:
            raise ValueError('Full mask offload requires box-only detector fusion')
        if args.model_profile == 'full' and box_only:
            install_box_only_detector_fusion(model)
        if mask_offload:
            install_tracker_mask_offload(model)
        return model

    module.make_model = make_model
    module.main()


if __name__ == '__main__':
    main()
