# Sources and attribution

The files listed in [docs/upstream_manifest.json](docs/upstream_manifest.json) originate from
[DeepCybo-PhysAI/PhysBrainEvalKit](https://github.com/DeepCybo-PhysAI/PhysBrainEvalKit), revision
`4b37ca2f184bd76b98827481a6739f89ea06d730`.
That project describes itself as based on and extending EmbodiedEvalKit.
The vendored benchmark and metric implementations retain their source content;
imports of `core.*` were rewritten to `core.physbrain.*` for isolation.
The point protocol document is copied from the same revision.

The upstream snapshot did not contain a top-level LICENSE file. This repository
does not assign a new license to the third-party files. Dataset terms remain those
of their owners; no benchmark media, model weights, or private API credentials are
included in this source tree.

The project-local VSI scorer is preserved separately, with source identity in
[docs/project_metric_source.json](docs/project_metric_source.json).

The direct perception adapters load the external
[WildDet3D evaluator](https://github.com/allenai/WildDet3D) at revision
`1b8aa52b6ff3f00d0ebfa07175efc0c0c440964a`. The evaluator is installed separately;
its model code and datasets are not vendored here. Dataset and evaluation
instructions are in [the perception guide](docs/perception_evaluation.md).

Dynamic annotation sources:

- [STI-Bench](https://mint-sjtu.github.io/STI-Bench.io/), [dataset](https://huggingface.co/datasets/MINT-SJTU/STI-Bench)
- [VLM4D](https://vlm4d.github.io/), [dataset](https://huggingface.co/datasets/shijiezhou/VLM4D)
- [DSI-Bench](https://dsibench.github.io/), [dataset](https://huggingface.co/datasets/Viglong/DSI-Bench)
- [MotionBench](https://github.com/zai-org/MotionBench), [dataset](https://huggingface.co/datasets/THUDM/MotionBench)
- [TempCompass](https://github.com/llyx97/TempCompass)
- [CLEVRER](https://github.com/chuangg/CLEVRER)
- [4D-Bench](https://github.com/WenxuanZhu1103/4D-Bench)
- [MVVBench](https://huggingface.co/datasets/everex/MVVBench)

The additional numerical scorer interfaces fetch exact upstream files recorded
in [the audit manifest](reports/upstream_eval_audit_20261002/manifest.json):
Meta Project Aria (Apache-2.0), Google TAPNet (Apache-2.0), V-STaR, and the
Physion evaluator (MIT). Original notices remain in downloaded sources under
the ignored `external` directory. The code selects unchanged numerical
functions to avoid unrelated visualization or judge-model initialization.
The HOT3D/BOP and nuScenes launchers require separately installed upstream
repositories. No upstream dataset or model weights are distributed here.

The deterministic dynamic MCQ runner is new code. In particular, VLM4D's direct-choice
scoring is explicitly distinguished from the authors' free-text/judge protocol.

```bibtex
@misc{physbrainevalkit,
  title={PhysBrainEvalKit},
  author={DeepCybo Team},
  year={2026},
  url={https://github.com/DeepCybo-PhysAI/PhysBrainEvalKit}
}
```
