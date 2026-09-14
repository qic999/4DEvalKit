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

Dynamic annotation sources:

- [STI-Bench](https://mint-sjtu.github.io/STI-Bench.io/), [dataset](https://huggingface.co/datasets/MINT-SJTU/STI-Bench)
- [VLM4D](https://vlm4d.github.io/), [dataset](https://huggingface.co/datasets/shijiezhou/VLM4D)
- [DSI-Bench](https://dsibench.github.io/), [dataset](https://huggingface.co/datasets/Viglong/DSI-Bench)

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
