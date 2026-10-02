# ADT / HOI4D evaluation split protocol

Updated: **2026-10-02**. Scope: `se-small-lite-wds518-8xh100-e100` and its
earlier ADT initialization data. Temporal evaluation adapters remain planned;
this document records the split policy for their implementation.

## Project decisions

| Dataset | Training history and evaluation setting | Basis |
|---|---|---|
| ADT | Earlier ADT training used exactly the same 64 sequences as the current WDS training inventory. Exclude these recordings when preparing a held-out evaluation. | Project-owner confirmation on 2026-10-02; the current WDS inventory was independently audited. |
| HOI4D | Proceed with the test-unseen setting: the test set is treated as unobserved during training. Retain the 522 known training sequences as exclusions when building evaluation manifests. | Project-owner instruction on 2026-10-02; this is the adopted evaluation assumption. |

These decisions allow ADT and HOI4D to remain first-priority temporal perception
additions. Describe scores as **in-domain held-out evaluation** and record the
chosen split and protocol with the results.

## Verified WDS inventory

The checkpoint's saved loader fingerprint matches its published resolved
configuration and manifest lock. The loader selects `train` shards and checks
each sample's split. Neither source has `val` or `test` shards in that lock.

| Source | Training sequences | Source frames | Train shards | Packed samples |
|---|---:|---:|---:|---:|
| ADT | 64 | 140,109 | 539 | 2,218 |
| HOI4D | 522 | 155,018 | 78 | 2,595 |

All sample keys were reconstructed from the original sequence IDs and frame
offsets. All 617 production receipts match their locked shard hashes, byte
sizes, split labels, and sample counts. Saved cursors for both sources occur in
all 32 loader workers. Packed samples describe the training corpus, not the
number of optimizer updates or individually confirmed gradient contributions.

Download the [training sequence IDs](../reports/wds_split_audit_20261002/training_sequence_ids.json)
and [audit summary with adopted protocol](../reports/wds_split_audit_20261002/summary.json).

## Preparing evaluation manifests

1. Select the intended official held-out split or publish the sequence list for
   a derived box/trajectory protocol.
2. Compare full recording IDs with the training inventory above. Keep all frames
   and synchronized views from a recording together; use the same ADT exclusions
   for the earlier initialization and WDS stages.
3. Save the final evaluation sequence IDs, dataset version, initialization rule,
   and input modalities alongside the scores. Record checkpoint-specific
   inventories separately when comparing Full or Old Small.

The HOI4D public-release comparison found 521 of 522 training sequences in the
[pinned release list](https://github.com/hoi4d/HOI4D_Script/blob/5d085f735aebdafc78c8df446a8754cfa934b17b/release.txt).
The remaining sequence, `ZY20210800001/H1/C20/N32/S292/s04/T3`, contributes
300 frames and five packed samples. This metadata discrepancy is retained for
traceability. Its absence from that public list did not establish membership
in a specific task's test split; the project adopts the test-unseen assumption
above rather than treating the discrepancy as a blocker.

The earlier ADT sequence equivalence is based on the project owner's training
history confirmation. The audit verified the WDS stage directly; it did not
independently reconstruct the earlier run or deduplicate all other data sources.

## Checkpoint provenance

- Repository: `Ju-Projects/Spatial-Models`.
- Revision: `140acdba93481834b8a8e6f790e00761f4e5fc96`.
- Checkpoint: `se-small-lite-wds518-8xh100-e100/checkpoint.pt`, epoch 100.
- [Resolved training configuration](https://huggingface.co/Ju-Projects/Spatial-Models/blob/140acdba93481834b8a8e6f790e00761f4e5fc96/se-small-lite-wds518-8xh100-e100/train_logs/config_resolved.yaml).
- [Frozen WDS loader](https://huggingface.co/Ju-Projects/Spatial-Models/blob/140acdba93481834b8a8e6f790e00761f4e5fc96/se-small-lite-wds518-8xh100-e100/code/sam3/train/data/hf_wds_dataset.py).
- [Manifest lock](https://huggingface.co/Ju-Projects/Spatial-Models/blob/140acdba93481834b8a8e6f790e00761f4e5fc96/se-small-lite-wds518-8xh100-e100/code/bolt/wds.lock.json).

The machine-readable audit summary preserves the checkpoint hash, loader
fingerprint, manifest hash, and the distinction between audited observations
and project-supplied decisions.
