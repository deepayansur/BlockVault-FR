# Evaluation (VGGFace2)

Datasets are not included. Point the scripts at your local dataset paths under `datasets/`.

## What changed
- Verification metrics are computed from score-derived thresholds instead of a fixed grid.
- `TAR@FAR` uses the best valid operating point under the FAR cap.
- Offline embedding supports `--num-workers` and uses isolated worker engines.
- The generated report path is VGG-only for now.

## VGGFace2 verification
```bash
python eval/vggface2_verify.py --root datasets/VGGface2_HQ_1/VGGface2_None_norm_512_true_bygfpgan --max-images 1000 --num-workers 4 --cache eval/cache_vgg_det448_1000.npz --json-out eval/vggface2_stage1.json
```

## Full CPU report
This runs the detector-size sweep, local latency benchmark, HTTP `/verify` worker-topology benchmark, and writes `results.json` plus `results.txt`.

```bash
python eval/generate_results.py --max-images 1000 --num-workers 4
```

## Notes
- Stage-one tuning defaults are CPU-focused and start from a 1000-image subset.
- On restricted Windows environments the eval executor may fall back from process workers to isolated thread workers if multiprocessing pipes are blocked.
- `eval/qmul_verify_pairs.py` is still available for separate experiments, but it is not included in the generated metrics/report flow.
