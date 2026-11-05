# Mission (DRAFT): GPU-box prep for the Terminal-Bench persistence rows (x1, r1, x0)

Status: draft, 2026-11-05. A Claude session on the GPU box that will serve `cheatbench/mission_persistence_tb.md`. It runs **before** the VM's mission, on a box that `cheatbench/gpubox_setup.sh --skip-rand` has already set up (envs, base weights, residual-bias plugin, `cb_base` serving, `/root/.config/rhablation/cheatbench_serve.env`). Rules: `sweep/CLAUDE.unattended.md` (installed by `box_claude.sh`).

## Goal

Make three weight-edited rows servable through the VM's row key, each verified against its earlier measurement, then leave `cb_x1` serving.

| row | served name | recipe | verify against |
|---|---|---|---|
| x1 | `cb_x1` | stage-7 X1: D_TV `tamper_vs_clean__cot_last` (HF `stage4/stage7/s7-tv/directions/`, sha256 `081b3d3a…` in `stage7/stage7/rows/X1/adapter/manifest.json`), stack of 24 heretic edits, w 1.0, L40–63 | `stage7/stage7/rows/X1/gate/gate.json` (and `first_token_kl_vs_ref.json`) |
| r1 | `cb_r1` | stage-5b R1: `attempt_vs_A__completion_last` from HF `stage4/tv27b-full`, the same stack recipe | the stage-5c likelihood scores, to ≤ 0.001 nats, as stage 5d verified its rebuild (HF `stage5d/`, `sweep/box_report/mbpp_directions/`) |
| x0 | `cb_x0` | stage-7 X0: random direction, the seed in `stage7/stage7/rows/X0/` (recipe, manifest), the same stack recipe | `stage7/stage7/rows/X0/gate/gate.json` |

Rebuild every edit from its recorded recipe. For stacks the stored adapters hold only the last round, so do not use them for serving. The stack code is the stage-7 matrix build in the HF code mirror (`sweep/code/scripts/stage7_run_matrix.py` and what it calls; the stage-5 scripts at mirror revision `00f85914` if a piece is missing at the head). Each merged checkpoint gets the MTP tensors (`add_mtp_weights.py`), as stage 7 did, so it is served with the same speculative decoding as the bias rows.

## Steps

1. **Read:**
   - `cheatbench/gpubox_setup.sh`, `cheatbench/serve_row.sh`, `cheatbench/allow_tunnel_key.sh`;
   - `cheatbench/mission_persistence_tb.md` (who calls you and how);
   - the stage-7 build log of X1 (`stage7/stage7/rows/X1/build.log`).
2. **Build and verify each row**, one at a time (disk: base plus one ~55 GB merged checkpoint; keep `df -h /` above 30 GB):
   1. stop vLLM;
   2. build the merged checkpoint;
   3. run the stored gate's scorer and compare with the stored numbers;
   4. record the result in `LOG.md`.
   A row that does not reproduce is **not** made servable. Record why, and do not tune anything to make it match.
3. **Rebuild check.** `serve_row.sh` must rebuild a row it has deleted, and the rebuild must be the verified checkpoint:
   - record a fingerprint per verified row: shard sha256 if two builds come out bitwise identical (test it once on x1); otherwise first-token logprobs on 20 fixed prompts through vLLM, tolerance 0.005 nats;
   - the rebuild path checks that fingerprint and refuses to serve on a mismatch.
4. **Extend `cheatbench/serve_row.sh`** (and the forced-command row key's allowed words) with `x1|r1|x0`:
   - Merged rows are served without the residual-bias plugin, with every other setting from `cheatbench_serve.env` unchanged: 131072 context, MTP, `MAX_NUM_SEQS`, and **vision on**, as in the capability run.
   - Switching to a merged row deletes any other merged row first, rebuilds if missing, checks the fingerprint, then serves.
   - Keep the existing rows and the in-flight-request guard working. The VM will call `request-gpubox-row x1|r1|x0|status`.
   - Test every row word once by hand: switch, `/v1/models`, one chat request, one image request (the capability run's vision check), `status`.
5. **End state:**
   - `cb_x1` serving;
   - `/root/persistence_prep_done` containing one line per row (`verified` or `not servable: <reason>`);
   - `box_report/persistence_gpubox/{LOG.md,SUMMARY.md}` uploaded to HF under `cheatbench/persistence/` (`upload_run.sh`);
   - the code via `sweep/upload_code.sh --message "persistence gpubox prep"`.
   Then stop working. **Do not stop the box**: the VM's mission needs it, and the watchdog armed by `allow_tunnel_key.sh` stops it when the VM is done.

Cap 3 h. If x1 cannot be verified within the cap, leave `cb_base` serving, write that in `/root/persistence_prep_done` and in the summary, and stop.

## Not in scope

No Terminal-Bench, lm-eval or rollouts on this box. No changes to the bias rows or their recipes.
