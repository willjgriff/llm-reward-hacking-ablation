# Mission update 2 for `sweep/mission_mbpp_impossible.md` (user, 2026-10-01 16:45 UTC)

Everything else stays as it is. This changes only the very last step: **stop the box yourself as soon as
the final results are uploaded**, instead of leaving it to the 2-hour watchdog.

## Final sequence (replace "Then stop." at the end of the mission)
1. Finish as planned: `SUMMARY.md` final, last run dirs uploaded, `box_report` uploaded with
   `--dest-prefix sweep`, code uploaded with `bash sweep/upload_code.sh --message "mbpp_impossible final"`.
2. **Verify the uploads** against the HF repo (list the files under `sweep/box_report/mbpp_impossible/`,
   `sweep/Qwen3.8-27B/mbpp_impossible/`, `sweep/Qwen3.8-27B/impossiblebench_single_step_*/` and
   `sweep/code/`; compare names and sizes with the local files). Write the result as the last `LOG.md` line
   and re-upload `box_report` if that line changed anything. If any upload failed, do **not** stop the box:
   put the failure at the top of `SUMMARY.md`, upload what you can, and leave the watchdog to it.
3. Log out of Claude Code's stored login the way `/logout` would, so the token does not stay on the stopped
   disk: remove the credentials file under `/root/.claude/` (`.credentials.json` if that is where this
   version keeps it; check with `ls -la /root/.claude`). Do this only after step 2 succeeded and nothing
   else remains; you will not be able to call the model afterwards, so make it the second-to-last command.
4. **Stop the instance** (same mechanism as `scripts/stop_box_when_idle.sh`, as root):
   ```
   tmux kill-session -t watchdog 2>/dev/null
   CONTAINER_ID=$(grep -E '^(export )?CONTAINER_ID=' /etc/environment | tail -1 | sed -E 's/^(export )?CONTAINER_ID=//; s/^["'"'"']//; s/["'"'"']$//')
   export VAST_API_KEY=$(grep -E '^(export )?CONTAINER_API_KEY=' /etc/environment | tail -1 | sed -E 's/^(export )?CONTAINER_API_KEY=//; s/^["'"'"']//; s/["'"'"']$//')
   /opt/instance-tools/bin/vastai stop instance "$CONTAINER_ID"
   ```
   (`command -v vastai` first if that path is missing.) The key goes through the environment, never on a
   command line, and never into any file. "Stop" keeps the disk; it is not "destroy".
