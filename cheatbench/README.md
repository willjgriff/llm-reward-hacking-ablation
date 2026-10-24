# CheatBench on a CPU VM, model on the GPU box

CheatBench (`centerforaisafety/cheatbench`) runs every episode in rootless enroot (user/pid/net
namespaces, squashfuse, fuse-overlayfs, slirp4netns). Vast instances are unprivileged Docker
containers that block all of that (checked 2026-10-23: `unshare --user` → "Operation not
permitted", no `/dev/fuse`, no Docker). So episodes run on a separate CPU VM, and the model stays
on the GPU box behind vLLM on `127.0.0.1`, reached through an SSH tunnel.

```
CPU VM (Hetzner)                                   GPU box (Vast)
 cheatbench user: run.py → enroot episodes          vLLM on 127.0.0.1:8000, serving cb_base / cb_a3 / cb_rand
   Terminus-2 → http://127.0.0.1:8000/v1
 rhtunnel user: autossh -L 8000 ───── ssh ─────▶   key: permitopen 127.0.0.1:8000, from=<VM IP>
 root: Claude Code, upload.env → HF                 watchdog ignores the VM's SSH, stops on vLLM idle
 root: request-gpubox-row <row> ───── ssh ─────▶   row key: forced command serve_row.sh base|a3|rand|status
 root: request-gpubox-stop ────────── ssh ─────▶   done key: forced command, creates the stop request
```

The laptop is needed only to set things up. The VM uploads results to the HF dataset repo itself
(`cheatbench/<run>/`, report in `cheatbench/box_report/`), switches the row the GPU box serves
(`request-gpubox-row`: a third VM key whose only effect is `cheatbench/serve_row.sh` with one of four fixed
words), and at the end asks the GPU box to stop (`request-gpubox-stop`, the same kind of key; the box stops
as soon as nothing on it is busy). If the VM never asks, the box stops after its idle window anyway. The VM
keeps billing until you delete it (Hetzner bills stopped servers too).

Rows (`cheatbench/gpubox_setup.sh`, from `stage9/RECIPES.md` on HF): `cb_base` = Qwen3.8-27B @ 1d4bf0f;
`cb_a3` = stage 9's `CONF_A3_G_L30` (G9 steering, layers 30-63, alpha 3.61); `cb_rand` = random directions
(seed 1), same window, alpha calibrated to the same harmless KL on the box. All three are bias-only checkpoints
served through the stage-9 plugin, so they differ only in the bias values.

## Steps

1. **Rent the GPU box.** Vast, 1x H200 (an H100 works with less concurrency), 120 GB disk.
   On the laptop: `gpubox-llm-set <ip> <port>`, `ssh gpubox-llm true`.
2. **Create the VM.** Hetzner Cloud, x86 dedicated vCPU, Ubuntu 24.04: CCX53 (32 vCPU / 128 GB) for the
   H200 run (CheatBench sizes 1 CPU + 4 GB per episode; the mission caps concurrency at
   `min(nproc, RAM/4) - 2`), CCX33 for a smoke test only. Public IPv4 on (GitHub has no IPv6), your SSH key,
   a firewall with inbound TCP 22 only, no backups, **empty cloud config**. On the laptop: `hetzner-set <ip>`,
   `ssh hetzner true`.
3. **Copy the repo to both, and the upload token to the VM** (laptop):
   ```
   rsync -a --exclude .venv --exclude data --exclude .git --exclude .claude ./ gpubox-llm:/root/llm-reward-hacking-ablation/
   rsync -a --exclude .venv --exclude data --exclude .git --exclude .claude ./ hetzner:/root/llm-reward-hacking-ablation/
   ssh hetzner 'install -d -m 700 /root/.config/rhablation'
   scp ~/.config/rhablation/upload.env hetzner:/root/.config/rhablation/upload.env
   ssh hetzner 'chmod 600 /root/.config/rhablation/upload.env'
   ```
4. **Set up the GPU box** (~1 h: envs, 55 GB of weights, the random control's calibration, three checkpoints;
   ends serving `cb_base`):
   `ssh -t gpubox-llm 'cd /root/llm-reward-hacking-ablation && bash cheatbench/gpubox_setup.sh'`
   (in tmux on the box if your connection may drop). `--skip-rand` leaves the random row out.
5. **Set up the VM** (can run while step 4 runs; host and port from `~/.ssh/gpubox-llm.conf`):
   `ssh hetzner 'cd /root/llm-reward-hacking-ablation && BOX_HOST=<gpu box ip> BOX_PORT=<gpu box ssh port> bash cheatbench/setup_cpu_vm.sh'`.
   It ends by printing the exact command for step 6
   (`FROM=<VM IP> DONE_KEY='<key>' ROW_KEY='<key>' bash cheatbench/allow_tunnel_key.sh '<key>'`).
6. **On the GPU box, as root, from the repo:** run that command. It authorises the tunnel key for the port
   forward only and the done and row keys for their forced commands only, all from the VM's address only,
   records the address for the watchdog and arms the watchdog (`WATCHDOG_MINUTES`, default 90). A watchdog
   that was already running predates the tunnel exception and would keep the box up: add `RESTART_WATCHDOG=1`.
7. **Check:** re-run step 5 with `--smoke` added. It checks the tunnel and runs one `prime_factorization`
   episode on `cb-base` without the judge. `ssh hetzner request-gpubox-row status` should say `serving cb_base`.
8. **Unattended:** `ssh -t hetzner 'cd /root/llm-reward-hacking-ablation && bash cheatbench/hetzner_claude.sh --mission cheatbench/mission_a3.md'`,
   then `ssh -t hetzner tmux attach -t claude`, `/login`, accept the bypass notice, detach (Ctrl-b d).
   Rules for that session: `cheatbench/CLAUDE.unattended.md`. You can now close the laptop.
   - The pilot's provisional read and gate decision land in `cheatbench/box_report/` on HF about 6-8 h in.
   - Cancel the random row at any time: `ssh hetzner touch /root/cheatbench-skip-random`.
9. **Afterwards:** reports and runs are in the HF repo under `cheatbench/`. The agent stops `rhtunnel` and
   runs `request-gpubox-stop` at the end, the GPU box stops itself within about a minute of being idle, and you
   delete the VM in the Hetzner console (`/logout` in Claude first if you will reuse a snapshot) and destroy the
   GPU box on Vast (a stopped box still bills its disk).

## What the setup does not cover yet

- **Token ids.** Terminus-2 talks to the OpenAI-compatible chat API, so the stored trajectory is
  messages, not the tokens Qwen saw. Activations need either a logging proxy in front of vLLM or
  vLLM's per-request token ids, plus a check of what the chat template keeps across turns.
- **Judge.** Runs use `--no-judge`; judge off the VM (CheatBench's `judge.py`, or our DeepSeek judge).
- **Software Engineering task.** Needs Docker through Harbor and is not installed (adding the run
  user to the `docker` group would make it root-equivalent).
- **Auto-stop gap.** The GPU box treats "no vLLM request for `WATCHDOG_MINUTES`" as done. A pause
  that long inside a run (a sandbox build, a stalled agent) stops the box early; the mission builds
  images first for that reason.
- Vast may rewrite `/root/.ssh/authorized_keys` on restart, and the SSH port may change: re-run
  step 4, then set `BOX_PORT` in `/etc/rhtunnel/box.env` on the VM and `systemctl restart rhtunnel`.
