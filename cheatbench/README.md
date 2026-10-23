# CheatBench on a CPU VM, model on the GPU box

CheatBench (`centerforaisafety/cheatbench`) runs every episode in rootless enroot (user/pid/net
namespaces, squashfuse, fuse-overlayfs, slirp4netns). Vast instances are unprivileged Docker
containers that block all of that (checked 2026-10-23: `unshare --user` → "Operation not
permitted", no `/dev/fuse`, no Docker). So episodes run on a separate CPU VM, and the model stays
on the GPU box behind vLLM on `127.0.0.1`, reached through an SSH tunnel.

```
CPU VM (Hetzner)                                   GPU box (Vast)
 cheatbench user: run.py → enroot episodes          vLLM on 127.0.0.1:8000
   Terminus-2 → http://127.0.0.1:8000/v1
 rhtunnel user: autossh -L 8000 ───── ssh ─────▶   key: permitopen 127.0.0.1:8000, from=<VM IP>
 root: Claude Code, upload.env → HF                 watchdog ignores the VM's SSH, stops on vLLM idle
```

The laptop is needed only to set things up. The VM uploads results to the HF dataset repo itself
(`cheatbench/<run>/`, report in `cheatbench/box_report/`), the GPU box stops itself once vLLM has
been idle for its watchdog window, and the VM keeps billing until you delete it (Hetzner bills
stopped servers too).

## Steps

1. **Create the VM.** Hetzner Cloud, x86 dedicated vCPU (CCX33 for a smoke test, CCX43 = 16 vCPU /
   64 GB for collection; CheatBench sizes 1 CPU + 4 GB per episode), Ubuntu 24.04, public IPv4 on
   (GitHub has no IPv6), your SSH key, a firewall with inbound TCP 22 only, no backups, **empty
   cloud config**. On the laptop: `hetzner-set <ip>` (writes `~/.ssh/hetzner.conf`), `ssh hetzner true`.
2. **Copy the repo and the upload token over** (laptop):
   ```
   rsync -a --exclude .venv --exclude data --exclude .git --exclude .claude ./ hetzner:/root/llm-reward-hacking-ablation/
   ssh hetzner 'install -d -m 700 /root/.config/rhablation'
   scp ~/.config/rhablation/upload.env hetzner:/root/.config/rhablation/upload.env
   ssh hetzner 'chmod 600 /root/.config/rhablation/upload.env'
   ```
3. **Set up the VM** (values from `~/.ssh/gpubox.conf`):
   `ssh hetzner 'cd /root/llm-reward-hacking-ablation && BOX_HOST=<gpu box ip> BOX_PORT=<gpu box ssh port> bash cheatbench/setup_cpu_vm.sh'`.
   It ends by printing the exact command for step 4 (`FROM=<VM IP> bash cheatbench/allow_tunnel_key.sh '<key>'`).
4. **On the GPU box, as root, from the repo:** run that command. It authorises the key for the tunnel
   only, from the VM's address only, records the address for the watchdog and arms the watchdog
   (`WATCHDOG_MINUTES`, default 120). A watchdog that was already running predates the tunnel
   exception and would keep the box up: add `RESTART_WATCHDOG=1`.
   Make sure vLLM serves `Qwen/Qwen3.8-27B` (`MODEL=Qwen/Qwen3.8-27B bash scripts/serve_vllm.sh`).
5. **Smoke:** re-run step 3 with `--smoke` added. It checks the tunnel and runs one
   `prime_factorization` episode without the judge.
6. **Unattended:** `ssh -t hetzner 'cd /root/llm-reward-hacking-ablation && bash cheatbench/hetzner_claude.sh --mission cheatbench/mission_smoke.md'`,
   then `ssh -t hetzner tmux attach -t claude`, `/login`, accept the bypass notice, detach (Ctrl-b d).
   Rules for that session: `cheatbench/CLAUDE.unattended.md`. You can now close the laptop.
7. **Afterwards:** reports and runs are in the HF repo under `cheatbench/`. The agent stops
   `rhtunnel` at the end, the GPU box stops itself, and you delete the VM in the Hetzner console
   (`/logout` in Claude first if you will reuse a snapshot).

## What the setup does not cover yet

- **Token ids.** Terminus-2 talks to the OpenAI-compatible chat API, so the stored trajectory is
  messages, not the tokens Qwen saw. Activations need either a logging proxy in front of vLLM or
  vLLM's per-request token ids, plus a check of what the chat template keeps across turns.
- **Sampling.** Terminus-2/LiteLLM defaults apply (temperature etc.), not our rh-envs settings.
- **Judge.** Runs use `--no-judge`; judge off the VM (CheatBench's `judge.py`, or our DeepSeek judge).
- **Software Engineering task.** Needs Docker through Harbor and is not installed (adding the run
  user to the `docker` group would make it root-equivalent).
- **Auto-stop gap.** The GPU box treats "no vLLM request for `WATCHDOG_MINUTES`" as done. A pause
  that long inside a run (a sandbox build, a stalled agent) stops the box early; the mission builds
  images first for that reason.
- Vast may rewrite `/root/.ssh/authorized_keys` on restart, and the SSH port may change: re-run
  step 4, then set `BOX_PORT` in `/etc/rhtunnel/box.env` on the VM and `systemctl restart rhtunnel`.
