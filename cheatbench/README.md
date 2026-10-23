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
 rhtunnel user: autossh -L 8000 ───── ssh ─────▶   key restricted to permitopen 127.0.0.1:8000
```

## Steps

1. **Create the VM.** Hetzner Cloud, x86 dedicated vCPU (CCX33 for a smoke test, CCX43 = 16 vCPU /
   64 GB for collection; CheatBench sizes 1 CPU + 4 GB per episode), Ubuntu 24.04, your SSH key.
   No secrets in cloud-init user-data.
2. **Copy this directory over:** `rsync -a cheatbench/ root@<vm>:/root/rh-cheatbench/`
3. **On the VM, as root:**
   `BOX_HOST=<gpu box ip> BOX_PORT=<gpu box ssh port> bash /root/rh-cheatbench/setup_cpu_vm.sh`
   (values from `~/.ssh/gpubox.conf`). It prints the tunnel's public key at the end.
4. **On the GPU box, as root** (rsync `cheatbench/allow_tunnel_key.sh` over first):
   `bash cheatbench/allow_tunnel_key.sh '<that public key>'`.
   Make sure vLLM serves `Qwen/Qwen3.8-27B` (`MODEL=Qwen/Qwen3.8-27B bash scripts/serve_vllm.sh`).
5. **Back on the VM:** re-run step 3 with `--smoke` added. It checks the tunnel and runs one
   `prime_factorization` episode without the judge.

## What the setup does not cover yet

- **Token ids.** Terminus-2 talks to the OpenAI-compatible chat API, so the stored trajectory is
  messages, not the tokens Qwen saw. Activations need either a logging proxy in front of vLLM or
  vLLM's per-request token ids, plus a check of what the chat template keeps across turns.
- **Sampling.** Terminus-2/LiteLLM defaults apply (temperature etc.), not our rh-envs settings.
- **Judge.** Runs use `--no-judge`; judge off the VM (CheatBench's `judge.py`, or our DeepSeek judge).
- **Software Engineering task.** Needs Docker through Harbor and is not installed (adding the run
  user to the `docker` group would make it root-equivalent).
- Vast may rewrite `/root/.ssh/authorized_keys` on restart: re-run `allow_tunnel_key.sh`.
