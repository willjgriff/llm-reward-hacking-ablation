"""The random-direction control row of cheatbench/mission_capability_kto.md: a hook file for ortho's scripts/vllm_proj.py in the format of
ortho's acts/kto_sweep/hooks/kto.safetensors, with a random unit direction in place of grader_kto's. Same layers (30-63), same generated-tokens-only
flag, and c[L] = mean_resid[L + 1] . u from grader_kto.safetensors, which is how kto.safetensors' own c is made (checked: equal to float32
precision), so the random direction is removed around the same mean ("a random unit direction at the grader pin", ortho kto_cfgs.rand).

    python cheatbench/make_rand_hook.py <kto.safetensors> <grader_kto.safetensors> <out.safetensors> [seed]
"""
import sys

import numpy as np
from safetensors.numpy import load_file, save_file

kto_path, vec_path, out = sys.argv[1:4]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
kto, vec = load_file(kto_path), load_file(vec_path)
mu = vec["mean_resid"].astype(np.float64)
layers = kto["layers"]
assert np.allclose(kto["c"][layers], mu[layers + 1] @ kto["u"].astype(np.float64), atol=1e-3), "kto's c is not mean_resid[L + 1] . u"
u = np.random.default_rng(seed).standard_normal(kto["u"].shape[0])
u /= np.linalg.norm(u)
c = np.zeros_like(kto["c"])
c[layers] = mu[layers + 1] @ u
save_file({"u": u.astype(np.float32), "c": c.astype(np.float32), "layers": layers, "gen_only": kto["gen_only"]}, out)
print(f"{out}: seed {seed}, layers {layers[0]}-{layers[-1]}, |u| {np.linalg.norm(u):.6f}, cos to kto u {float(u @ kto['u']):+.4f}, c[30,45,63] {c[[30, 45, 63]].round(3).tolist()}")
