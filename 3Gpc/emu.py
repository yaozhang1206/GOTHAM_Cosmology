"""Transformer halo emulator: fields -> token sentences -> halo catalog.

Generalisation of ../model/generate_catalog_fast.py (same checkpoint, sampling, chunking and
decoding rules) to any box made of grid^3 subvolumes of 15.625 Mpc/h and to a slab of subvolume rows
(x offset), with a vectorised decoder.  Catalog columns: x y z [Mpc/h], M [Msun/h], vx vy vz [km/s], c.

    python emu.py --fields DIR/{id}/dmo_fields_subvols_grid_8_LH_{id}.npy --cid 1335 --out cat.npy [--seed-base S]

Env: conda nn_test (torch 2.5.1), module cudatoolkit/25.5_12.9.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))  # HaloDecoderModel
CKPT = ("/work/hdd/bdne/yzhang116/checkpoints_quijote/1872236/checkpoint_cos_cross_entropy_epoch_0_step_7050_"
        "embed_384_batch_800_lrmax_1e-05_lrmin_1e-05_layer_8_head_12_layervit_4_headvit_8_patch_4_dropout_0.0.pt")
COSMO_FILE = "/work/hdd/bdne/yzhang116/quijote_galaxy/LH_points_HOD_cosmo_ext_40000.txt"

NVOCAB = 131
START, SPACE, PAD, END = NVOCAB + 1, NVOCAB + 2, NVOCAB + 3, NVOCAB + 4
TARGET_LEN, MAX_NEW = 296, 289
SUB_L = 1000.0 / 64                    # subvolume side, Mpc/h
POS_VOCAB, DIM_TOT, DIM_PROP = 40, 8, 5

BINS = np.zeros((DIM_PROP, NVOCAB + 1))
BINS[0, :-1] = np.linspace(12.7, 15.0, NVOCAB); BINS[0, -1] = 15.0
for _i in range(1, 4):
    BINS[_i, :-1] = np.linspace(-1250, 1250, NVOCAB); BINS[_i, -1] = 1250.0
BINS[4, :-1] = np.linspace(1.0, 16.0, NVOCAB); BINS[4, -1] = 16.0
STEP = np.array([2.3, 2500.0, 2500.0, 2500.0, 15.0]) / (NVOCAB - 1)
XARR = np.concatenate((np.arange(POS_VOCAB) * (SUB_L / POS_VOCAB), [SUB_L]))
DXPOS = SUB_L / POS_VOCAB


def cosmologies():
    a = np.loadtxt(COSMO_FILE)
    ids, first = np.unique(a[:, 15].astype(int), return_index=True)
    return a[first, 10:15]


class Emulator:
    def __init__(self, ckpt=CKPT, compile_blocks=True):
        import torch
        from model_enc_dec_cos_fast import HaloDecoderModel
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.preferred_blas_library("cublaslt")
        self.torch = torch
        self.dev = torch.device("cuda")
        ck = torch.load(ckpt, map_location=self.dev)
        sd = {k.replace("module.", ""): v for k, v in ck["model"].items()}
        self.model = HaloDecoderModel(ck["config"]).to(self.dev)
        self.model.load_state_dict(sd)
        self.model.eval()
        if compile_blocks:
            for i in range(len(self.model.transformer.h)):
                self.model.transformer.h[i] = torch.compile(self.model.transformer.h[i], mode="default")

    def generate(self, fields, params, seed_base, chunk=16384, log=None):
        """fields: (n, 8, 8, 8, C) float16 numpy (or memmap); params (5,). -> (n, TARGET_LEN) int16 tokens.
        Chunk c is sampled with torch seed seed_base + c (generate_catalog_fast: simid * 16 + c)."""
        torch = self.torch
        n = fields.shape[0]
        out = np.empty((n, TARGET_LEN), np.int16)
        ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        p = torch.tensor(np.repeat(np.asarray(params, np.float64)[None], min(chunk, n), 0), device=self.dev).bfloat16()
        with torch.no_grad(), ctx:
            for c, s in enumerate(range(0, n, chunk)):
                e = min(s + chunk, n)
                x = torch.from_numpy(np.ascontiguousarray(fields[s:e])).to(self.dev, non_blocking=True)
                x = torch.moveaxis(x.bfloat16(), -1, 1)
                torch.manual_seed(seed_base + c)
                torch.cuda.manual_seed_all(seed_base + c)
                y = self.model.generate(x, params=p[: e - s], max_new_tokens=MAX_NEW, start_token=START,
                                        end_token=END, pad_token=PAD, temperature=1.0)
                L = y.shape[1]
                if L < TARGET_LEN:
                    y = torch.cat([y, torch.full((e - s, TARGET_LEN - L), PAD, dtype=y.dtype, device=y.device)], 1)
                out[s:e] = y[:, :TARGET_LEN].to(torch.int16).cpu().numpy()
                if log:
                    log(f"chunk {c}: {s}-{e}")
                del x, y
        return out


def decode(tokens, grid, box, x0=0, rng=None):
    """Tokens (n_rows, TARGET_LEN) of subvolume rows r = (jx - x0) * grid^2 + jy * grid + jz -> catalog (N, 8).
    Same rules as generate_catalog_fast.get_prop_pos: rows without end token or with a non-integer halo count
    are dropped; a halo with any out-of-range token is dropped; uniform jitter of one bin."""
    rng = np.random.default_rng() if rng is None else rng
    t = np.delete(tokens.astype(np.int64), [1, 2, 3, 4, 5, 6], axis=1)
    n, L = t.shape
    is_s, is_e = t == START, t == END
    has_s, has_e = is_s.any(1), is_e.any(1)
    i_s, i_e = is_s.argmax(1), is_e.argmax(1)
    span = i_e - i_s - 1
    ok = has_s & has_e & (span % DIM_TOT == 0) & (span > 0)
    nh = np.where(ok, span // DIM_TOT, 0)
    rows = np.repeat(np.arange(n), nh)
    jh = np.arange(nh.sum()) - np.repeat(np.cumsum(nh) - nh, nh)
    base = i_s[rows] + jh * DIM_TOT
    tok = t[rows[:, None], base[:, None] + np.arange(1, DIM_TOT + 1)[None, :]]      # (N, 8)
    good = (tok[:, :3] <= POS_VOCAB).all(1) & (tok[:, 3:] <= NVOCAB).all(1) & (tok >= 0).all(1)
    tok, rows = tok[good], rows[good]
    N = len(rows)
    prop = BINS[np.arange(DIM_PROP)[None, :], tok[:, 3:]] + rng.uniform(-0.5, 0.5, (N, DIM_PROP)) * STEP
    prop = np.clip(prop, BINS[:, 0], BINS[:, -1])
    jx = rows // grid ** 2 + x0
    jy = (rows // grid) % grid
    jz = rows % grid
    sub = box / grid
    pos = np.stack([XARR[tok[:, 0]] + sub * jx, XARR[tok[:, 1]] + sub * jy, XARR[tok[:, 2]] + sub * jz], 1)
    pos = np.mod(pos + rng.uniform(-0.5, 0.5, (N, 3)) * DXPOS, box)
    prop[:, 0] = 10.0 ** prop[:, 0]
    return np.concatenate([pos, prop], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True)
    ap.add_argument("--cid", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-base", type=int, default=None, help="default cid * 16 (generate_catalog_fast)")
    ap.add_argument("--box", type=float, default=1000.0)
    ap.add_argument("--save-tokens", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    emu = Emulator()
    f = np.load(a.fields, mmap_mode="r")
    grid = round(f.shape[0] ** (1 / 3))
    assert grid ** 3 == f.shape[0] and abs(a.box / grid - SUB_L) < 1e-9
    par = cosmologies()[a.cid]
    seed = a.cid * 16 if a.seed_base is None else a.seed_base
    tok = emu.generate(f, par, seed, log=lambda m: print(m, f"{time.time()-t0:.0f}s", flush=True))
    t1 = time.time()
    cat = decode(tok, grid, a.box, rng=np.random.default_rng(seed))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.save(a.out, cat)
    if a.save_tokens:
        np.save(a.out.replace(".npy", "_tokens.npy"), tok)
    print(f"CID {a.cid} seed {seed}: {len(cat)} halos, generate {t1-t0:.0f} s, decode {time.time()-t1:.1f} s -> {a.out}",
          flush=True)


if __name__ == "__main__":
    main()
