import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file


def resolve_model(model, cache_dir):
    path = Path(model)
    if path.exists():
        return path
    cache_path = Path(cache_dir) / ("models--" + model.replace("/", "--")) / "snapshots"
    if cache_path.exists():
        snapshots = sorted([p for p in cache_path.iterdir() if (p / "config.json").exists()], key=lambda p: p.stat().st_mtime, reverse=True)
        if snapshots:
            return snapshots[0]
    try:
        return Path(snapshot_download(model, cache_dir=cache_dir, local_files_only=True))
    except Exception:
        return Path(snapshot_download(model, cache_dir=cache_dir))


def read_weight_map(model_dir):
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        return index["weight_map"], index.get("metadata", {}), True
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")
    weight_map = {}
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                weight_map[key] = file.name
    return weight_map, {}, False


def copy_model_files(base_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for source in base_dir.rglob("*"):
        if source.is_dir():
            continue
        rel = source.relative_to(base_dir)
        if ".git" in rel.parts:
            continue
        if source.suffix in {".safetensors", ".bin", ".pt", ".pth"}:
            continue
        if source.name == "model.safetensors.index.json":
            continue
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)


def tensor_from(model_dir, weight_map, key):
    file = model_dir / weight_map[key]
    with safe_open(file, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def key_seed(seed, key):
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def random_like(key, tensor, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(key_seed(seed, key))
    return torch.randn(tensor.shape, dtype=torch.float32, generator=generator)


def merge_tensor(key, base, sft, rl, seed, alpha):
    if not torch.is_floating_point(base):
        return base
    base32 = base.to(torch.float32)
    delta = rl.to(torch.float32) - sft.to(torch.float32)
    delta_norm = torch.linalg.vector_norm(delta)
    if delta_norm.item() == 0:
        return base
    noise = random_like(key, base, seed)
    noise_norm = torch.linalg.vector_norm(noise)
    if noise_norm.item() == 0:
        return base
    output = base32 + alpha * noise * (delta_norm / noise_norm)
    return output.to(base.dtype)


def build_random_model(base, sft, rl, out, cache_dir, seed, alpha):
    base_dir = resolve_model(base, cache_dir)
    sft_dir = resolve_model(sft, cache_dir)
    rl_dir = resolve_model(rl, cache_dir)
    out_dir = Path(out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    copy_model_files(base_dir, out_dir)
    base_map, metadata, has_index = read_weight_map(base_dir)
    sft_map, _, _ = read_weight_map(sft_dir)
    rl_map, _, _ = read_weight_map(rl_dir)
    missing = sorted((set(base_map) - set(sft_map)) | (set(base_map) - set(rl_map)))
    if missing:
        raise KeyError(f"Missing keys in SFT or RL model: {missing[:10]}")
    files = sorted(set(base_map.values()))
    total_size = 0
    for file_name in files:
        keys = [key for key, mapped in base_map.items() if mapped == file_name]
        tensors = {}
        for key in keys:
            base_tensor = tensor_from(base_dir, base_map, key)
            sft_tensor = tensor_from(sft_dir, sft_map, key)
            rl_tensor = tensor_from(rl_dir, rl_map, key)
            tensors[key] = merge_tensor(key, base_tensor, sft_tensor, rl_tensor, seed, alpha)
            total_size += tensors[key].numel() * tensors[key].element_size()
        save_file(tensors, out_dir / file_name)
        print(f"wrote {file_name}", flush=True)
    if has_index:
        metadata = dict(metadata)
        metadata["total_size"] = total_size
        metadata["random_vector_seed"] = str(seed)
        metadata["random_vector_alpha"] = str(alpha)
        metadata["random_vector_scale"] = "tensor_l2_delta"
        (out_dir / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": metadata, "weight_map": base_map}, indent=2) + "\n"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--sft", required=True)
    parser.add_argument("--rl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-dir", default="/workspace/hf_cache/hub")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    build_random_model(args.base, args.sft, args.rl, args.out, args.cache_dir, args.seed, args.alpha)


if __name__ == "__main__":
    main()
