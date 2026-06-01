import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def safe_name(model):
    return model.replace("/", "__")


def snapshot(model, cache_root, patterns=None):
    target = Path(cache_root) / safe_name(model)
    kwargs = {
        "repo_id": model,
        "local_dir": target,
        "local_dir_use_symlinks": False,
    }
    if patterns:
        kwargs["allow_patterns"] = patterns
    return Path(snapshot_download(**kwargs))


def prepare(model, cache_root, code_model):
    model_dir = snapshot(model, cache_root)
    code_dir = snapshot(
        code_model, cache_root, ["modeling_instella.py", "configuration_instella.py"]
    )
    for filename in ["modeling_instella.py"]:
        target = model_dir / filename
        source = code_dir / filename
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    return model_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache_root", default="models/instella_local")
    parser.add_argument("--code_model", default="amd/Instella-3B-Instruct")
    return parser.parse_args()


def main():
    args = parse_args()
    print(prepare(args.model, args.cache_root, args.code_model))


if __name__ == "__main__":
    main()
