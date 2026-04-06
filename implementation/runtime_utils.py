import os

import torch
from sentence_transformers import SentenceTransformer


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_CPU_THREADS_CONFIGURED = False


def resolve_dataset_path(dataset_arg, default_name):
    if dataset_arg:
        if os.path.isabs(dataset_arg):
            return dataset_arg
        candidate = os.path.join(DATA_DIR, dataset_arg)
        if os.path.exists(candidate):
            return candidate
        return os.path.abspath(dataset_arg)
    return os.path.join(DATA_DIR, default_name)


def resolve_device(device_arg="auto"):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested device 'cuda' but CUDA is not available in this session.")
    if device_arg == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("Requested device 'mps' but MPS is not available in this session.")
    return device_arg


def configure_cpu_runtime(threads=1):
    global _CPU_THREADS_CONFIGURED
    if _CPU_THREADS_CONFIGURED:
        return
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass
    _CPU_THREADS_CONFIGURED = True


def load_sentence_transformer(model_name, device=None, offline=False):
    kwargs = {}
    if device:
        kwargs["device"] = device
    cache_folder = os.environ.get("SENTENCE_TRANSFORMERS_HOME")
    if cache_folder:
        kwargs["cache_folder"] = cache_folder
    if offline:
        kwargs["local_files_only"] = True
    return SentenceTransformer(model_name, **kwargs)
