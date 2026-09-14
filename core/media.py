"""Explicitly export native media for the encoder, without benchmark labels.

Lazy HF columns are read through Arrow to avoid loading torch/video decoders.
Frame indices remain indices: this function never invents physical timestamps.
"""
import io
from pathlib import Path


def export_media(value, directory, stem="input", kind="image"):
    from PIL import Image, UnidentifiedImageError
    root = Path(directory)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [export_media(item, root, f"{stem}_{i:04d}", kind) for i,item in enumerate(value)]
    if isinstance(value, dict):
        if value.get("__lazy_hf_images__") or value.get("__lazy_hf_video__"):
            raw = value["dataset"].data.column(value["column"])[value["row_index"]].as_py()
            return export_media(raw, root, stem, kind)
        if value.get("bytes") is not None:
            return export_media(value["bytes"], root, stem, kind)
        if value.get("path"):
            return {"path": str(value["path"]), "copied": False}
        raise ValueError(f"Unsupported media descriptor keys: {sorted(value)}")
    if isinstance(value, (str, Path)):
        return {"path": str(value), "copied": False}
    if isinstance(value, bytes):
        try:
            value = Image.open(io.BytesIO(value)).convert("RGB")
        except UnidentifiedImageError:
            # Preserve encoded video, without decoding or resampling it.
            suffix = ".mp4" if value[4:8] == b"ftyp" else ".webm" if value.startswith(b"\x1aE\xdf\xa3") else None
            if suffix is None:
                raise ValueError("Unsupported embedded media bytes; export using the dataset's native media loader")
            root.mkdir(parents=True, exist_ok=True)
            path = root / (stem + suffix)
            with path.open("xb") as stream:
                stream.write(value)
            return {"path": str(path.resolve()), "copied": True, "kind": "encoded_video"}
    if hasattr(value, "shape"):
        value = Image.fromarray(value)
    if isinstance(value, Image.Image):
        root.mkdir(parents=True, exist_ok=True)
        path = root / (stem + ".png")
        with path.open("xb") as stream:
            value.convert("RGB").save(stream, format="PNG")
        return {"path": str(path.resolve()), "copied": True, "size": list(value.size)}
    raise ValueError(f"Unsupported native media value: {type(value).__name__}")
