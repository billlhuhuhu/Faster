import csv
import json
from pathlib import Path


def _read_json_records(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ["data", "annotations", "records", "examples"]:
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"Unsupported AudioCaps json format: {path}")


def _read_jsonl_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _read_csv_records(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def find_annotation_file(data_root, split):
    data_root = Path(data_root)
    split = str(split).lower()
    candidates = []
    for suffix in ("csv", "json", "jsonl"):
        candidates.extend(data_root.glob(f"*{split}*.{suffix}"))
        candidates.extend((data_root / "annotations").glob(f"*{split}*.{suffix}") if (data_root / "annotations").exists() else [])
        candidates.extend((data_root / "metadata").glob(f"*{split}*.{suffix}") if (data_root / "metadata").exists() else [])
    if not candidates:
        raise FileNotFoundError(f"No AudioCaps annotation file for split={split} under {data_root}")
    return sorted(candidates, key=lambda item: (len(str(item)), str(item)))[0]


def read_annotation_records(annotation_path):
    path = Path(annotation_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv_records(path)
    if suffix == ".json":
        return _read_json_records(path)
    if suffix == ".jsonl":
        return _read_jsonl_records(path)
    raise ValueError(f"Unsupported AudioCaps annotation suffix: {path}")


def _first_present(record, keys):
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _normalize_caption(record):
    caption = _first_present(record, ["caption", "text", "sentence", "description", "audio_caption"])
    if caption is None:
        raise KeyError(f"AudioCaps record has no caption/text field. Keys: {sorted(record.keys())}")
    return str(caption)


def build_audio_index(data_root):
    data_root = Path(data_root)
    audio_exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    index = {}
    for path in data_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in audio_exts:
            continue
        keys = {path.name, path.stem}
        stem = path.stem
        if stem.startswith("Y"):
            keys.add(stem[1:])
        for key in keys:
            index.setdefault(str(key), path)
    return index


def resolve_audio_path(record, data_root, audio_index):
    data_root = Path(data_root)
    direct = _first_present(record, ["audio", "audio_path", "path", "file", "file_name", "filename", "wav", "wav_path"])
    if direct is not None:
        direct_path = Path(str(direct))
        candidates = [direct_path] if direct_path.is_absolute() else [data_root / direct_path]
        candidates.append(data_root / direct_path.name)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        if direct_path.name in audio_index:
            return audio_index[direct_path.name]
        if direct_path.stem in audio_index:
            return audio_index[direct_path.stem]

    youtube_id = _first_present(record, ["youtube_id", "youtubeid", "ytid", "video_id", "video"])
    start_time = _first_present(record, ["start_time", "start", "start_seconds", "start_sec"])
    candidate_keys = []
    if youtube_id is not None:
        yid = str(youtube_id)
        candidate_keys.extend([yid, f"Y{yid}"])
        if start_time is not None:
            start_raw = str(start_time)
            candidate_keys.extend([f"{yid}_{start_raw}", f"Y{yid}_{start_raw}"])
            try:
                start_int = int(float(start_raw))
                candidate_keys.extend([f"{yid}_{start_int}", f"Y{yid}_{start_int}", f"{yid}_{start_int:06d}"])
            except ValueError:
                pass
    audiocap_id = _first_present(record, ["audiocap_id", "id", "sample_id"])
    if audiocap_id is not None:
        candidate_keys.append(str(audiocap_id))

    for key in candidate_keys:
        if key in audio_index:
            return audio_index[key]
        for suffix in [".wav", ".flac", ".mp3", ".m4a", ".ogg"]:
            path = data_root / f"{key}{suffix}"
            if path.exists():
                return path
    raise FileNotFoundError(f"Could not resolve AudioCaps audio path for record keys={sorted(record.keys())}")


def load_audiocaps_records(data_root, split="train", annotation_path=None, max_samples=None):
    data_root = Path(data_root)
    annotation_path = Path(annotation_path) if annotation_path else find_annotation_file(data_root, split)
    raw_records = read_annotation_records(annotation_path)
    audio_index = build_audio_index(data_root)
    records = []
    for sample_idx, record in enumerate(raw_records):
        audio_path = resolve_audio_path(record, data_root, audio_index)
        caption = _normalize_caption(record)
        audio_id = _first_present(record, ["youtube_id", "youtubeid", "ytid", "video_id", "audio_id", "audiocap_id", "id"])
        if audio_id is None:
            audio_id = audio_path.stem
        records.append(
            {
                "sample_idx": int(sample_idx),
                "dataset": "audiocaps",
                "split": str(split),
                "caption": caption,
                "audio": str(audio_path),
                "audio_id": str(audio_id),
                "raw_record": record,
            }
        )
        if max_samples is not None and len(records) >= int(max_samples):
            break
    return records, str(annotation_path)
