# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-stage annotation preview: filtered counts/ratios from annotation parquet (keep_mask); sample rows from tars."""

from __future__ import annotations

import argparse
import base64
import glob
import html
import io
import re
import http.server
import json
import os
import socketserver
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

try:
    from PIL import Image  # noqa: E402
except ImportError:
    Image = None

from preview_annotation_server import (  # noqa: E402
    _content_mask,
    _load_annotation_all_and_kept_sets,
    _materialize_rows_parallel,
    _safe_str,
    _to_bytes,
)
from preview_annotation_standalone import (  # noqa: E402
    DEFAULT_JSON_EXTENSIONS,
    DEFAULT_WEBDATASET_EXTENSIONS,
    FileGroupTask,
    WebdatasetReaderStage,
    get_all_file_paths_under,
)

PER_STAGE_SAMPLES = 5
DEFAULT_WORKERS = 16
DISPLAY_IMAGE_HEIGHT_PX = 400
THUMBNAIL_JPEG_QUALITY = 100

_TRAILING_INT_RE = re.compile(r"(\d+)$")


def _format_filtered_ratio(n_filtered: int, n_total: int) -> str:
    if n_total <= 0:
        return "—"
    return f"{100.0 * n_filtered / n_total:.1f}%"


def _resolve_tar_paths(input_path: str | list[str], read_kwargs: dict, max_tars: int | None) -> list[str]:
    read_kwargs = read_kwargs or {}
    if isinstance(input_path, str):
        paths = get_all_file_paths_under(
            input_path,
            recurse_subdirectories=True,
            keep_extensions=list(DEFAULT_WEBDATASET_EXTENSIONS),
            storage_options=read_kwargs.get("storage_options"),
        )
    else:
        paths = list(input_path)
    if max_tars is not None and max_tars > 0:
        paths = paths[:max_tars]
    return paths


def _stage_newly_filtered_plan(
    all_keys_per_stage: list[set[tuple[str, int]]],
    kept_sets: list[set[tuple[str, int]]],
    annotation_dirs: list[str],
    per_stage: int,
) -> list[tuple[str, int, list[tuple[str, int]], int]]:
    """Per stage: (annotation_dir, n_total_filtered, sample_keys_for_tars, total keys in parquet).

    ``n_total_filtered`` is |all_keys − kept| for that stage. ``sample_keys_for_tars`` are up to
    ``per_stage`` keys that are newly filtered vs the previous stage (for thumbnails only).
    """
    out: list[tuple[str, int, list[tuple[str, int]], int]] = []
    for i, ann_dir in enumerate(annotation_dirs):
        all_k = all_keys_per_stage[i]
        kept_i = kept_sets[i]
        n_total_filtered = len(all_k - kept_i)
        if i == 0:
            new_keys = all_k - kept_i
        else:
            new_keys = (kept_sets[i - 1] - kept_i) & all_k
        picked = sorted(new_keys)[:per_stage]
        out.append((ann_dir, n_total_filtered, picked, len(all_k)))
    return out


def _read_one_tar_rows_for_keys(
    index: int,
    tar_path: str,
    read_kwargs: dict,
    need: frozenset[tuple[str, int]],
) -> tuple[int, dict[tuple[str, int], pd.Series]]:
    reader = WebdatasetReaderStage(
        source_id_field="pdf_name",
        read_kwargs=read_kwargs,
        materialize_on_read=False,
        max_batch_bytes=None,
        json_extensions=tuple(DEFAULT_JSON_EXTENSIONS),
    )
    task = FileGroupTask(
        task_id=f"fetch_{index}",
        dataset_name="multi_preview",
        data=[tar_path],
        _metadata={"source_files": [tar_path]},
    )
    out = reader.process(task)
    batches = out if isinstance(out, list) else [out]
    found: dict[tuple[str, int], pd.Series] = {}
    for batch in batches:
        df = batch.to_pandas()
        if df.empty:
            continue
        content = _content_mask(df)
        sub = df.loc[content]
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            k = (str(row["sample_id"]), int(row["position"]))
            if k in need and k not in found:
                found[k] = row
    return index, found


def _fetch_rows_for_keys(
    paths: list[str],
    read_kwargs: dict,
    need_keys: set[tuple[str, int]],
    num_workers: int,
) -> dict[tuple[str, int], pd.Series]:
    if not need_keys or not paths:
        return {}
    need = frozenset(need_keys)
    if num_workers <= 1:
        merged: dict[tuple[str, int], pd.Series] = {}
        for i, tar_path in enumerate(paths):
            if len(merged) >= len(need_keys):
                break
            _, part = _read_one_tar_rows_for_keys(i, tar_path, read_kwargs, need)
            for k, row in part.items():
                if k not in merged:
                    merged[k] = row
        return merged

    workers = min(num_workers, len(paths))
    per_index: dict[int, dict[tuple[str, int], pd.Series]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_read_one_tar_rows_for_keys, i, p, read_kwargs, need): i for i, p in enumerate(paths)
        }
        for fut in as_completed(futures):
            idx, part = fut.result()
            per_index[idx] = part
    merged = {}
    for i in range(len(paths)):
        part = per_index.get(i, {})
        for k, row in part.items():
            if k not in merged:
                merged[k] = row
    return merged


def _rows_dict_to_dataframe(keys_order: list[tuple[str, int]], by_key: dict[tuple[str, int], pd.Series]) -> pd.DataFrame:
    rows = [by_key[k] for k in keys_order if k in by_key]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


def _scale_image_fixed_height(b: bytes, height_px: int) -> tuple[bytes, str] | None:
    if Image is None:
        return None
    try:
        with Image.open(io.BytesIO(b)) as img:
            img = img.convert("RGB")
            w, h = img.size
            if h <= 0:
                return None
            new_w = max(1, int(round(w * height_px / h)))
            resample = getattr(Image, "Resampling", Image).LANCZOS
            if (w, h) != (new_w, height_px):
                img = img.resize((new_w, height_px), resample)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=THUMBNAIL_JPEG_QUALITY, optimize=True)
            return buf.getvalue(), "image/jpeg"
    except (OSError, ValueError, TypeError):
        return None


def _row_to_html(row: pd.Series, index: int, _kind: str) -> str:
    sid = html.escape(_safe_str(row.get("sample_id")))
    pos = row.get("position", "")
    mod = html.escape(_safe_str(row.get("modality")))
    text_val = row.get("text_content")
    text = html.escape(_safe_str(text_val))

    parts = [f"<div class='row'><span class='meta'>#{index} sample_id={sid} position={pos} modality={mod}</span>"]
    if _safe_str(row.get("modality")) == "image":
        raw = row.get("binary_content")
        b = _to_bytes(raw)
        if b and len(b) > 0:
            scaled = _scale_image_fixed_height(b, DISPLAY_IMAGE_HEIGHT_PX)
            if scaled is not None:
                thumb_b, ct = scaled
                b64 = base64.b64encode(thumb_b).decode("ascii")
                parts.append(f"<img src=\"data:{ct};base64,{b64}\" alt=\"image\" />")
            else:
                parts.append("<span class='noimg'>[image not loaded or invalid]</span>")
        else:
            parts.append("<span class='noimg'>[image not loaded]</span>")
    else:
        parts.append(f"<pre class='text'>{text}</pre>")
    parts.append("</div>")
    return "\n".join(parts)


def _shared_styles() -> str:
    return """
    body { font-family: system-ui, sans-serif; margin: 1rem 2rem; }
    h1 { font-size: 1.2rem; }
    h2 { font-size: 1rem; margin-top: 1.5rem; color: #06c; }
    h3 { font-size: 0.95rem; margin-top: 1rem; }
    section.sample { margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 1px solid #ccc; }
    .row { margin: 0.5rem 0; padding: 0.5rem; border: 1px solid #ddd; border-radius: 4px; }
    .meta { color: #666; font-size: 0.85rem; }
    .row img { display: block; margin-top: 0.25rem; }
    .text { margin: 0.25rem 0; white-space: pre-wrap; word-break: break-word; font-size: 0.9rem; }
    .noimg { color: #999; }
    .empty { color: #999; font-style: italic; }
    .summary-table { border-collapse: collapse; margin-top: 1rem; }
    .summary-table th, .summary-table td { padding: 0.4rem 0.8rem; text-align: left; border: 1px solid #ddd; }
    .summary-table a { color: #06c; }
    .img-stats { margin: 0.5rem 0 1rem; padding: 0.5rem 0.75rem; background: #f5f5f5; border-radius: 4px; font-size: 0.95rem; }
    .img-stats strong { color: #333; }
"""


def _annotation_dir_sort_key(path: str) -> tuple:
    base = os.path.basename(path.rstrip(os.sep))
    m = _TRAILING_INT_RE.search(base)
    if m:
        return (0, int(m.group(1)), path.lower())
    return (1, base.lower(), path)


def _expand_annotation_dirs(pattern: str) -> list[str]:
    matches = glob.glob(pattern)
    dirs = [os.path.abspath(m) for m in matches if os.path.isdir(m)]
    dirs.sort(key=_annotation_dir_sort_key)
    if not matches:
        msg = f"No paths matched annotation glob: {pattern!r}"
        raise SystemExit(msg)
    if not dirs:
        msg = f"Annotation glob matched {len(matches)} path(s) but none are directories: {pattern!r}"
        raise SystemExit(msg)
    return dirs


def _build_html(
    stage_rows: list[tuple[str, pd.DataFrame, int, int]],
    annotation_pattern: str,
    per_stage: int,
) -> str:
    style = _shared_styles()
    extra = """
    .stage-block { margin-bottom: 2rem; padding: 1rem; border: 1px solid #ccc; border-radius: 6px; }
    .stage-title { font-size: 1.05rem; color: #06c; margin-bottom: 0.5rem; }
    .stage-path { font-size: 0.8rem; color: #666; word-break: break-all; }
    .thumb-grid { display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: flex-start; }
    .explain { color: #555; font-size: 0.9rem; margin: 0.5rem 0 1rem; }
    """
    sections: list[str] = []
    for ann_dir, df, n_total_filtered, total_keys in stage_rows:
        label = html.escape(os.path.basename(ann_dir.rstrip(os.sep)) or ann_dir)
        path_esc = html.escape(ann_dir)
        ratio_s = html.escape(_format_filtered_ratio(n_total_filtered, total_keys))
        stats_esc = (
            f"<p class='img-stats'><strong>Content keys in this stage&rsquo;s parquet:</strong> total {total_keys}; "
            f"<strong>total filtered</strong> (<code>keep_mask</code> false): {n_total_filtered}; "
            f"<strong>filtered ratio</strong>: {ratio_s}</p>"
        )
        if df.empty:
            body = (
                f"<p class='empty'>No rows loaded from <code>--input-path</code> for up to {per_stage} "
                "picked keys (preview assumes image-like payloads where applicable).</p>"
            )
        else:
            thumbs = "".join(_row_to_html(row, j, "newly_filtered") for j, (_, row) in enumerate(df.iterrows(), start=1))
            body = f"<div class='thumb-grid'>{thumbs}</div>"
        sections.append(
            f"<section class='stage-block'><h2 class='stage-title'>Stage: {label}</h2>"
            f"<div class='stage-path'>{path_esc}</div>"
            f"{stats_esc}"
            f"{body}</section>"
        )

    body_join = "\n".join(sections)
    pat_esc = html.escape(annotation_pattern)
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Multi-stage annotation preview</title>
  <style>{style}{extra}</style>
</head>
<body>
  <h1>Preview: annotation stages</h1>
  <p class='img-stats'><strong>Workflow:</strong> (1) load annotation parquets per stage (expects <code>keep_mask</code> from <code>mint1t_annotation_pipeline</code>); (2) show <strong>total filtered</strong> count and ratio per stage from parquet; (3) load up to {per_stage} <em>newly</em> filtered keys per stage from <code>--input-path</code> tars for sample thumbnails. Optional <code>--max-tars</code> limits tar reads for step (3).</p>
  <p class='explain'>Annotation glob: <code>{pat_esc}</code>. Stage order uses the trailing integer in each directory name. Thumbnails use fixed height {DISPLAY_IMAGE_HEIGHT_PX}px for image modality.</p>
{body_join}
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-stage annotation preview: full content-key stats; sample rows from input."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="WebDataset tar path or directory (used only to load row payloads for preview samples)",
    )
    parser.add_argument(
        "--annotation-pattern",
        type=str,
        required=True,
        help=(
            "Glob matching one directory per stage (e.g. .../a_*). "
            "Dirs are ordered by trailing integer in the name. Each dir is scanned for parquet."
        ),
    )
    parser.add_argument("--port", type=int, default=8080, help="HTTP port")
    parser.add_argument(
        "--storage-options-json",
        type=str,
        default=None,
        help="JSON fsspec storage options for cloud paths",
    )
    parser.add_argument(
        "--max-tars",
        type=int,
        default=None,
        metavar="N",
        help="Optional: scan at most N tar files (sorted paths). Default: all tars under --input-path.",
    )
    parser.add_argument(
        "--per-stage",
        type=int,
        default=PER_STAGE_SAMPLES,
        help=f"Max sample rows to load per stage from input (default: {PER_STAGE_SAMPLES})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Thread workers (default: {DEFAULT_WORKERS})",
    )
    args = parser.parse_args()

    read_kwargs: dict = {}
    if args.storage_options_json:
        read_kwargs["storage_options"] = json.loads(args.storage_options_json)

    workers = max(1, args.workers)
    per_stage = max(1, args.per_stage)

    annotation_dirs = _expand_annotation_dirs(args.annotation_pattern)
    ak_and_kept = [
        _load_annotation_all_and_kept_sets(d, read_kwargs.get("storage_options"), workers) for d in annotation_dirs
    ]
    all_keys_per_stage = [t[0] for t in ak_and_kept]
    kept_sets = [t[1] for t in ak_and_kept]
    if not any(all_keys_per_stage):
        print("No annotation rows found in parquets under matched directories.")
        raise SystemExit(1)

    paths = _resolve_tar_paths(args.input_path, read_kwargs, args.max_tars)
    if not paths:
        print("No tar files found under --input-path.")
        raise SystemExit(1)

    plan = _stage_newly_filtered_plan(all_keys_per_stage, kept_sets, annotation_dirs, per_stage)
    need_keys: set[tuple[str, int]] = set()
    for _, _, keys, _ in plan:
        need_keys.update(keys)

    by_key = _fetch_rows_for_keys(paths, read_kwargs, need_keys, workers)
    stage_pick: list[tuple[str, pd.DataFrame, int, int]] = [
        (ann_dir, _rows_dict_to_dataframe(picked, by_key), n_total_f, total_k)
        for ann_dir, n_total_f, picked, total_k in plan
    ]

    to_mat = [df for _, df, _, _ in stage_pick if not df.empty]
    if to_mat:
        merged = pd.concat(to_mat, ignore_index=True)
        merged_mat = _materialize_rows_parallel(merged, read_kwargs, workers)
        rebuilt: list[tuple[str, pd.DataFrame, int, int]] = []
        start = 0
        for ann_dir, df, n_total_f, total_k in stage_pick:
            n = len(df)
            if n == 0:
                rebuilt.append((ann_dir, df, n_total_f, total_k))
            else:
                rebuilt.append((ann_dir, merged_mat.iloc[start : start + n].copy(), n_total_f, total_k))
                start += n
        stage_pick = rebuilt

    page = _build_html(stage_pick, args.annotation_pattern, per_stage)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?")[0] or "/"
            if path != "/":
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(page.encode("utf-8"))

        def log_message(self, format: str, *args: object) -> None:
            print(args[0] if args else "")

    class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        daemon_threads = True

    with ThreadingHTTPServer(("", args.port), Handler) as httpd:
        print(f"Open http://localhost:{args.port} — {len(annotation_dirs)} stage(s), glob {args.annotation_pattern!r}")
        try:
            httpd.serve_forever()
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    main()
