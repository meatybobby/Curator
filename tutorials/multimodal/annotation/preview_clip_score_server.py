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

"""HTTP server to preview interleaved rows with CLIP scores (omnicorpus_clip_score_pipeline parquet).

Mirrors :mod:`preview_annotation_server` (same shard readers and materialization) but joins
``sample_id`` / ``position`` to ``clip_score`` and shows one detail page per sample with
text, images, and scores (NaN shown as em dash). Tar and parquet loading are sequential;
``--workers`` is passed to :func:`materialize_omnicorpus_binary_content` for OmniCorpus only.
"""

from __future__ import annotations

import argparse
import base64
import html
import http.server
import io
import json
import socketserver
import sys
import urllib.parse
from pathlib import Path

import pandas as pd

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

try:
    from PIL import Image
except ImportError:
    Image = None

from preview_annotation_standalone import (  # noqa: E402
    DEFAULT_JSON_EXTENSIONS,
    DEFAULT_WEBDATASET_EXTENSIONS,
    FileGroupTask,
    InterleavedBatch,
    OmniCorpusReaderStage,
    WebdatasetReaderStage,
    get_all_file_paths_under,
    materialize_omnicorpus_binary_content,
    materialize_task_binary_content,
)

LIMIT_SAMPLES = 50
MAX_TARS = 30
DEFAULT_WORKERS = 16
MAX_IMAGE_PX = 800
THUMBNAIL_JPEG_QUALITY = 100


def _content_mask(df: pd.DataFrame) -> pd.Series:
    return (df["modality"] != "metadata") & (df["position"] >= 0)


def _read_one_clip_parquet(path: str, storage_options: dict | None) -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(path, storage_options=storage_options)
    except Exception:
        df = pd.read_parquet(path)
    if "sample_id" not in df.columns or "position" not in df.columns or "clip_score" not in df.columns:
        return None
    return df[["sample_id", "position", "clip_score"]]


def _combine_clip_parquets(annotation_path: str, storage_options: dict | None) -> pd.DataFrame:
    paths = get_all_file_paths_under(
        annotation_path,
        recurse_subdirectories=True,
        keep_extensions=[".parquet"],
        storage_options=storage_options,
    )
    if not paths:
        return pd.DataFrame()
    dfs: list[pd.DataFrame] = []
    for p in paths:
        part = _read_one_clip_parquet(p, storage_options)
        if part is not None:
            dfs.append(part)
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    combined["sample_id"] = combined["sample_id"].astype(str)
    combined["position"] = combined["position"].astype(int)
    return combined.drop_duplicates(subset=["sample_id", "position"], keep="last")


def _target_sample_ids(scores_df: pd.DataFrame, limit_samples: int) -> list[str]:
    if scores_df.empty:
        return []
    sids = scores_df["sample_id"].drop_duplicates().astype(str).tolist()
    sids.sort()
    return sids[:limit_samples]


def _read_one_tar_omnicorpus_for_samples(
    tar_path: str,
    read_kwargs: dict,
    target_sids: set[str],
    include_general_metadata: bool,
    max_batch_bytes: int | None,
    omni_materialize_workers: int,
) -> pd.DataFrame:
    reader = OmniCorpusReaderStage(
        max_batch_bytes=max_batch_bytes,
        read_kwargs=read_kwargs,
        include_general_metadata=include_general_metadata,
    )
    task = FileGroupTask(
        task_id="clip_preview",
        dataset_name="preview",
        data=[tar_path],
        _metadata={"source_files": [tar_path]},
    )
    out = reader.process(task)
    batches = out if isinstance(out, list) else [out]
    parts: list[pd.DataFrame] = []
    mat_workers = max(1, omni_materialize_workers)
    for batch in batches:
        mat = materialize_omnicorpus_binary_content(
            batch,
            io_kwargs=read_kwargs,
            num_workers=mat_workers,
        )
        df = mat.to_pandas()
        if df.empty:
            continue
        content = _content_mask(df)
        in_sid = df["sample_id"].astype(str).isin(target_sids)
        parts.append(df[content & in_sid])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _collect_omnicorpus_content(
    paths: list[str],
    read_kwargs: dict,
    target_sids: set[str],
    include_general_metadata: bool,
    omni_max_batch_bytes: int | None,
    omni_materialize_workers: int,
) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    chunks: list[pd.DataFrame] = []
    for tar_path in paths:
        df = _read_one_tar_omnicorpus_for_samples(
            tar_path,
            read_kwargs,
            target_sids,
            include_general_metadata,
            omni_max_batch_bytes,
            omni_materialize_workers,
        )
        if not df.empty:
            chunks.append(df)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _read_one_tar_webdataset_for_samples(
    tar_path: str,
    read_kwargs: dict,
    target_sids: set[str],
) -> pd.DataFrame:
    reader = WebdatasetReaderStage(
        source_id_field="pdf_name",
        read_kwargs=read_kwargs,
        materialize_on_read=False,
        max_batch_bytes=None,
        json_extensions=tuple(DEFAULT_JSON_EXTENSIONS),
    )
    task = FileGroupTask(
        task_id="clip_preview",
        dataset_name="preview",
        data=[tar_path],
        _metadata={"source_files": [tar_path]},
    )
    out = reader.process(task)
    batches = out if isinstance(out, list) else [out]
    parts: list[pd.DataFrame] = []
    for batch in batches:
        df = batch.to_pandas()
        if df.empty:
            continue
        content = _content_mask(df)
        in_sid = df["sample_id"].astype(str).isin(target_sids)
        parts.append(df[content & in_sid])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _collect_webdataset_content(paths: list[str], read_kwargs: dict, target_sids: set[str]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    chunks: list[pd.DataFrame] = []
    for tar_path in paths:
        df = _read_one_tar_webdataset_for_samples(tar_path, read_kwargs, target_sids)
        if not df.empty:
            chunks.append(df)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _collect_merged_content(
    input_path: str,
    scores_df: pd.DataFrame,
    read_kwargs: dict | None,
    max_tars: int,
    limit_samples: int,
    *,
    input_source: str,
    include_general_metadata: bool,
    omni_max_batch_bytes: int | None,
    omni_materialize_workers: int,
) -> pd.DataFrame:
    read_kwargs = read_kwargs or {}
    target_list = _target_sample_ids(scores_df, limit_samples)
    if not target_list:
        return pd.DataFrame()
    target_sids = set(target_list)

    if isinstance(input_path, str):
        paths = get_all_file_paths_under(
            input_path,
            recurse_subdirectories=True,
            keep_extensions=list(DEFAULT_WEBDATASET_EXTENSIONS),
            storage_options=read_kwargs.get("storage_options"),
        )
    else:
        paths = list(input_path)
    paths = paths[:max_tars]

    if input_source == "omnicorpus":
        content = _collect_omnicorpus_content(
            paths,
            read_kwargs,
            target_sids,
            include_general_metadata,
            omni_max_batch_bytes,
            omni_materialize_workers,
        )
    else:
        content = _collect_webdataset_content(paths, read_kwargs, target_sids)
        if not content.empty:
            content = _materialize_rows(content, read_kwargs)

    if content.empty:
        return content

    content = content.copy()
    content["sample_id"] = content["sample_id"].astype(str)
    content["position"] = content["position"].astype(int)
    merged = content.merge(scores_df, on=["sample_id", "position"], how="left")
    return merged[merged["sample_id"].isin(target_list)]


def _materialize_rows(df: pd.DataFrame, io_kwargs: dict | None) -> pd.DataFrame:
    if df.empty:
        return df
    task = InterleavedBatch(
        task_id="clip_preview_materialize",
        dataset_name="preview",
        data=df,
        _metadata={},
    )
    out = materialize_task_binary_content(task, only_missing_binary=True, io_kwargs=io_kwargs)
    return out.to_pandas()


def _safe_str(val: object) -> str:
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val)


def _format_clip_score(val: object) -> str:
    if val is None:
        return "—"
    try:
        if pd.isna(val):
            return "—"
    except (TypeError, ValueError):
        pass
    try:
        return f"{float(val):.4f}"
    except (TypeError, ValueError):
        return "—"


def _image_clip_min_max_strings(sub: pd.DataFrame) -> tuple[str, str]:
    """Min/max of ``clip_score`` over image rows only; ``('—', '—')`` if none are numeric."""
    if sub.empty or "clip_score" not in sub.columns:
        return "—", "—"
    imgs = sub[sub["modality"].astype(str) == "image"]
    if imgs.empty:
        return "—", "—"
    scores = pd.to_numeric(imgs["clip_score"], errors="coerce").dropna()
    if scores.empty:
        return "—", "—"
    lo, hi = float(scores.min()), float(scores.max())
    return f"{lo:.4f}", f"{hi:.4f}"


def _to_bytes(raw: object) -> bytes | None:
    if raw is None:
        return None
    try:
        if pd.isna(raw):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, bytearray):
        return bytes(raw)
    if isinstance(raw, memoryview):
        return raw.tobytes()
    try:
        if hasattr(raw, "as_py"):
            raw = raw.as_py()
        if hasattr(raw, "tobytes"):
            return raw.tobytes()
    except Exception:
        pass
    try:
        return bytes(raw)
    except (TypeError, ValueError):
        return None


def _scale_image_for_display(b: bytes) -> tuple[bytes, str] | None:
    if Image is None:
        return None
    try:
        with Image.open(io.BytesIO(b)) as img:
            img = img.convert("RGB")
            img.thumbnail((MAX_IMAGE_PX, MAX_IMAGE_PX), getattr(Image, "Resampling", Image).LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=THUMBNAIL_JPEG_QUALITY, optimize=True)
            return buf.getvalue(), "image/jpeg"
    except (OSError, ValueError, TypeError):
        return None


def _row_to_html(row: pd.Series, index: int) -> str:
    sid = html.escape(_safe_str(row.get("sample_id")))
    pos = row.get("position", "")
    mod = html.escape(_safe_str(row.get("modality")))
    is_image = _safe_str(row.get("modality")) == "image"
    clip_suffix = ""
    if is_image:
        clip_suffix = " clip_score=" + html.escape(_format_clip_score(row.get("clip_score")))
    text_val = row.get("text_content")
    text = html.escape(_safe_str(text_val))

    parts = [
        f"<div class='row'><span class='meta'>#{index} sample_id={sid} position={pos} "
        f"modality={mod}{clip_suffix}</span>"
    ]
    if is_image:
        raw = row.get("binary_content")
        b = _to_bytes(raw)
        if b and len(b) > 0:
            scaled = _scale_image_for_display(b)
            if scaled is not None:
                thumb_b, ct = scaled
                b64 = base64.b64encode(thumb_b).decode("ascii")
                parts.append(f'<img src="data:{ct};base64,{b64}" alt="image" />')
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
    .back-link { display: inline-block; margin-bottom: 1rem; }
"""


def _build_detail_html(sid: str, df: pd.DataFrame) -> str:
    if "position" in df.columns:
        df = df.sort_values("position", kind="stable")
    rows_html = "".join(_row_to_html(row, i) for i, (_, row) in enumerate(df.iterrows(), start=1))
    sid_esc = html.escape(sid)
    lo, hi = _image_clip_min_max_strings(df)
    if lo != "—" and hi != "—":
        clip_summary = f" Image rows: CLIP min {html.escape(lo)}, max {html.escape(hi)}."
    else:
        clip_summary = ""
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>CLIP scores: {sid_esc}</title>
  <style>{_shared_styles()}</style>
</head>
<body>
  <a class="back-link" href="/">&larr; Back to summary</a>
  <section class="sample">
    <h2>sample_id: {sid_esc}</h2>
    <p class="meta">{len(df)} content rows (text + image), ordered by position.{clip_summary}</p>
    {rows_html if not df.empty else "<p class='empty'>none</p>"}
  </section>
</body>
</html>"""


def _build_main_html(merged_df: pd.DataFrame) -> str:
    if merged_df.empty:
        sample_ids: list[str] = []
    else:
        sample_ids = sorted(merged_df["sample_id"].astype(str).drop_duplicates().tolist())

    def table_rows(sids: list[str]) -> str:
        out = []
        for s in sids:
            sub = merged_df[merged_df["sample_id"].astype(str) == s]
            sid_esc = html.escape(s)
            link = "/?sample=" + urllib.parse.quote(s, safe="")
            n_img = len(sub[sub["modality"].astype(str) == "image"])
            lo, hi = _image_clip_min_max_strings(sub)
            lo_esc = html.escape(lo)
            hi_esc = html.escape(hi)
            out.append(
                f"    <tr><td>{sid_esc}</td><td>{len(sub)}</td><td>{n_img}</td>"
                f"<td>{lo_esc}</td><td>{hi_esc}</td>"
                f'<td><a href="{html.escape(link)}">Details</a></td></tr>'
            )
        return "\n".join(out) if out else "    <tr><td colspan='6'>None</td></tr>"

    body = table_rows(sample_ids)
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>CLIP score preview – by sample_id</title>
  <style>{_shared_styles()}</style>
</head>
<body>
  <h1>CLIP scores: {len(sample_ids)} samples ({len(merged_df)} rows)</h1>
  <p>Parquet <code>clip_score</code> joined to shard rows. Min/max are over image rows only. Click Details for per-row view.</p>
  <table class="summary-table">
    <thead><tr><th>sample_id</th><th>rows</th><th>images</th><th>min clip</th><th>max clip</th><th></th></tr></thead>
    <tbody>
{body}
    </tbody>
  </table>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Web server: preview interleaved rows with CLIP scores from clip-score parquet"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Original tar path or directory (format depends on --input-source)",
    )
    parser.add_argument(
        "--annotation-path",
        type=str,
        required=True,
        help="Directory of parquet files from omnicorpus_clip_score_pipeline (sample_id, position, clip_score)",
    )
    parser.add_argument(
        "--input-source",
        type=str,
        choices=("webdataset", "omnicorpus"),
        default="omnicorpus",
        help="Shard format (default: omnicorpus for OmniCorpus-CC tars)",
    )
    parser.add_argument(
        "--include-general-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="OmniCorpus only: join url/safety from .general_metadata.pkl",
    )
    parser.add_argument(
        "--omni-max-batch-bytes",
        type=int,
        default=None,
        help="OmniCorpus only: OmniCorpusReaderStage.max_batch_bytes",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--storage-options-json",
        type=str,
        default=None,
        help="JSON fsspec storage options for cloud paths",
    )
    parser.add_argument("--max-samples", type=int, default=LIMIT_SAMPLES, help="Max distinct sample_ids from parquet")
    parser.add_argument("--max-tars", type=int, default=MAX_TARS)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "OmniCorpus only: passed as num_workers to materialize_omnicorpus_binary_content per batch "
            f"(default: {DEFAULT_WORKERS}). Ignored for --input-source webdataset."
        ),
    )
    args = parser.parse_args()

    read_kwargs: dict = {}
    if args.storage_options_json:
        read_kwargs["storage_options"] = json.loads(args.storage_options_json)

    workers = max(1, args.workers)
    storage_opts = read_kwargs.get("storage_options")
    scores_df = _combine_clip_parquets(args.annotation_path, storage_opts)
    if scores_df.empty:
        print("No clip_score parquet found under --annotation-path (need sample_id, position, clip_score).")
        return

    omni_mat = workers if args.input_source == "omnicorpus" else 1
    merged = _collect_merged_content(
        args.input_path,
        scores_df,
        read_kwargs,
        max_tars=args.max_tars,
        limit_samples=args.max_samples,
        input_source=args.input_source,
        include_general_metadata=args.include_general_metadata,
        omni_max_batch_bytes=args.omni_max_batch_bytes,
        omni_materialize_workers=omni_mat,
    )
    if merged.empty:
        hint = "OmniCorpus tars" if args.input_source == "omnicorpus" else "WebDataset tars"
        print(
            f"No matching rows in shards for the first {args.max_samples} sample_ids from parquet. Check --input-path and {hint}."
        )
        return

    main_html = _build_main_html(merged)
    sample_ids = sorted(merged["sample_id"].astype(str).drop_duplicates().tolist())
    detail_html_by_id: dict[str, str] = {}
    for sid in sample_ids:
        sub = merged[merged["sample_id"].astype(str) == sid].copy()
        detail_html_by_id[sid] = _build_detail_html(sid, sub)

    def handler_factory(main: str, details: dict[str, str]):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = self.path.split("?")[0] or "/"
                query = self.path.split("?", 1)[1] if "?" in self.path else ""
                params = urllib.parse.parse_qs(query)
                sample_id = params.get("sample", [None])[0]
                if path != "/":
                    self.send_response(404)
                    self.end_headers()
                    return
                if sample_id is not None:
                    sample_id = urllib.parse.unquote(sample_id)
                    body = details.get(sample_id)
                    if body is None:
                        self.send_response(404)
                        self.end_headers()
                        return
                else:
                    body = main
                self.send_response(200)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body.encode("utf-8"))

            def log_message(self, format: str, *args: object) -> None:
                print(args[0] if args else "")

        return Handler

    Handler = handler_factory(main_html, detail_html_by_id)

    class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        daemon_threads = True

    with ThreadingHTTPServer(("", args.port), Handler) as httpd:
        omni_note = f" (OmniCorpus materialize workers: {workers})" if args.input_source == "omnicorpus" else ""
        print(f"Open http://localhost:{args.port} for CLIP score preview.{omni_note}")
        try:
            httpd.serve_forever()
        finally:
            print("\nShutting down.")
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    main()
