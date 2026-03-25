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

"""Self-contained WebDataset read + binary materialization for preview_annotation_server (no nemo_curator)."""

from __future__ import annotations

import io
import json
import logging
import mimetypes
import posixpath
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import fsspec
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from fsspec.core import url_to_fs
from PIL import Image as _Image

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_JSON_EXTENSIONS",
    "DEFAULT_WEBDATASET_EXTENSIONS",
    "FileGroupTask",
    "InterleavedBatch",
    "WebdatasetReaderStage",
    "get_all_file_paths_under",
    "materialize_task_binary_content",
]

# -- constants (mirrors nemo_curator.stages.interleaved.utils.constants) --

DEFAULT_WEBDATASET_EXTENSIONS = (".tar", ".tar.gz", ".tgz")
DEFAULT_JSON_EXTENSIONS = (".json",)
DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp", ".gif")

# -- filesystem listing (mirrors nemo_curator.utils.file_utils) --


def _is_remote_url(url: str) -> bool:
    fs, _ = url_to_fs(url)
    proto = fs.protocol[0] if isinstance(fs.protocol, (list, tuple)) else fs.protocol
    return proto not in (None, "file")


def _gather_extention(path: str) -> str:
    name = posixpath.basename(path.rstrip("/"))
    return posixpath.splitext(name)[1][1:].casefold()


def _gather_file_records(
    path: str,
    recurse_subdirectories: bool,
    keep_extensions: str | list[str] | None,
    storage_options: dict[str, str] | None,
    fs: fsspec.AbstractFileSystem | None,
    include_size: bool,
) -> list[tuple[str, int]]:
    fs = fs or fsspec.core.url_to_fs(path, **(storage_options or {}))[0]
    allowed_exts = (
        None
        if keep_extensions is None
        else {
            e.casefold().lstrip(".")
            for e in ([keep_extensions] if isinstance(keep_extensions, str) else keep_extensions)
        }
    )
    normalize = fs.unstrip_protocol if _is_remote_url(path) else (lambda x: x)
    roots = fs.expand_path(path, recursive=False)
    records: list[tuple[str, int]] = []

    for root in roots:
        if fs.isdir(root):
            listing = fs.find(
                root,
                maxdepth=None if recurse_subdirectories else 1,
                withdirs=False,
                detail=include_size,
            )
            if include_size:
                entries = [(p, info.get("size")) for p, info in listing.items()]
            else:
                entries = [(p, None) for p in listing]

        elif fs.exists(root):
            entries = [(root, fs.info(root).get("size") if include_size else None)]
        else:
            entries = []

        for raw_path, raw_size in entries:
            if (allowed_exts is None) or (_gather_extention(raw_path) in allowed_exts):
                records.append((normalize(raw_path), -1 if include_size and raw_size is None else raw_size))

    return records


def get_all_file_paths_under(
    path: str,
    recurse_subdirectories: bool = False,
    keep_extensions: str | list[str] | None = None,
    storage_options: dict[str, str] | None = None,
    fs: fsspec.AbstractFileSystem | None = None,
) -> list[str]:
    return sorted(
        p
        for p, _ in _gather_file_records(
            path, recurse_subdirectories, keep_extensions, storage_options, fs, include_size=False
        )
    )


# -- validation helpers (mirrors nemo_curator.stages.interleaved.utils.validation_utils) --


def require_source_id_field(source_id_field: str) -> str:
    if source_id_field:
        return source_id_field
    msg = "source_id_field must be provided explicitly (e.g., 'pdf_name')"
    raise ValueError(msg)


def resolve_storage_options(
    task: object | None = None,
    io_kwargs: dict[str, object] | None = None,
) -> dict[str, object]:
    source_storage_options = None
    if task is not None and hasattr(task, "_metadata"):
        meta = getattr(task, "_metadata", None)
        if isinstance(meta, dict):
            source_storage_options = meta.get("source_storage_options")
    if isinstance(source_storage_options, dict) and source_storage_options:
        return source_storage_options
    storage_options = (io_kwargs or {}).get("storage_options")
    return storage_options if isinstance(storage_options, dict) else {}


def validate_and_project_source_fields(
    sample: dict[str, Any],
    fields: tuple[str, ...] | None,
    excluded_fields: set[str],
) -> dict[str, Any]:
    selected = [key for key in sample if key not in excluded_fields] if fields is None else list(fields)
    if fields is not None:
        reserved = sorted(f for f in selected if f in excluded_fields)
        if reserved:
            msg = f"fields contains reserved keys: {reserved}"
            raise ValueError(msg)
        missing = sorted(f for f in selected if f not in sample)
        if missing:
            logger.warning("Requested fields not found in source sample (filling with None): %s", missing)
    result: dict[str, Any] = {}
    for fname in selected:
        if fname not in sample:
            result[fname] = None
        else:
            value = sample[fname]
            result[fname] = json.dumps(value, ensure_ascii=True) if isinstance(value, (dict, list)) else value
    return result


# -- interleaved task types (mirrors nemo_curator.tasks.interleaved) --

INTERLEAVED_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("position", pa.int32(), nullable=False),
        pa.field("modality", pa.string(), nullable=False),
        pa.field("content_type", pa.string(), nullable=True),
        pa.field("text_content", pa.string(), nullable=True),
        pa.field("binary_content", pa.large_binary(), nullable=True),
        pa.field("source_ref", pa.string(), nullable=True),
        pa.field("materialize_error", pa.string(), nullable=True),
    ]
)

RESERVED_COLUMNS: frozenset[str] = frozenset(INTERLEAVED_SCHEMA.names)


@dataclass
class InterleavedBatch:
    task_id: str
    dataset_name: str
    data: pa.Table | pd.DataFrame = field(default_factory=lambda: pa.Table.from_pylist([], schema=INTERLEAVED_SCHEMA))
    _stage_perf: list[Any] = field(default_factory=list)
    _metadata: dict[str, Any] = field(default_factory=dict)

    def to_pyarrow(self) -> pa.Table:
        if isinstance(self.data, pa.Table):
            return self.data
        if isinstance(self.data, pd.DataFrame):
            return pa.Table.from_pandas(self.data, preserve_index=False)
        msg = f"Cannot convert {type(self.data)} to PyArrow table"
        raise TypeError(msg)

    def to_pandas(self) -> pd.DataFrame:
        if isinstance(self.data, pd.DataFrame):
            return self.data
        if isinstance(self.data, pa.Table):
            return self.data.to_pandas(types_mapper=pd.ArrowDtype)
        msg = f"Cannot convert {type(self.data)} to Pandas DataFrame"
        raise TypeError(msg)

    @staticmethod
    def build_source_ref(
        path: str | None,
        member: str | None,
        byte_offset: int | None = None,
        byte_size: int | None = None,
        frame_index: int | None = None,
    ) -> str:
        ref: dict[str, object] = {
            "path": path,
            "member": member,
            "byte_offset": byte_offset,
            "byte_size": byte_size,
        }
        if frame_index is not None:
            ref["frame_index"] = frame_index
        return json.dumps(ref, ensure_ascii=True)

    @staticmethod
    def parse_source_ref(source_value: str | None) -> dict[str, str | int | None]:
        if source_value is None or pd.isna(source_value) or source_value == "":
            return {"path": None, "member": None, "byte_offset": None, "byte_size": None, "frame_index": None}
        parsed = json.loads(source_value)
        if not isinstance(parsed, dict):
            msg = "source_ref must decode to a JSON object"
            raise TypeError(msg)

        path = parsed.get("path")
        member = parsed.get("member")
        byte_offset = parsed.get("byte_offset")
        byte_size = parsed.get("byte_size")
        frame_index = parsed.get("frame_index")

        return {
            "path": path if path is None else str(path),
            "member": member if member is None else str(member),
            "byte_offset": int(byte_offset) if byte_offset is not None else None,
            "byte_size": int(byte_size) if byte_size is not None else None,
            "frame_index": int(frame_index) if frame_index is not None else None,
        }

    def with_parsed_source_ref_columns(self, prefix: str = "_src_") -> pd.DataFrame:
        df = self.to_pandas().copy()
        parsed = [self.parse_source_ref(value) for value in df["source_ref"].tolist()]
        parsed_df = pd.DataFrame.from_records(
            parsed,
            columns=["path", "member", "byte_offset", "byte_size", "frame_index"],
        )
        for col in parsed_df.columns:
            df[f"{prefix}{col}"] = parsed_df[col].to_numpy(copy=False)
        return df


@dataclass
class FileGroupTask:
    task_id: str
    dataset_name: str
    data: list[str] = field(default_factory=list)
    reader_config: dict[str, Any] = field(default_factory=dict)
    _stage_perf: list[Any] = field(default_factory=list)
    _metadata: dict[str, Any] = field(default_factory=dict)


# -- table splitting (mirrors nemo_curator.core.utils.split_table_by_group_max_bytes) --


def split_table_by_group_max_bytes(
    table: pa.Table,
    group_column: str,
    max_batch_bytes: int | None,
) -> list[pa.Table]:
    if max_batch_bytes is None or table.num_rows == 0:
        return [table]
    if max_batch_bytes <= 0:
        msg = f"max_batch_bytes must be > 0, got {max_batch_bytes}"
        raise ValueError(msg)
    if group_column not in table.column_names:
        msg = f"Group column '{group_column}' not found in table"
        raise ValueError(msg)

    sort_indices = pc.sort_indices(table, sort_keys=[(group_column, "ascending")])
    table = table.take(sort_indices)
    col = table[group_column]
    n = table.num_rows

    if n <= 1:
        return [table]

    ne = pc.not_equal(col.slice(1), col.slice(0, n - 1))
    split_points = pc.indices_nonzero(ne).to_pylist()
    group_starts = [0, *(p + 1 for p in split_points)]
    group_ends = [*(p + 1 for p in split_points), n]

    avg_bytes_per_row = table.nbytes / n
    chunk_split_indices: list[int] = []
    chunk_bytes = 0.0
    for i, (gs, ge) in enumerate(zip(group_starts, group_ends, strict=True)):
        group_bytes = (ge - gs) * avg_bytes_per_row
        if i > 0 and chunk_bytes > 0 and (chunk_bytes + group_bytes > max_batch_bytes):
            chunk_split_indices.append(gs)
            chunk_bytes = 0.0
        chunk_bytes += group_bytes

    if not chunk_split_indices:
        return [table]
    all_starts = [0, *chunk_split_indices]
    all_ends = [*chunk_split_indices, n]
    return [table.slice(s, e - s) for s, e in zip(all_starts, all_ends, strict=True)]


def _extract_tiff_frame(tiff_bytes: bytes, frame_index: int) -> bytes | None:
    try:
        with _Image.open(io.BytesIO(tiff_bytes)) as img:
            if img.format != "TIFF":
                return tiff_bytes
            if frame_index >= getattr(img, "n_frames", 1):
                return None
            img.seek(frame_index)
            frame = img.copy()
        buf = io.BytesIO()
        frame.save(buf, format="TIFF")
        return buf.getvalue()
    except (OSError, SyntaxError, ValueError):
        return None


# -- Webdataset reader (mirrors nemo_curator.stages.interleaved.io.readers.webdataset) --


@dataclass
class _ReadContext:
    tar_path: str
    member_names: set[str]
    member_info: dict[str, tarfile.TarInfo]
    storage_options: dict[str, object]
    byte_cache: dict[str, bytes | None]


@dataclass
class _SampleContext:
    sample_id: str
    sample: dict[str, Any]
    tar_path: str
    json_member_name: str
    member_names: set[str]
    member_info: dict[str, tarfile.TarInfo] | None
    passthrough: dict[str, Any]
    per_image_passthrough: dict[str, list[Any]]
    per_text_passthrough: dict[str, list[Any]]


@dataclass
class WebdatasetReaderStage:
    read_kwargs: dict[str, Any] = field(default_factory=dict)
    materialize_on_read: bool = False
    max_batch_bytes: int | None = None
    json_extensions: tuple[str, ...] = DEFAULT_JSON_EXTENSIONS
    image_extensions: tuple[str, ...] = field(default_factory=lambda: DEFAULT_IMAGE_EXTENSIONS)
    source_id_field: str = ""
    sample_id_field: str | None = None
    texts_field: str = "texts"
    images_field: str = "images"
    image_member_field: str | None = None
    fields: tuple[str, ...] | None = None
    per_image_fields: tuple[str, ...] = ()
    per_text_fields: tuple[str, ...] = ()
    name: str = "webdataset_reader"

    def __post_init__(self) -> None:
        self.source_id_field = require_source_id_field(self.source_id_field)

    def _build_source_ref(
        self,
        ctx: _SampleContext,
        content_key: str | None,
        *,
        frame_index: int | None = None,
    ) -> str:
        if content_key is None:
            return InterleavedBatch.build_source_ref(path=None, member=None)
        byte_offset = None
        byte_size = None
        if ctx.member_info and content_key in ctx.member_info:
            info = ctx.member_info[content_key]
            byte_offset = info.offset_data
            byte_size = info.size
        return InterleavedBatch.build_source_ref(
            path=ctx.tar_path,
            member=content_key,
            byte_offset=byte_offset,
            byte_size=byte_size,
            frame_index=frame_index,
        )

    @staticmethod
    def _build_row(ctx: _SampleContext, row_fields: dict[str, Any]) -> dict[str, Any]:
        return {
            "sample_id": ctx.sample_id,
            "position": row_fields.get("position"),
            "modality": row_fields.get("modality"),
            "content_type": row_fields.get("content_type"),
            "text_content": row_fields.get("text_content"),
            "binary_content": row_fields.get("binary_content"),
            "source_ref": row_fields.get("source_ref"),
            "materialize_error": None,
        }

    def _metadata_row(self, ctx: _SampleContext) -> dict[str, Any]:
        return {
            **self._build_row(
                ctx,
                {
                    "position": -1,
                    "modality": "metadata",
                    "content_type": "application/json",
                    "source_ref": self._build_source_ref(ctx, ctx.json_member_name),
                },
            ),
            **ctx.passthrough,
        }

    @staticmethod
    def _apply_per_modality_fields(
        row: dict[str, Any],
        passthrough: dict[str, list[Any]],
        index: int,
    ) -> None:
        for field_name, values in passthrough.items():
            if index < len(values):
                val = values[index]
                row[field_name] = json.dumps(val, ensure_ascii=True) if isinstance(val, (dict, list)) else val

    @staticmethod
    def _warn_per_modality_length_mismatch(
        sample_id: str,
        passthrough: dict[str, list[Any]],
        actual_count: int,
        modality: str,
    ) -> None:
        for field_name, values in passthrough.items():
            if actual_count != len(values):
                logger.warning(
                    "sample_id=%s: per_%s_field %r has %d values but %d non-None %ss",
                    sample_id,
                    modality,
                    field_name,
                    len(values),
                    actual_count,
                    modality,
                )

    def _text_rows(self, ctx: _SampleContext) -> list[dict[str, Any]]:
        texts = ctx.sample.get(self.texts_field)
        if not isinstance(texts, list):
            return []
        source_ref = self._build_source_ref(ctx, ctx.json_member_name)
        rows: list[dict[str, Any]] = []
        non_none_counter = 0
        for idx, text_value in enumerate(texts):
            if text_value is None:
                continue
            row = self._build_row(
                ctx,
                {
                    "position": idx,
                    "modality": "text",
                    "content_type": "text/plain",
                    "text_content": str(text_value),
                    "source_ref": source_ref,
                },
            )
            self._apply_per_modality_fields(row, ctx.per_text_passthrough, non_none_counter)
            non_none_counter += 1
            rows.append(row)
        self._warn_per_modality_length_mismatch(ctx.sample_id, ctx.per_text_passthrough, non_none_counter, "text")
        return rows

    def _image_rows(self, ctx: _SampleContext) -> list[dict[str, Any]]:
        images = ctx.sample.get(self.images_field)
        if not isinstance(images, list):
            return []
        image_member_name = self._resolve_default_image_member_name(
            ctx.sample_id,
            ctx.sample,
            images,
            ctx.member_names,
        )
        rows: list[dict[str, Any]] = []
        frame_counters: dict[str, int] = {}
        non_none_counter = 0
        for idx, image_token in enumerate(images):
            if image_token is None:
                continue
            content_key = self._resolve_image_content_key(image_token, image_member_name, ctx.member_names)
            content_type, _ = mimetypes.guess_type(content_key or image_member_name or "")
            frame_index = None
            is_multiframe_candidate = content_type == "image/tiff"
            if content_key is not None and is_multiframe_candidate:
                frame_index = frame_counters.get(content_key, 0)
                frame_counters[content_key] = frame_index + 1
            row = self._build_row(
                ctx,
                {
                    "position": idx,
                    "modality": "image",
                    "content_type": content_type or ("application/octet-stream" if image_member_name else None),
                    "source_ref": self._build_source_ref(ctx, content_key, frame_index=frame_index),
                },
            )
            self._apply_per_modality_fields(row, ctx.per_image_passthrough, non_none_counter)
            non_none_counter += 1
            rows.append(row)
        self._warn_per_modality_length_mismatch(ctx.sample_id, ctx.per_image_passthrough, non_none_counter, "image")
        return rows

    def _rows_from_sample(self, ctx: _SampleContext) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        rows.append(self._metadata_row(ctx))
        content_rows = self._text_rows(ctx) + self._image_rows(ctx)
        content_rows.sort(key=lambda r: r["position"])
        rows.extend(content_rows)
        per_modality_keys = set(ctx.per_image_passthrough) | set(ctx.per_text_passthrough)
        if per_modality_keys:
            for row in rows:
                for key in per_modality_keys:
                    row.setdefault(key, None)
        return rows

    def _build_passthrough_row(self, sample: dict[str, Any]) -> dict[str, Any]:
        excluded = RESERVED_COLUMNS | {
            self.source_id_field,
            *([self.sample_id_field] if self.sample_id_field else []),
            self.texts_field,
            self.images_field,
            *([self.image_member_field] if self.image_member_field else []),
            *self.per_image_fields,
            *self.per_text_fields,
        }
        return validate_and_project_source_fields(sample=sample, fields=self.fields, excluded_fields=excluded)

    @staticmethod
    def _extract_per_modality_fields(
        sample: dict[str, Any],
        field_names: tuple[str, ...],
    ) -> dict[str, list[Any]]:
        result: dict[str, list[Any]] = {}
        for fname in field_names:
            if fname not in sample:
                logger.warning("per-modality field %r not found in source sample", fname)
                continue
            value = sample[fname]
            if isinstance(value, list):
                result[fname] = value
            else:
                msg = f"per-modality field '{fname}' must be a list, got {type(value).__name__}"
                raise TypeError(msg)
        return result

    def _empty_output_schema(self) -> pa.Schema:
        schema = INTERLEAVED_SCHEMA
        seen = set(self.fields or ())
        all_extra = list(self.fields or ())
        for f in (*self.per_image_fields, *self.per_text_fields):
            if f not in seen:
                all_extra.append(f)
                seen.add(f)
        if not all_extra:
            return schema
        existing = set(schema.names)
        extra_fields = [pa.field(name, pa.null()) for name in all_extra if name not in existing]
        return pa.schema([*schema, *extra_fields]) if extra_fields else schema

    @staticmethod
    def _reconcile_schema(inferred: pa.Schema) -> pa.Schema:
        canonical = {f.name: f for f in INTERLEAVED_SCHEMA}
        fields = []
        for f in inferred:
            if f.name in canonical:
                fields.append(canonical[f.name])
            else:
                fields.append(f)
        return pa.schema(fields)

    def _resolve_default_image_member_name(
        self,
        sample_id: str,
        sample: dict[str, Any],
        images: list[object] | None,
        member_names: set[str],
    ) -> str | None:
        if self.image_member_field:
            image_member_name = sample.get(self.image_member_field)
            if isinstance(image_member_name, str) and image_member_name in member_names:
                return image_member_name
        if isinstance(images, list):
            for image_token in images:
                if isinstance(image_token, str) and image_token in member_names:
                    return image_token
        return next(
            (f"{sample_id}{ext}" for ext in self.image_extensions if f"{sample_id}{ext}" in member_names),
            None,
        )

    @staticmethod
    def _resolve_image_content_key(
        image_token: object,
        default_image_member_name: str | None,
        member_names: set[str],
    ) -> str | None:
        if image_token is None:
            return None
        if isinstance(image_token, str) and image_token in member_names:
            return image_token
        return default_image_member_name

    @staticmethod
    def _extract_tar_member(tf: tarfile.TarFile, member_name: str, cache: dict[str, bytes | None]) -> bytes | None:
        if member_name in cache:
            return cache[member_name]
        try:
            extracted = tf.extractfile(member_name)
        except KeyError:
            extracted = None
        payload = extracted.read() if extracted is not None else None
        cache[member_name] = payload
        return payload

    def _rows_from_member(
        self,
        tf: tarfile.TarFile,
        member: tarfile.TarInfo,
        read_ctx: _ReadContext,
    ) -> list[dict[str, Any]]:
        extracted = tf.extractfile(member)
        if extracted is None:
            return []
        payload = json.load(extracted)
        sample_id = (
            str(payload.get(self.sample_id_field))
            if self.sample_id_field and payload.get(self.sample_id_field) is not None
            else Path(member.name).stem
        )
        ctx = _SampleContext(
            sample_id=sample_id,
            sample=payload,
            tar_path=read_ctx.tar_path,
            json_member_name=member.name,
            member_names=read_ctx.member_names,
            member_info=read_ctx.member_info,
            passthrough=self._build_passthrough_row(payload),
            per_image_passthrough=self._extract_per_modality_fields(payload, self.per_image_fields),
            per_text_passthrough=self._extract_per_modality_fields(payload, self.per_text_fields),
        )
        sample_rows = self._rows_from_sample(ctx)
        if self.materialize_on_read:
            for row in sample_rows:
                if row["modality"] != "image" or row["position"] < 0:
                    continue
                parsed_ref = InterleavedBatch.parse_source_ref(row["source_ref"])
                content_key = parsed_ref.get("member")
                if not content_key:
                    continue
                raw_bytes = self._extract_tar_member(tf, content_key, read_ctx.byte_cache)
                if raw_bytes is None:
                    row["materialize_error"] = f"missing member '{content_key}'"
                else:
                    frame_index = parsed_ref.get("frame_index")
                    if frame_index is not None:
                        extracted_frame = _extract_tiff_frame(raw_bytes, frame_index)
                        if extracted_frame is None:
                            row["materialize_error"] = f"failed to extract frame {frame_index} from '{content_key}'"
                        else:
                            raw_bytes = extracted_frame
                row["binary_content"] = raw_bytes
            read_ctx.byte_cache.clear()
        return sample_rows

    def process(self, task: FileGroupTask) -> InterleavedBatch | list[InterleavedBatch]:
        rows: list[dict[str, Any]] = []
        storage_options = resolve_storage_options(io_kwargs=self.read_kwargs)

        for tar_path in task.data:
            with (
                fsspec.open(tar_path, mode="rb", **storage_options) as fobj,
                tarfile.open(fileobj=fobj, mode="r:*") as tf,
            ):
                members = [m for m in tf.getmembers() if m.isfile()]
                member_names = {m.name for m in members}
                read_ctx = _ReadContext(
                    tar_path=tar_path,
                    member_names=member_names,
                    member_info={m.name: m for m in members},
                    storage_options=storage_options,
                    byte_cache={},
                )
                for m in members:
                    if not m.name.endswith(self.json_extensions):
                        continue
                    rows.extend(self._rows_from_member(tf=tf, member=m, read_ctx=read_ctx))

        if rows:
            table = pa.Table.from_pylist(rows)
            table = table.cast(self._reconcile_schema(table.schema))
        else:
            table = pa.Table.from_pylist([], schema=self._empty_output_schema())
        splits = split_table_by_group_max_bytes(table, "sample_id", self.max_batch_bytes)
        batches: list[InterleavedBatch] = []
        for idx, split in enumerate(splits):
            task_id = f"{task.task_id}_processed" if len(splits) == 1 else f"{task.task_id}_processed_{idx:05d}"
            metadata = dict(task._metadata)
            if storage_options:
                metadata["source_storage_options"] = storage_options
            batches.append(
                InterleavedBatch(
                    task_id=task_id,
                    dataset_name=task.dataset_name,
                    data=split,
                    _metadata=metadata,
                    _stage_perf=task._stage_perf,
                )
            )
        return batches if len(batches) > 1 else batches[0]


# -- binary materialization (mirrors nemo_curator.stages.interleaved.utils.materialization) --


class _ClassifiedRows(NamedTuple):
    tar_extract: dict[str, list[tuple[int, str, int | None]]]
    range_read: dict[str, list[tuple[int, str, int, int, int | None]]]
    direct_read: dict[str, list[int]]
    missing: list[int]


def _get_frame_index(df: pd.DataFrame, idx: int) -> int | None:
    if "_src_frame_index" not in df.columns:
        return None
    val = df.loc[idx, "_src_frame_index"]
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    return int(val)


def _classify_rows(df: pd.DataFrame, image_mask: pd.Series) -> _ClassifiedRows:
    tar_extract: dict[str, list[tuple[int, str, int | None]]] = {}
    range_read: dict[str, list[tuple[int, str, int, int, int | None]]] = {}
    direct_read: dict[str, list[int]] = {}
    missing: list[int] = []

    for idx in df[image_mask].index:
        path = df.loc[idx, "_src_path"]
        if path is None or (isinstance(path, float) and pd.isna(path)) or path == "":
            missing.append(idx)
            continue

        path_str = str(path)
        raw_member = df.loc[idx, "_src_member"]
        has_member = raw_member not in (None, "") and pd.notna(raw_member)

        if not has_member:
            direct_read.setdefault(path_str, []).append(idx)
            continue

        member_str = str(raw_member)
        frame_idx = _get_frame_index(df, idx)
        raw_offset = df.loc[idx, "_src_byte_offset"]
        raw_size = df.loc[idx, "_src_byte_size"]
        has_range = raw_offset is not None and raw_size is not None and pd.notna(raw_offset) and pd.notna(raw_size)

        if has_range and int(raw_size) > 0:
            range_read.setdefault(path_str, []).append((idx, member_str, int(raw_offset), int(raw_size), frame_idx))
        else:
            tar_extract.setdefault(path_str, []).append((idx, member_str, frame_idx))

    return _ClassifiedRows(tar_extract=tar_extract, range_read=range_read, direct_read=direct_read, missing=missing)


def _fill_tar_extract_rows(
    groups: dict[str, list[tuple[int, str, int | None]]],
    storage_options: dict[str, object],
    binary_values: list[object],
    error_values: list[str | None],
) -> None:
    for path, keyed_rows in groups.items():
        key_cache: dict[str, bytes | None] = {}
        try:
            with fsspec.open(path, mode="rb", **storage_options) as fobj, tarfile.open(fileobj=fobj, mode="r:*") as tf:
                for idx, member, frame_idx in keyed_rows:
                    if member not in key_cache:
                        try:
                            extracted = tf.extractfile(member)
                        except KeyError:
                            extracted = None
                        key_cache[member] = extracted.read() if extracted is not None else None

                    payload = key_cache[member]
                    if payload is None:
                        error_values[idx] = f"missing member '{member}'"
                        continue

                    if frame_idx is not None:
                        payload = _extract_tiff_frame(payload, frame_idx)
                        if payload is None:
                            error_values[idx] = f"failed to extract frame {frame_idx} from '{member}'"
                            continue

                    binary_values[idx] = payload
                    error_values[idx] = None
        except (OSError, tarfile.TarError):
            for idx, *_ in keyed_rows:
                error_values[idx] = "failed to read path"


def _scatter_range_blobs(
    blobs: list[object],
    range_keys: list[tuple[int, int]],
    unique_ranges: dict[tuple[int, int], list[tuple[int, str, int | None]]],
    binary_values: list[object],
    error_values: list[str | None],
) -> None:
    for key, blob in zip(range_keys, blobs, strict=True):
        if isinstance(blob, Exception):
            for idx, member, _fi in unique_ranges[key]:
                error_values[idx] = f"range read error for member '{member}'"
        elif blob is None or len(blob) == 0:
            for idx, member, _fi in unique_ranges[key]:
                error_values[idx] = f"empty range read for member '{member}'"
        else:
            raw = bytes(blob) if not isinstance(blob, bytes) else blob
            for idx, member, frame_idx in unique_ranges[key]:
                payload = _extract_tiff_frame(raw, frame_idx) if frame_idx is not None else raw
                if payload is None:
                    error_values[idx] = f"failed to extract frame {frame_idx} from '{member}'"
                else:
                    binary_values[idx] = payload
                    error_values[idx] = None


def _fill_range_read_rows(
    groups: dict[str, list[tuple[int, str, int, int, int | None]]],
    storage_options: dict[str, object],
    binary_values: list[object],
    error_values: list[str | None],
) -> None:
    for path, entries in groups.items():
        try:
            fs, fs_path = url_to_fs(path, **storage_options)
        except (ValueError, OSError):
            for idx, *_ in entries:
                error_values[idx] = "failed to resolve filesystem"
            continue

        unique_ranges: dict[tuple[int, int], list[tuple[int, str, int | None]]] = {}
        for idx, member, offset, size, frame_idx in entries:
            unique_ranges.setdefault((offset, size), []).append((idx, member, frame_idx))

        range_keys = list(unique_ranges.keys())
        dedup_paths = [fs_path] * len(range_keys)
        starts = [offset for offset, _ in range_keys]
        ends = [offset + size for offset, size in range_keys]

        try:
            blobs = fs.cat_ranges(dedup_paths, starts, ends)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("cat_ranges failed for %s (%d ranges): %s", path, len(entries), exc)
            for idx, *_ in entries:
                error_values[idx] = "cat_ranges failed"
            continue

        _scatter_range_blobs(blobs, range_keys, unique_ranges, binary_values, error_values)


def _fill_direct_read_rows(
    groups: dict[str, list[int]],
    storage_options: dict[str, object],
    binary_values: list[object],
    error_values: list[str | None],
) -> None:
    for path, row_idxs in groups.items():
        payload = _read_direct_file(path, storage_options)
        for idx in row_idxs:
            if payload is not None:
                binary_values[idx] = payload
                error_values[idx] = None
            else:
                error_values[idx] = "failed to read path"


def _read_direct_file(path: str, storage_options: dict[str, object]) -> bytes | None:
    try:
        with fsspec.open(path, mode="rb", **storage_options) as fobj:
            return fobj.read()
    except (OSError, RuntimeError, ValueError):
        return None


def _fill_materialized_bytes(
    df: pd.DataFrame,
    image_mask: pd.Series,
    *,
    storage_options: dict[str, object],
    binary_values: list[object],
    error_values: list[str | None],
) -> None:
    classified = _classify_rows(df, image_mask)

    for idx in classified.missing:
        error_values[idx] = "missing path"

    _fill_tar_extract_rows(classified.tar_extract, storage_options, binary_values, error_values)
    _fill_range_read_rows(classified.range_read, storage_options, binary_values, error_values)
    _fill_direct_read_rows(classified.direct_read, storage_options, binary_values, error_values)


def _init_materialization_buffers(df: pd.DataFrame) -> tuple[list[object], list[str | None]]:
    error_values = (
        df["materialize_error"].astype("object").tolist() if "materialize_error" in df.columns else [None] * len(df)
    )
    binary_values = (
        df["binary_content"].astype("object").tolist() if "binary_content" in df.columns else [None] * len(df)
    )
    return binary_values, error_values


def _build_image_mask(
    df: pd.DataFrame,
    *,
    only_missing_binary: bool,
    image_content_types: tuple[str, ...] | None,
) -> pd.Series:
    image_mask = (
        (df["modality"] == "image") if "modality" in df.columns else pd.Series(False, index=df.index, dtype=bool)
    )
    if image_content_types is not None and "content_type" in df.columns:
        image_mask &= df["content_type"].isin(image_content_types)
    if only_missing_binary and "binary_content" in df.columns:
        image_mask &= df["binary_content"].isna()
    return image_mask


def _task_with_dataframe(task: InterleavedBatch, df: pd.DataFrame) -> InterleavedBatch:
    return InterleavedBatch(
        task_id=task.task_id,
        dataset_name=task.dataset_name,
        data=df,
        _metadata=task._metadata,
        _stage_perf=task._stage_perf,
    )


def materialize_task_binary_content(
    task: InterleavedBatch,
    *,
    io_kwargs: dict[str, object] | None = None,
    only_missing_binary: bool = True,
    image_content_types: tuple[str, ...] | None = None,
) -> InterleavedBatch:
    df = task.with_parsed_source_ref_columns(prefix="_src_").reset_index(drop=True)
    if df.empty:
        return task

    binary_values, error_values = _init_materialization_buffers(df)
    image_mask = _build_image_mask(
        df,
        only_missing_binary=only_missing_binary,
        image_content_types=image_content_types,
    )
    if not image_mask.any():
        out = df.drop(columns=[c for c in df.columns if c.startswith("_src_")], errors="ignore")
        return _task_with_dataframe(task, out)

    storage_options = resolve_storage_options(task=task, io_kwargs=io_kwargs)
    _fill_materialized_bytes(
        df,
        image_mask,
        storage_options=storage_options,
        binary_values=binary_values,
        error_values=error_values,
    )

    out = df.drop(columns=[c for c in df.columns if c.startswith("_src_")], errors="ignore")
    out["binary_content"] = pd.Series(binary_values, dtype="object")
    out["materialize_error"] = pd.Series(error_values, dtype="object")
    return _task_with_dataframe(task, out)
