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

"""OmniCorpus WebDataset -> same annotation filters as mint1t_annotation_pipeline.

Reads OmniCorpus-CC tar shards via :class:`OmniCorpusReaderStage`, optionally
runs :class:`OmniCorpusMaterializeStage` when ``blur``, ``qrcode``, or ``clip``
filters are selected (images live in pickled ``.images`` members, not plain tar
ranges). Writes per-row ``sample_id``, ``position``, ``keep_mask`` parquet
like :mod:`mint1t_annotation_pipeline`.

Requires the ``omni_corpus_annotation`` package (e.g. multi-modal-data-curation repo)
on ``PYTHONPATH`` or installed in the same environment as NeMo Curator.
"""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from fsspec.core import url_to_fs
from omni_corpus_annotation.stages.materialize import OmniCorpusMaterializeStage
from omni_corpus_annotation.stages.omnicorpus_reader import OmniCorpusReaderStage

import nemo_curator.stages.text.io.writer.utils as writer_utils
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.interleaved.filter import (
    InterleavedBlurFilterStage,
    InterleavedCLIPScoreFilterStage,
    InterleavedImageToTextRatioFilterStage,
    InterleavedQRCodeFilterStage,
)
from nemo_curator.stages.interleaved.utils import resolve_storage_options
from nemo_curator.tasks import InterleavedBatch
from nemo_curator.utils.client_utils import is_remote_url
from nemo_curator.utils.file_utils import check_output_mode

if TYPE_CHECKING:
    from nemo_curator.stages.interleaved.stages import BaseInterleavedFilterStage

ANNOTATION_METADATA_KEY = "annotation"

FILTER_CHOICES = ("blur", "qrcode", "clip", "ratio")
CLIP_MODEL_DIR = "./model_weights"

# Filters that need per-image bytes (OmniCorpus uses pickle members; see OmniCorpusMaterializeStage).
_IMAGE_BYTE_FILTERS = frozenset({"blur", "qrcode", "clip"})


def add_annotation_filters(pipe: Pipeline, args: argparse.Namespace) -> None:
    """Append InterleavedAnnotationFilterStage instances in the order given by --filters."""
    for name in args.filters:
        if name == "blur":
            score = args.score if args.score is not None else 100.0
            pipe.add_stage(
                InterleavedAnnotationFilterStage(
                    filter_stage=InterleavedBlurFilterStage(score_threshold=score),
                )
            )
        elif name == "qrcode":
            score = args.score if args.score is not None else 0.05
            pipe.add_stage(
                InterleavedAnnotationFilterStage(
                    filter_stage=InterleavedQRCodeFilterStage(score_threshold=score),
                )
            )
        elif name == "clip":
            score = args.score if args.score is not None else 0.15
            pipe.add_stage(
                InterleavedAnnotationFilterStage(
                    filter_stage=InterleavedCLIPScoreFilterStage(
                        model_dir=CLIP_MODEL_DIR,
                        min_score=score,
                    ),
                )
            )
        elif name == "ratio":
            max_ratio = args.max_ratio if args.max_ratio is not None else float("inf")
            pipe.add_stage(
                InterleavedAnnotationFilterStage(
                    filter_stage=InterleavedImageToTextRatioFilterStage(
                        min_ratio=args.min_ratio,
                        max_ratio=max_ratio,
                    ),
                )
            )


@dataclass
class InterleavedAnnotationFilterStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """Runs a single interleaved filter and updates annotation for all content rows.

    Each row has ``sample_id``, ``position``, and cumulative ``keep_mask`` (AND across
    filter stages so far). Task ``data`` is unchanged; only ``_metadata[annotation]`` is updated.
    Uses the same resources as the wrapped filter_stage.
    """

    filter_stage: BaseInterleavedFilterStage
    name: str = "interleaved_annotation_filter"

    def __post_init__(self) -> None:
        self.resources = self.filter_stage.resources

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: InterleavedBatch) -> InterleavedBatch:
        df = task.to_pandas()
        if df.empty:
            task._metadata[ANNOTATION_METADATA_KEY] = None
            return task
        keep_mask = self.filter_stage.keep_mask(task, df)
        content = (df["modality"] != "metadata") & (df["position"] >= 0)
        sub = df.loc[content]
        base = sub[["sample_id", "position"]].copy()
        base["keep_this"] = keep_mask.loc[sub.index].fillna(False).to_numpy(dtype=bool)

        current = task._metadata.get(ANNOTATION_METADATA_KEY)
        if current is not None and not current.empty:
            prev = current[["sample_id", "position"]].copy()
            if "keep_mask" in current.columns:
                prev["keep_mask"] = current["keep_mask"].fillna(False).astype(bool)
            else:
                prev["keep_mask"] = True
            merged = base.merge(prev, on=["sample_id", "position"], how="left")
            merged["keep_mask"] = merged["keep_this"] & merged["keep_mask"].fillna(False)
        else:
            merged = base
            merged["keep_mask"] = merged["keep_this"]

        annotation = merged[["sample_id", "position", "keep_mask"]].drop_duplicates(
            subset=["sample_id", "position"], keep="last"
        )
        task._metadata[ANNOTATION_METADATA_KEY] = annotation if not annotation.empty else None
        return task


@dataclass
class InterleavedAnnotationParquetWriterStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """Writes annotation ``sample_id``, ``position``, ``keep_mask`` from task metadata to parquet.

    Pass-through stage. Expects annotation from ``InterleavedAnnotationFilterStage``.
    """

    path: str
    write_kwargs: dict[str, Any] | None = None
    mode: Literal["ignore", "overwrite", "append", "error"] = "ignore"
    name: str = "interleaved_annotation_parquet_writer"

    def __post_init__(self) -> None:
        self.write_kwargs = self.write_kwargs or {}
        self.storage_options = self.write_kwargs.get("storage_options", {})
        self.fs, self._fs_path = url_to_fs(self.path, **self.storage_options)
        check_output_mode(self.mode, self.fs, self._fs_path, append_mode_implemented=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: InterleavedBatch) -> InterleavedBatch:
        to_write = task._metadata.get(ANNOTATION_METADATA_KEY)
        if to_write is None or to_write.empty:
            return task
        if source_files := task._metadata.get("source_files"):
            filename = writer_utils.get_deterministic_hash(source_files, task.task_id)
        else:
            filename = uuid.uuid4().hex
        file_path = self.fs.sep.join([self._fs_path, f"{filename}.parquet"])
        file_path_with_protocol = self.fs.unstrip_protocol(file_path) if is_remote_url(self.path) else file_path
        write_kwargs = dict(self.write_kwargs)
        write_kwargs.setdefault("compression", "snappy")
        write_kwargs.setdefault("row_group_size", 128_000)
        write_kwargs["index"] = False
        to_write.to_parquet(file_path_with_protocol, **write_kwargs)
        return task


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    read_kwargs: dict[str, Any] = {}
    write_kwargs: dict[str, Any] = {}
    if args.storage_options_json:
        storage_options = json.loads(args.storage_options_json)
        read_kwargs["storage_options"] = storage_options
        write_kwargs["storage_options"] = storage_options

    storage_opts = resolve_storage_options(io_kwargs=read_kwargs)

    pipe = Pipeline(
        name="omnicorpus_annotation_multimodal",
        description=("OmniCorpus WebDataset -> annotation (sample_id, position, keep_mask) for all content rows"),
    )
    pipe.add_stage(
        FilePartitioningStage(
            file_paths=args.input_path,
            files_per_partition=args.files_per_partition,
            blocksize=args.input_blocksize,
            file_extensions=[".tar"],
            storage_options=storage_opts,
        )
    )
    pipe.add_stage(
        OmniCorpusReaderStage(
            max_batch_bytes=args.output_max_batch_bytes,
            read_kwargs=read_kwargs,
            include_general_metadata=args.include_general_metadata,
        )
    )
    if _IMAGE_BYTE_FILTERS.intersection(args.filters):
        pipe.add_stage(OmniCorpusMaterializeStage())

    add_annotation_filters(pipe, args)
    pipe.add_stage(
        InterleavedAnnotationParquetWriterStage(
            path=args.output_path,
            write_kwargs=write_kwargs,
            mode=args.mode,
        )
    )
    return pipe


def main(args: argparse.Namespace) -> None:
    ray_client = RayClient()
    ray_client.start()
    pipeline = build_pipeline(args)
    print(pipeline.describe())
    pipeline.run()
    ray_client.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "OmniCorpus multimodal pipeline: save annotation (sample_id, position, keep_mask) for all content rows"
        )
    )
    parser.add_argument("--input-path", type=str, required=True, help="Input tar shard path or directory")
    parser.add_argument("--output-path", type=str, required=True, help="Output directory for annotation parquet")
    parser.add_argument("--files-per-partition", type=int, default=1)
    parser.add_argument("--input-blocksize", type=str, default=None)
    parser.add_argument("--output-max-batch-bytes", type=int, default=None)
    parser.add_argument(
        "--include-general-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Join url/safety fields from .general_metadata.pkl onto rows (OmniCorpus reader default).",
    )
    parser.add_argument("--mode", type=str, default="ignore", choices=["ignore", "overwrite", "append", "error"])
    parser.add_argument(
        "--storage-options-json",
        type=str,
        default=None,
        help="JSON-encoded fsspec storage options for cloud paths",
    )
    parser.add_argument(
        "--filters",
        nargs="+",
        choices=list(FILTER_CHOICES),
        default=["qrcode"],
        help="Interleaved filters to run (order matters). clip loads weights from ./model_weights.",
    )
    parser.add_argument(
        "--score",
        type=float,
        default=None,
        help=(
            "Threshold for blur (min sharpness), qrcode (max QR area ratio), and clip (min similarity). "
            "If omitted, each filter uses its stage default (blur 100, qrcode 0.05, clip 0.15)."
        ),
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.0,
        dest="min_ratio",
        help="Image-to-text ratio filter: min images-per-word for a sample",
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=None,
        dest="max_ratio",
        help="Image-to-text ratio filter: max images-per-word (omit for no upper bound)",
    )
    main(parser.parse_args())
