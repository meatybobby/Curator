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

"""Pipeline that runs the same filters as mint1t_mvp_pipeline but saves only annotation
information (sample_id, original position) for kept rows. Position is the original index
so preview/restore can match rows to the original dataset without re-running the filter.
"""

import argparse
import json
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from fsspec.core import url_to_fs

import nemo_curator.stages.text.io.writer.utils as writer_utils
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.io import WebdatasetReader
from nemo_curator.stages.interleaved.stages import BaseInterleavedFilterStage
from nemo_curator.tasks import InterleavedBatch
from nemo_curator.utils.client_utils import is_remote_url
from nemo_curator.utils.file_utils import check_output_mode
from nemo_curator.stages.interleaved.filter import (
    InterleavedBlurFilterStage,
    InterleavedQRCodeFilterStage,
    InterleavedCLIPScoreFilterStage,
    InterleavedImageToTextRatioFilterStage,
)

ANNOTATION_METADATA_KEY = "annotation"


@dataclass
class InterleavedAnnotationFilterStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """Runs a single interleaved filter and attaches/updates annotation (sample_id, position)
    for kept content rows in task metadata. Add multiple stages in the pipeline for multiple filters.
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
        passed = df.loc[keep_mask & content, ["sample_id", "position"]].drop_duplicates()
        current = task._metadata.get(ANNOTATION_METADATA_KEY)
        if current is not None and not current.empty:
            annotation = passed.merge(current, on=["sample_id", "position"], how="inner")
        else:
            annotation = passed
        task._metadata[ANNOTATION_METADATA_KEY] = annotation if not annotation.empty else None
        return task


@dataclass
class InterleavedAnnotationParquetWriterStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """Writes annotation (sample_id, position) from task metadata to parquet. Pass-through stage.
    Expects annotation from InterleavedAnnotationFilterStage in task._metadata.
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
    read_kwargs = {}
    write_kwargs = {}
    if args.storage_options_json:
        storage_options = json.loads(args.storage_options_json)
        read_kwargs["storage_options"] = storage_options
        write_kwargs["storage_options"] = storage_options

    pipe = Pipeline(
        name="mint1t_annotation_multimodal",
        description="WebDataset MINT1T -> annotation (sample_id, original position) for kept rows only",
    )
    pipe.add_stage(
        WebdatasetReader(
            source_id_field="pdf_name",
            file_paths=args.input_path,
            files_per_partition=args.files_per_partition,
            blocksize=args.input_blocksize,
            max_batch_bytes=args.output_max_batch_bytes,
            read_kwargs=read_kwargs,
            materialize_on_read=args.materialize_on_read,
            fields=tuple(args.fields) if args.fields else None,
            per_image_fields=tuple(args.per_image_fields) if args.per_image_fields else (),
            per_text_fields=tuple(args.per_text_fields) if args.per_text_fields else (),
        )
    )
    # Add one annotation filter stage per filter; annotation is intersected across stages
    # pipe.add_stage(InterleavedAnnotationFilterStage(filter_stage=InterleavedBlurFilterStage()))
    # pipe.add_stage(InterleavedAnnotationFilterStage(filter_stage=InterleavedQRCodeFilterStage()))
    # pipe.add_stage(
    #     InterleavedAnnotationFilterStage(
    #         filter_stage=InterleavedCLIPScoreFilterStage(model_dir="./model", min_score=0.15)
    #     )
    # )
    pipe.add_stage(
        InterleavedAnnotationFilterStage(
            filter_stage=InterleavedImageToTextRatioFilterStage(
                min_ratio=0.001,
                max_ratio=2,
            )
        )
    )
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
        description="MINT1T multimodal pipeline: save only annotation (e.g. position id) of kept data"
    )
    parser.add_argument("--input-path", type=str, required=True, help="Input tar shard path or directory")
    parser.add_argument("--output-path", type=str, required=True, help="Output directory for annotation parquet")
    parser.add_argument("--files-per-partition", type=int, default=1)
    parser.add_argument("--input-blocksize", type=str, default=None)
    parser.add_argument("--output-max-batch-bytes", type=int, default=None)
    parser.add_argument("--materialize-on-read", action="store_true", dest="materialize_on_read")
    parser.add_argument("--no-materialize-on-read", action="store_false", dest="materialize_on_read")
    parser.set_defaults(materialize_on_read=False)
    parser.add_argument("--mode", type=str, default="ignore", choices=["ignore", "overwrite", "append", "error"])
    parser.add_argument("--fields", nargs="*", default=None)
    parser.add_argument("--per-image-fields", nargs="*", default=["image_metadata"])
    parser.add_argument("--per-text-fields", nargs="*", default=[])
    parser.add_argument(
        "--storage-options-json",
        type=str,
        default=None,
        help="JSON-encoded fsspec storage options for cloud paths",
    )
    main(parser.parse_args())
