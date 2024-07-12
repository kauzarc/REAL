# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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

import json
import logging
import re
from typing import List, Generator, Tuple
from typing_extensions import TypedDict

import datasets
from datasets import (
    BuilderConfig,
    DatasetInfo,
    Features,
    Value,
    DownloadManager,
    SplitGenerator,
    GeneratorBasedBuilder,
)

_DESCRIPTION = """\
REBEL is a silver dataset created for the paper 
REBEL: Relation Extraction By End-to-end Language generation
"""


class Annotation(TypedDict):
    uri: str
    boundaries: List[int]
    surfaceform: str
    annotator: str


class AnnotationTriple(TypedDict):
    subject: Annotation
    predicate: Annotation
    object: Annotation


class Document(TypedDict):
    docid: str
    title: str
    uri: str
    text: str
    entities: List[Annotation]
    triples: List[AnnotationTriple]


class RebelConfig(BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Rebel(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        RebelConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self) -> DatasetInfo:
        return DatasetInfo(
            description=_DESCRIPTION,
            features=Features(
                {
                    "id": Value("string"),
                    "title": Value("string"),
                    "context": Value("string"),
                    "triplets": Value("string"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager: DownloadManager) -> List[SplitGenerator]:
        return [
            SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": self.config.data_files["train"]},
            ),
            SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": self.config.data_files["dev"]},
            ),
            SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": self.config.data_files["test"]},
            ),
        ]

    def _generate_examples(
        self, filepath: str
    ) -> Generator[Tuple[str, dict], None, None]:
        logging.info(f"generating examples from {filepath}")

        with open(self.config.data_files["tokens"][0], "r") as file:
            tokens = {line.replace("\n", "") for line in file}

        sentence_re = re.compile(r"\b.+?[.!?]+(?=\s|$)")

        with open(filepath[0], "r") as file:
            for line in file:
                document: Document = json.loads(line)

                if len(document["triples"]) == 0:
                    continue

                count = 0
                context = ""
                span = None
                for match in sentence_re.finditer(document["text"]):
                    if span is None:
                        span = match.span()
                    else:
                        span = (span[0], match.end())
                    start, end = span

                    context += " " + document["text"][match.start() : match.end()]

                    entities = sorted(
                        (
                            entity
                            for entity in document["entities"]
                            if start < entity["boundaries"][1] <= end
                            and entity["uri"] in tokens
                        ),
                        key=lambda tup: tup["boundaries"][0],
                    )

                    decoder_output = ""
                    for entity in entities:
                        triples = sorted(
                            (
                                triple
                                for triple in document["triples"]
                                if triple["subject"] == entity
                                and triple["predicate"]["uri"] in tokens
                                and triple["object"]["uri"] in tokens
                                and start < triple["subject"]["boundaries"][1] <= end
                                and start < triple["object"]["boundaries"][1] <= end
                            ),
                            key=lambda tup: tup["object"]["boundaries"][0],
                        )

                        if len(triples) == 0:
                            continue

                        decoder_output += (
                            f"<triplet> {entity['surfaceform']} <{entity['uri']}> "
                            + " ".join(
                                map(
                                    lambda triple: triple["object"]["surfaceform"]
                                    + " <"
                                    + triple["object"]["uri"]
                                    + "> <"
                                    + triple["predicate"]["uri"]
                                    + ">",
                                    triples,
                                )
                            )
                        )

                    if len(decoder_output) > 0:
                        id_ = document["docid"] + "-" + str(count)

                        yield id_, {
                            "title": document["title"],
                            "context": context,
                            "id": id_,
                            "triplets": decoder_output,
                        }

                        count += 1
                        context = ""
                        span = None
