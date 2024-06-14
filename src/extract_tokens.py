from argparse import ArgumentParser
from typing_extensions import TypedDict
from typing import Set, List, Tuple, Dict, Callable
import json
from dataclasses import dataclass
from re import compile


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


def read_rebel_file(file_path: str) -> Tuple[Set[str], Set[str]]:
    entities = set()
    relations = set()

    entity_re = compile("Q[0-9]+")
    relation_re = compile("P[0-9]+")

    with open(file_path, "r") as file:
        for line in file:
            document: Document = json.loads(line)

            for triple in document["triples"]:
                entity_tuple = triple["subject"]["uri"], triple["object"]["uri"]
                for entity in entity_tuple:
                    if entity_re.fullmatch(entity):
                        entities.add(entity)

                relation = triple["predicate"]["uri"]
                if relation_re.fullmatch(relation):
                    relations.add(relation)

    return entities, relations


@dataclass
class DatasetInfo:
    data_path: str
    filenames: Tuple[str, str, str]
    tokens_file: str
    read_function: Callable[[str], Tuple[Set[str], Set[str]]]


DATASETS_INFO: Dict[str, DatasetInfo] = {
    "rebel": DatasetInfo(
        "../data/rebel_dataset",
        ("en_train.jsonl", "en_val.jsonl", "en_test.jsonl"),
        "tokens.txt",
        read_rebel_file,
    )
}


def extract_tokens(dataset: str) -> None:
    dataset_info = DATASETS_INFO[dataset]

    tokens = set()
    for filename in dataset_info.filenames:
        entities, relations = dataset_info.read_function(
            f"{dataset_info.data_path}/{filename}"
        )
        tokens.update(entities)
        tokens.update(relations)

    with open(f"{dataset_info.data_path}/{dataset_info.tokens_file}", "w") as file:
        file.writelines(map(lambda x: f"{x}\n", tokens))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()

    extract_tokens(args.dataset)


if __name__ == "__main__":
    main()
