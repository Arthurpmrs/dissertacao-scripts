import os
import re
import csv
from typing import Union


class NoModelError(Exception):
    """Raised when Model file does not exists."""


class ParamAnalysisMissingError(Exception):
    """Raised when a Param Analysis is missing."""


def check_model_path(path):
    """Check if model path is absolute and check if file exists."""

    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)

    if not os.path.exists(path):
        raise NoModelError("The model does not exist in the path provided.")

    return path


def cleanup_csv(csv_filepath):
    """Read EES arrays CSV, cleans it up and saves a new cleaned CSV. Also returns CSV content."""
    regex = re.compile(r"\[.*\]")
    with open(csv_filepath, 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        clean_data = []
        for i, line in enumerate(data):
            if not i:
                clean_line = []
                for item in line[0].split("\t"):
                    new_item = regex.sub("", item)
                    new_item = new_item.replace("'", "")
                    clean_line.append(new_item)
            else:
                clean_line = [item.replace(".", ",") for item in line]

            clean_data.append(clean_line)

    # Write new cleaned up CSV (Excel Ready)
    folderpath = os.path.dirname(csv_filepath)
    clean_csvfilepath = os.path.join(folderpath, "clean_arrays.csv")
    with open(clean_csvfilepath, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=";")
        for line in clean_data:
            csvwriter.writerow(line)

    return clean_data


def get_base_folder(model_path: str) -> str:
    model_filename = os.path.basename(model_path).split(".")[0]
    base_folder = os.path.join(os.path.dirname(model_path), model_filename)

    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    return base_folder


def add_folder(base_folder: str, *folders: str) -> str:
    new_folder = os.path.join(base_folder, *folders)

    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    return new_folder


def d_difference(d1: dict, d2: dict) -> Union[str, int, float]:
    for (_, v1), (_, v2) in zip(d1.items(), d2.items()):
        if v1 != v2:
            return v2


def main():
    """Experiment with utilites functions."""
    base_folder = os.getcwd()
    print(add_folder(base_folder, 'teste', 'test2'))


if __name__ == "__main__":
    main()
