from typing import Union
from pathlib import Path
import csv


def load_prompt_from_csv(file_dir: Union[Path, str], skip_header: bool = True):
    with open(file_dir, "r") as f:
        reader = csv.reader(f, delimiter=",")
        if skip_header:
            next(reader, None)
        data = list(reader)
    # print(data)
    data[:] = [entry[0] for entry in data]
    return data

