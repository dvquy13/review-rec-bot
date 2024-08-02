import string
import re
import json
from pydantic import BaseModel


def substitute_punctuation(text):
    # Create a translation table that maps each punctuation character to an underscore
    translator = str.maketrans(string.punctuation, "_" * len(string.punctuation))
    # Translate the text using the translation table
    return text.translate(translator)


def flatten_dict(d, parent_key="", sep="__"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def cast_string_to_number(s):
    try:
        # Try to cast to integer
        num = int(s)
        return num
    except ValueError:
        try:
            # Try to cast to float
            num = float(s)
            return num
        except ValueError:
            # Return None if both conversions fail
            return None


def parse_collect_log(file_name, remove_punc: bool = True):
    collect_dict = {}
    pattern = re.compile(r"\[COLLECT\]\s*(.+?)=(.+?)\s")

    with open(file_name, "r") as log_file:
        for line in log_file:
            match = pattern.search(line)
            if match:
                key = match.group(1)
                value = match.group(2)
                if remove_punc:
                    key = substitute_punctuation(key)
                value = cast_string_to_number(value)
                collect_dict[key] = value

    return collect_dict


def pprint_pydantic_model(model: BaseModel):
    return json.dumps(model.model_dump(), indent=2)
