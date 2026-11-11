import ast
import difflib
import json
import os
import pickle
import re
import string
from pathlib import Path
from typing import List

CACHE_DIR = os.environ.get("CACHE_DIR", str(Path(__file__).resolve().parents[1] / ".cache" / "hf"))
os.environ.setdefault("HF_HOME", CACHE_DIR)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(CACHE_DIR, "datasets"))
os.environ.setdefault("HF_MODULES_CACHE", os.path.join(CACHE_DIR, "modules"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(CACHE_DIR, "transformers"))

_LETTERS = tuple("abcdef")


def _resolve_save_name(save_name: str | None = None, modeltype: str | None = None) -> str:
    resolved = save_name if save_name is not None else modeltype
    if resolved is None:
        raise TypeError("Expected save_name or modeltype.")
    return resolved


def _is_empty_file(p: Path) -> bool:
    return (not p.exists()) or (p.stat().st_size == 0) or (p.read_text(encoding="utf-8").strip() == "")


def make_json_safe(obj):
    """Recursively convert sets to lists while preserving nested container contents."""
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(make_json_safe(v) for v in obj)
    return obj


def write_to_file(
    save_dir: str | Path,
    save_name: str | None = None,
    result=None,
    print_log: bool = True,
    modeltype: str | None = None,
):
    """Write results as JSON, creating the destination directory if needed."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_name = _resolve_save_name(save_name=save_name, modeltype=modeltype)
    write_file = save_dir / f"{save_name}.json"

    if print_log:
        print(f"write to file {write_file}")

    with write_file.open("w") as f:
        json.dump(make_json_safe(result), f, indent=4)


def load_exist_file(
    save_dir: str | Path,
    save_name: str | None = None,
    print_log: bool = True,
    modeltype: str | None = None,
):
    """Read saved JSON results, returning None when the file does not exist."""
    save_dir = Path(save_dir)
    save_name = _resolve_save_name(save_name=save_name, modeltype=modeltype)
    load_file = save_dir / f"{save_name}.json"

    if load_file.exists():
        if print_log:
            print(f"loading file {load_file}")
        with load_file.open("r") as f:
            return json.load(f)
    return None


def load_existing_data(save_dir: str | Path, modeltype: str, print_log: bool = True):
    result = load_exist_file(save_dir=save_dir, modeltype=modeltype, print_log=False)
    if result is not None and print_log:
        print(f"Exist data: {Path(save_dir) / f'{modeltype}.json'}")
    return result


def write_to_pkl(save_dir: str | Path, save_name: str, result, print_log: bool = True):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    write_file = save_dir / f"{save_name}.pkl"

    if print_log:
        print(f"write to file {write_file}")

    with write_file.open("wb") as f:
        pickle.dump(result, f)


def load_exist_pkl(save_dir: str | Path, save_name: str, print_log: bool = True):
    load_file = Path(save_dir) / f"{save_name}.pkl"

    if load_file.exists():
        if print_log:
            print(f"loading file {load_file}")
        with load_file.open("rb") as f:
            return pickle.load(f)
    return None


def get_text_prompt_planning(options):
    prompt = (
        "Choose the single most helpful intermediate question from the options below.\n"
        "On the FIRST line, output ONLY the option letter in parentheses (e.g., (A)).\n"
        "That first line must match exactly: ^\\([A-F]\\)$\n"
        "After a blank line, you MAY add brief reasoning.\n"
    )
    for i, option in enumerate(options):
        prompt += f"{chr(65 + i)}. {option}\n"
    return prompt


def get_text_prompt(question, options):
    prompt = (
        f"{question}\n"
        "Choose the single best answer from the options below.\n"
        "On the FIRST line, output ONLY the option letter in parentheses (e.g., (A)).\n"
        "That first line must match exactly: ^\\([A-F]\\)$\n"
        "After a blank line, you MAY add brief reasoning. Do not repeat the question or options.\n"
    )
    for i, option in enumerate(options):
        prompt += f"({chr(65 + i)}) {option}\n"
    return prompt


def find_qa_pair(options: List, answer: List):
    letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}

    try:
        answer_letter = answer[0]
        answer_index = letter_to_index[answer_letter]
        return options[answer_index]
    except Exception:
        return None


def _normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = s.strip('"\'' + "".join("()[]{}"))
    s = s.strip(string.punctuation + " ")
    return re.sub(r"\s+", " ", s)


def unify_ans(answer: str, options=None) -> str:
    """Extract an a-f option label, falling back to normalized or fuzzy option text."""
    if not isinstance(answer, str):
        return ""

    raw = answer

    match = re.search(r"\(([a-fA-F])\)", raw)
    if match:
        return match.group(1).lower()

    match = re.search(r"(?m)^\s*([A-Fa-f])[.)]\s", raw)
    if match:
        return match.group(1).lower()

    ans_safe = raw.lower().strip()
    ans_safe = re.sub(r"\be\.g\.", "eg", ans_safe)
    ans_safe = re.sub(r"\bi\.e\.", "ie", ans_safe)

    for pattern in [r"(?<![a-z])([a-f])(?![a-z])", r"\banswer[:\s]*([a-f])\b"]:
        match = re.search(pattern, ans_safe)
        if match:
            return match.group(1)

    for pattern in [r"\banswer[:\s]*([1-6])\b", r"\b([1-6])\b"]:
        match = re.search(pattern, ans_safe)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(_LETTERS):
                return _LETTERS[idx]

    if options:
        norm_opts = [_normalize_text(str(o)) for o in options]
        cleaned = re.sub(r"^\s*(selected\s+question|question)\s*[:\-]\s*", "", raw, flags=re.IGNORECASE)
        candidate = _normalize_text(cleaned)

        if candidate in norm_opts:
            idx = norm_opts.index(candidate)
            if 0 <= idx < len(_LETTERS):
                return _LETTERS[idx]

        for i, opt in enumerate(norm_opts):
            if opt and (opt in candidate or candidate in opt):
                if 0 <= i < len(_LETTERS):
                    return _LETTERS[i]

        match = difflib.get_close_matches(candidate, norm_opts, n=1, cutoff=0.75)
        if match:
            idx = norm_opts.index(match[0])
            if 0 <= idx < len(_LETTERS):
                return _LETTERS[idx]

    return ""


def postprocess_output(output: str):
    """Parse fenced JSON or Python literals, retaining text when parsing fails."""
    try:
        output = (
            output.split("```json")[-1]
            .replace("```", "")
            .replace("\n", "")
            .replace(",]", "]")
            .replace(",}", "}")
            .strip()
        )
        try:
            return json.loads(output)
        except Exception:
            return ast.literal_eval(output)
    except Exception:
        return output
