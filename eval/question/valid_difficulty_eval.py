"""Valid-difficulty evaluation pipeline for generated visual questions.

This script normalizes the mixed schemas in output/eval/question_gene, builds
image-capable OpenAI judge requests, collects judge outputs, and aggregates
model-level valid-difficulty metrics.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

sys.path.insert(0, os.getcwd())


DEFAULT_INPUT_DIR = Path("output/eval/question_gene")
DEFAULT_OUTPUT_DIR = Path("output/question_eval/valid_difficulty")
DEFAULT_MODELS = [
    "qwen25_3b",
    "qwen3_4b",
]
DIMENSION_KEYS = (
    "visual_search_difficulty",
    "visual_evidence_coverage",
    "visual_context_reasoning",
    "visual_spatial_reasoning",
)


def make_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is required for API submit/collect/demo calls.") from exc
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _load_batch_jsonl(openai_client, file_id: str) -> list[dict[str, Any]]:
    content = openai_client.files.content(file_id)
    return [json.loads(line) for line in content.text.splitlines() if line.strip()]


def _format_batch_errors(job) -> str:
    errors = getattr(job, "errors", None)
    data = getattr(errors, "data", None) if errors is not None else None
    if not data:
        return "No batch error details were returned."

    lines = []
    for err in data:
        parts = []
        code = getattr(err, "code", None)
        line = getattr(err, "line", None)
        param = getattr(err, "param", None)
        if code:
            parts.append(f"code={code}")
        if line is not None:
            parts.append(f"line={line}")
        if param:
            parts.append(f"param={param}")
        prefix = f" ({', '.join(parts)})" if parts else ""
        lines.append(f"-{prefix} {getattr(err, 'message', '')}")
    return "\n".join(lines)


def extract_raw_result(openai_client, batch_id: str) -> list[dict[str, Any]]:
    job = openai_client.batches.retrieve(batch_id)
    if job.status == "completed":
        if not job.output_file_id:
            print("OpenAI batch completed but did not return an output_file_id.")
            print(_format_batch_errors(job))
            raise SystemExit(1)
        print("\nBatch completed. Downloading results...")
        return _load_batch_jsonl(openai_client, job.output_file_id)

    if job.status in ["failed", "cancelled", "expired"]:
        print(f"OpenAI job failed with status: {job.status}. requests: {job.request_counts}")
        print(_format_batch_errors(job))
        if job.output_file_id:
            print("Downloading partial successful results from output_file_id...")
            return _load_batch_jsonl(openai_client, job.output_file_id)
        if getattr(job, "error_file_id", None):
            print("Downloading request error records from error_file_id...")
            return _load_batch_jsonl(openai_client, job.error_file_id)
        raise SystemExit(1)

    print(f"Job status: {job.status}. processing requests: {job.request_counts}")
    raise SystemExit(0)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).name)


def _clean_question(text: Any) -> str:
    return str(text or "").replace("\n###", "").replace("###", "").strip()


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _index_sort_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, int):
        return (0, value)
    try:
        return (0, int(str(value)))
    except (TypeError, ValueError):
        return (1, str(value))


def find_model_file(input_dir: Path, model_name: str) -> Path:
    stem = Path(model_name).name
    matches = [p for p in input_dir.glob("*.json") if p.stem == stem]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one JSON file for {model_name} in {input_dir}, found {len(matches)}")
    return matches[0]


def _reference_answer(item: dict[str, Any]) -> str:
    for key in ("gene_answer_ori", "gene_answer", "answer", "reference_answer"):
        value = item.get(key)
        if value:
            return str(value).strip()
    return ""


def _base_record(
    model_name: str,
    source_file: str,
    entry: dict[str, Any],
    entry_pos: int,
    question: str,
    answer: str = "",
    prompt_idx: int | None = None,
    gen_idx: int | None = None,
    prompt: str = "",
) -> dict[str, Any]:
    image_index = entry.get("index", entry.get("id", entry_pos))
    image_path = entry.get("image_path") or entry.get("image") or entry.get("img_path") or ""
    qid = f"{_safe_name(model_name)}:{image_index}:{prompt_idx if prompt_idx is not None else 'x'}:{gen_idx if gen_idx is not None else 'x'}"
    return {
        "model_name": model_name,
        "source_file": source_file,
        "image_index": image_index,
        "image_path": image_path,
        "question_id": qid,
        "question": question,
        "reference_answer": answer,
        "prompt_idx": prompt_idx,
        "gen_idx": gen_idx,
        "prompt": prompt,
        "missing_image": not bool(image_path) or not Path(image_path).exists(),
    }


def normalize_model(input_dir: Path, model_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = find_model_file(input_dir, model_name)
    data = _read_json(path)
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list, got {type(data).__name__}")
    for entry_pos, entry in enumerate(data):
        if not isinstance(entry, dict):
            warnings.append(f"skip non-dict entry at position {entry_pos}")
            continue
        if isinstance(entry.get("generations"), list):
            for prompt_idx, batch in enumerate(entry.get("generations") or []):
                if not isinstance(batch, list):
                    continue
                for gen_idx, gen in enumerate(batch):
                    if not isinstance(gen, dict):
                        continue
                    question = _clean_question(gen.get("gene_question") or gen.get("question"))
                    if not question:
                        continue
                    merged = {**entry, **gen}
                    records.append(
                        _base_record(
                            model_name,
                            path.name,
                            entry,
                            entry_pos,
                            question,
                            _reference_answer(merged),
                            prompt_idx,
                            gen_idx,
                            str(gen.get("prompt") or ""),
                        )
                    )
        elif entry.get("gene_question"):
            records.append(
                _base_record(model_name, path.name, entry, entry_pos, _clean_question(entry.get("gene_question")), _reference_answer(entry))
            )
        elif entry.get("question"):
            records.append(
                _base_record(model_name, path.name, entry, entry_pos, _clean_question(entry.get("question")), _reference_answer(entry))
            )
    image_paths = {r["image_path"] for r in records if r.get("image_path")}
    summary = {
        "model_name": model_name,
        "source_file": path.name,
        "entries": len(data),
        "questions": len(records),
        "unique_questions": len({r["question"] for r in records}),
        "unique_images": len(image_paths),
        "missing_image_records": sum(1 for r in records if r["missing_image"]),
        "warnings": warnings[:50],
    }
    return records, summary


def load_records(input_dir: Path, models: list[str], num_images: int | None = None, require_common_images: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    summaries = []
    for model in models:
        records, summary = normalize_model(input_dir, model)
        by_model[model] = records
        summaries.append(summary)

    selected_images: set[Any] | None = None
    if num_images is not None:
        image_sets = [{r["image_index"] for r in rows} for rows in by_model.values()]
        if require_common_images and image_sets:
            candidate = set.intersection(*image_sets)
        else:
            candidate = set.union(*image_sets) if image_sets else set()
        selected_images = set(sorted(candidate, key=_index_sort_key)[:num_images])

    all_rows = []
    for rows in by_model.values():
        for row in rows:
            if selected_images is not None and row["image_index"] not in selected_images:
                continue
            all_rows.append(row)
    return all_rows, summaries


def limit_records_per_model(records: list[dict[str, Any]], max_questions_per_model: int | None) -> list[dict[str, Any]]:
    if max_questions_per_model is None:
        return records
    limited = []
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        model = record["model_name"]
        if counts[model] < max_questions_per_model:
            limited.append(record)
            counts[model] += 1
    return limited


def judge_system_prompt() -> str:
    return """You are a strict JSON evaluator for deep visual question generation.
Evaluate whether a generated question is useful for testing deeper visual perception and reasoning over the image.
Prefer questions that require inspecting multiple objects, object regions, spatial relations, relative area/distance, fine details, occlusion, or multi-step visual inference.
Basic recognition, type/category, main object, primary color, and simple color-scheme questions should receive a high basic_question_penalty even if they are valid.
Subjective or functional questions may receive credit when they are visually plausible and tied to concrete visible evidence. Penalize only severe hallucination, hidden/future facts, nonsensical questions, or completely unsupported assumptions.
Return exactly one valid JSON object and no extra text.
Evaluate the four visual-question dimensions independently using only the attached image and current question.
Do not compare the question with other questions. Do not produce or rely on an overall score."""


def judge_user_prompt(record: dict[str, Any]) -> str:
    payload = {
        "question_id": record["question_id"],
        "question": record["question"],
        "reference_answer": record.get("reference_answer", ""),
        "prompt": record.get("prompt", ""),
    }
    return f"""Evaluate this generated visual question for the attached image.

Use only the image and the current question. Ignore model identity and do not compare with other questions. Score what the question requires to answer, not how long or fluent it is. Give credit to visually grounded conceptual or functional questions when the image contains concrete evidence; penalize generic world-knowledge answers, unsupported assumptions, and unanswerable questions.

Record:
{json.dumps(payload, indent=2, ensure_ascii=False)}

Score independently from 0 to 5:
- visual_search_difficulty: visual search/inspection effort. 0=no image needed; 1=obvious evidence; 3=specific region or nearby items; 5=multi-region search, counting, distance/area, or easy-to-miss fine detail.
- visual_evidence_coverage: breadth of concrete visual evidence. 0=no visible evidence; 1=one dominant object/scene; 3=two objects or one local region; 5=broad scene evidence, multiple important regions, or overall arrangement.
- visual_context_reasoning: conceptual interpretation beyond direct observation. 0=non-visual/unsupported; 1=direct identifying, reading, counting, locating, or comparing; 2=simple visible state/action/category judgment; 3=single-step interpretation from one cue; 4=multi-cue scene/situation interpretation; 5=non-obvious explanation or implication requiring weighed visual cues.
- visual_spatial_reasoning: use of layout/position/object relations. 0=none; 1=absolute location; 2=one simple relation; 3=one spatial comparison; 4=multiple linked relations; 5=complex spatial inference such as path, occlusion, containment, stability, or scene structure.

Return JSON only:
{{
  "visual_search_difficulty": int,
  "visual_evidence_coverage": int,
  "visual_context_reasoning": int,
  "visual_spatial_reasoning": int
}}"""


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_request(record: dict[str, Any], judge_model: str) -> dict[str, Any]:
    b64 = encode_image(record["image_path"])
    return {
        "custom_id": record["question_id"],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": judge_model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": judge_system_prompt()}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": judge_user_prompt(record)},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                },
            ],
        },
    }


def build_batch(records: list[dict[str, Any]], judge_model: str, batch_path: Path) -> list[dict[str, Any]]:
    usable = [r for r in records if not r.get("missing_image")]
    requests = [build_request(r, judge_model) for r in tqdm(usable, desc="Building judge requests")]
    _write_jsonl(batch_path, requests)
    return requests


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def score_judge(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Clamp dimension scores to 0-5 and normalize their mean; this scorer does not gate validity."""
    raw = raw if isinstance(raw, dict) else {}
    scored = dict(raw)
    dimensions = {key: _clamp(_as_float(raw.get(key), 0.0), 0.0, 5.0) for key in DIMENSION_KEYS}
    scored.update(
        {
            **dimensions,
            "raw_difficulty": _mean([dimensions[k] / 5.0 for k in DIMENSION_KEYS]),
            "valid_difficulty": _mean([dimensions[k] / 5.0 for k in DIMENSION_KEYS]),
            "invalid": False,
            "invalid_reason": "",
        }
    )
    return scored


def parse_batch_results(records: list[dict[str, Any]], raw_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join batch replies by question ID, retaining missing replies and skipping missing images."""
    by_id: dict[str, Any] = {}
    for line in raw_results:
        custom_id = line.get("custom_id")
        try:
            content = line["response"]["body"]["choices"][0]["message"]["content"]
            by_id[custom_id] = json.loads(content)
        except Exception:
            by_id[custom_id] = None
    judged = []
    for record in records:
        if record.get("missing_image"):
            continue
        api_res = by_id.get(record["question_id"])
        out = dict(record)
        out["api_res"] = api_res
        out["scores"] = score_judge(api_res)
        judged.append(out)
    return judged


def call_one(client: Any, record: dict[str, Any], judge_model: str) -> dict[str, Any]:
    request = build_request(record, judge_model)
    body = request["body"]
    response = client.chat.completions.create(**body)
    content = response.choices[0].message.content
    api_res = json.loads(content)
    out = dict(record)
    out["api_res"] = api_res
    out["scores"] = score_judge(api_res)
    return out


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[row["model_name"]].append(row)
    summary = []
    for model, items in sorted(by_model.items()):
        scores = [item.get("scores") or {} for item in items]
        questions = [item["question"] for item in items]
        image_indices = {item["image_index"] for item in items}
        exact_duplicate_rate = 1.0 - (len(set(questions)) / len(questions)) if questions else 0.0
        row = {
                "model": model,
                "num_questions": len(items),
                "num_images": len(image_indices),
                "image_coverage": len(image_indices) / len(items) if items else 0.0,
                "mean_valid_difficulty": round(_mean([_as_float(s.get("valid_difficulty")) for s in scores]), 4),
                "mean_raw_difficulty": round(_mean([_as_float(s.get("raw_difficulty")) for s in scores]), 4),
                "invalid_rate": round(_mean([1.0 if s.get("invalid") else 0.0 for s in scores]), 4),
                "exact_duplicate_rate": round(exact_duplicate_rate, 4),
            }
        if any(key in s for s in scores for key in DIMENSION_KEYS):
            row.update(
                {
                    key: round(_mean([_as_float(s.get(key)) / 5.0 for s in scores]), 4)
                    for key in DIMENSION_KEYS
                }
            )
        summary.append(row)
    return summary


def write_summary(summary: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not summary:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def _is_dimension_row(row: dict[str, Any]) -> bool:
    scores = row.get("scores") or {}
    return all(key in scores for key in DIMENSION_KEYS)


def _dimension_row(model: str, source_run: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if not (row.get("scores") or {}).get("invalid")]
    out: dict[str, Any] = {
        "model": model,
        "source_run": source_run,
        "num_questions": len(rows),
        "num_valid_questions": len(valid_rows),
        "valid_rate": round(len(valid_rows) / len(rows), 4) if rows else 0.0,
    }
    for key in DIMENSION_KEYS:
        out[key] = round(_mean([_as_float((row.get("scores") or {}).get(key)) / 5.0 for row in rows]), 4) if rows else ""
    return out


def write_dimension_means(rows: list[dict[str, Any]], source_run: str, path: Path) -> Path:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _is_dimension_row(row):
            by_model[row["model_name"]].append(row)
    out = [_dimension_row(model, source_run, by_model[model]) for model in sorted(by_model)]
    fieldnames = [
        "model",
        "source_run",
        "num_questions",
        "num_valid_questions",
        "valid_rate",
        *DIMENSION_KEYS,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out)
    return path


def load_sampled_by_image_records(sampled_json: Path, models: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = _read_json(sampled_json)
    if not isinstance(data, list):
        raise ValueError(f"{sampled_json} must contain a JSON list")
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"{sampled_json} row {row_index} must be an object")
        image_path = str(row.get("image_path") or "")
        out_row = {"image_path": image_path}
        rows.append(out_row)
        for model in models:
            candidates = [key for key in row if key.startswith(f"{model}:")]
            if len(candidates) != 1:
                raise ValueError(f"Expected exactly one question key for {model} in row {row_index}, found {len(candidates)}")
            question_id = candidates[0]
            question = _clean_question(row.get(question_id))
            if not question:
                raise ValueError(f"Empty question for {question_id} in row {row_index}")
            image_index = question_id.removeprefix(f"{model}:").split(":", 1)[0]
            records.append(
                {
                    "model_name": model,
                    "source_file": sampled_json.name,
                    "image_index": image_index,
                    "image_path": image_path,
                    "question_id": question_id,
                    "question": question,
                    "reference_answer": "",
                    "prompt_idx": None,
                    "gen_idx": None,
                    "prompt": "",
                    "missing_image": not bool(image_path) or not Path(image_path).exists(),
                    "sampled_image_row": row_index,
                    "sampled_question_key": question_id,
                }
            )
    return records, rows


def write_sampled_scores_by_image(
    sampled_rows: list[dict[str, Any]],
    judged: list[dict[str, Any]],
    models: list[str],
    path: Path,
) -> Path:
    by_row_model: dict[tuple[int, str], dict[str, Any]] = {}
    for row in judged:
        key = (int(row.get("sampled_image_row", -1)), row["model_name"])
        by_row_model[key] = row
    out = []
    score_keys = list(DIMENSION_KEYS)
    for row_index, sampled_row in enumerate(sampled_rows):
        out_row: dict[str, Any] = {"image_path": sampled_row.get("image_path", "")}
        for model in models:
            judged_row = by_row_model.get((row_index, model))
            if judged_row is None:
                raise ValueError(f"Missing judged row for sampled image row {row_index}, model {model}")
            scores = judged_row.get("scores") or {}
            out_row[model] = {
                "question_id": judged_row.get("question_id", ""),
                "question": judged_row.get("question", ""),
                "scores": {key: scores.get(key, "") for key in score_keys},
            }
        out.append(out_row)
    _write_json(path, out)
    return path


def command_normalize(args: argparse.Namespace) -> None:
    records, summaries = load_records(args.input_dir, args.models, args.num_images, args.common_images)
    records = limit_records_per_model(records, args.max_questions_per_model)
    run_name = args.run_name or "normalize"
    _write_jsonl(args.output_dir / "normalized" / f"{run_name}.jsonl", records)
    _write_json(args.output_dir / "normalized" / f"{run_name}_summary.json", summaries)
    print(f"Normalized {len(records)} questions from {len(args.models)} models.")
    for summary in summaries:
        print(f"{summary['model_name']}: questions={summary['questions']} images={summary['unique_images']} missing_images={summary['missing_image_records']}")


def command_build_batch(args: argparse.Namespace) -> Path:
    records, summaries = load_records(args.input_dir, args.models, args.num_images, args.common_images)
    records = limit_records_per_model(records, args.max_questions_per_model)
    run_name = args.run_name or "demo"
    batch_path = args.output_dir / "batches" / f"{run_name}_{args.judge_model}_batch.jsonl"
    build_batch(records, args.judge_model, batch_path)
    _write_jsonl(args.output_dir / "normalized" / f"{run_name}.jsonl", records)
    _write_json(args.output_dir / "normalized" / f"{run_name}_summary.json", summaries)
    print(f"Wrote batch JSONL to {batch_path}")
    return batch_path


def sampled_paths(output_dir: Path, run_name: str, judge_model: str = "gpt-5-mini") -> dict[str, Path]:
    return {
        "normalized": output_dir / "normalized" / f"{run_name}.jsonl",
        "source_rows": output_dir / "normalized" / f"{run_name}_source_rows.json",
        "summary": output_dir / "normalized" / f"{run_name}_summary.json",
        "batch": output_dir / "batches" / f"{run_name}_{judge_model}_batch.jsonl",
        "batch_id": output_dir / "batches" / f"{run_name}_batch_id.txt",
        "judged": output_dir / "results" / f"{run_name}_judged.jsonl",
        "dimension_means": output_dir / f"{run_name}_dimension_means.csv",
        "scores_by_image": output_dir / f"{run_name}_scores_by_image.json",
        "summary_csv": output_dir / f"{run_name}_summary.csv",
    }


def command_build_sampled_batch(args: argparse.Namespace) -> Path:
    paths = sampled_paths(args.output_dir, args.run_name, args.judge_model)
    records, sampled_rows = load_sampled_by_image_records(args.sampled_json, args.models)
    build_batch(records, args.judge_model, paths["batch"])
    _write_jsonl(paths["normalized"], records)
    _write_json(paths["source_rows"], sampled_rows)
    _write_json(
        paths["summary"],
        {
            "run_name": args.run_name,
            "sampled_json": str(args.sampled_json),
            "models": args.models,
            "num_image_rows": len(sampled_rows),
            "num_records": len(records),
            "num_missing_images": sum(1 for row in records if row.get("missing_image")),
        },
    )
    print(f"Wrote sampled batch JSONL to {paths['batch']}")
    return paths["batch"]


def command_submit_sampled(args: argparse.Namespace) -> None:
    paths = sampled_paths(args.output_dir, args.run_name, args.judge_model)
    batch_path = command_build_sampled_batch(args)
    client = make_openai_client()
    batch_file = client.files.create(file=batch_path.open("rb"), purpose="batch")
    job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"sampled-by-image {args.run_name}"},
    )
    paths["batch_id"].parent.mkdir(parents=True, exist_ok=True)
    paths["batch_id"].write_text(job.id, encoding="utf-8")
    print(f"Submitted sampled batch {job.id}; saved ID to {paths['batch_id']}")


def command_submit(args: argparse.Namespace) -> None:
    batch_path = command_build_batch(args)
    client = make_openai_client()
    batch_file = client.files.create(file=batch_path.open("rb"), purpose="batch")
    job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"valid-difficulty {args.run_name or 'demo'}"},
    )
    batch_id_path = args.output_dir / "batches" / f"{args.run_name or 'demo'}_batch_id.txt"
    batch_id_path.write_text(job.id, encoding="utf-8")
    print(f"Submitted batch {job.id}; saved ID to {batch_id_path}")


def command_status(args: argparse.Namespace) -> None:
    if not args.batch_id and not args.batch_id_file:
        raise SystemExit("Provide --batch-id or --batch-id-file.")
    batch_id = args.batch_id or Path(args.batch_id_file).read_text(encoding="utf-8").strip()
    client = make_openai_client()
    batch = client.batches.retrieve(batch_id)
    print(f"batch_id: {batch_id}")
    print(f"status: {batch.status}")
    print(f"request_counts: {batch.request_counts}")
    print(f"output_file_id: {batch.output_file_id}")
    print(f"error_file_id: {batch.error_file_id}")
    print(f"errors: {batch.errors}")


def command_collect(args: argparse.Namespace) -> None:
    batch_id = args.batch_id or Path(args.batch_id_file).read_text(encoding="utf-8").strip()
    normalized_path = Path(args.normalized_jsonl) if args.normalized_jsonl else args.output_dir / "normalized" / f"{args.run_name}.jsonl"
    records = _read_jsonl(normalized_path)
    client = make_openai_client()

    raw_results = extract_raw_result(client, batch_id)
    judged = parse_batch_results(records, raw_results)
    result_path = args.output_dir / "results" / f"{args.run_name}_judged.jsonl"
    _write_jsonl(result_path, judged)
    summary = aggregate_rows(judged)
    write_summary(summary, args.output_dir / f"{args.run_name}_summary.csv")
    means_path = args.output_dir / f"{args.run_name}_dimension_means.csv"
    write_dimension_means(judged, args.run_name, means_path)
    print(f"Saved dimension means to {means_path}")
    print(f"Saved judged records to {result_path}")
    print(json.dumps(summary, indent=2))


def command_collect_sampled(args: argparse.Namespace) -> None:
    paths = sampled_paths(args.output_dir, args.run_name, args.judge_model)
    batch_id_path = Path(args.batch_id_file) if args.batch_id_file else paths["batch_id"]
    batch_id = args.batch_id or batch_id_path.read_text(encoding="utf-8").strip()
    records = _read_jsonl(paths["normalized"])
    sampled_rows = _read_json(paths["source_rows"])
    client = make_openai_client()

    raw_results = extract_raw_result(client, batch_id)
    judged = parse_batch_results(records, raw_results)
    _write_jsonl(paths["judged"], judged)
    summary = aggregate_rows(judged)
    write_summary(summary, paths["summary_csv"])
    write_dimension_means(judged, args.run_name, paths["dimension_means"])
    write_sampled_scores_by_image(sampled_rows, judged, args.models, paths["scores_by_image"])
    print(f"Saved sampled judged records to {paths['judged']}")
    print(f"Saved sampled dimension means to {paths['dimension_means']}")
    print(f"Saved sampled scores by image to {paths['scores_by_image']}")
    print(json.dumps(summary, indent=2))


def command_aggregate(args: argparse.Namespace) -> None:
    paths = [Path(p) for p in args.inputs] if args.inputs else sorted((args.output_dir / "results").glob("*_judged.jsonl"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_read_jsonl(path))
    summary = aggregate_rows(rows)
    write_summary(summary, args.output_dir / "summary.csv")
    print(f"Aggregated {len(rows)} judged records from {len(paths)} files.")
    print(json.dumps(summary, indent=2))


def command_demo(args: argparse.Namespace) -> None:
    args.common_images = True
    records, summaries = load_records(args.input_dir, args.models, args.num_images, True)
    if args.max_questions_per_model is not None:
        limited = []
        counts: dict[str, int] = defaultdict(int)
        for record in records:
            model = record["model_name"]
            if counts[model] < args.max_questions_per_model:
                limited.append(record)
                counts[model] += 1
        records = limited
    run_name = args.run_name or "demo"
    _write_jsonl(args.output_dir / "normalized" / f"{run_name}.jsonl", records)
    _write_json(args.output_dir / "normalized" / f"{run_name}_summary.json", summaries)
    print(f"Demo normalized {len(records)} questions.")
    if args.dry_run:
        for record in records[: args.preview]:
            print(json.dumps({k: record[k] for k in ("model_name", "image_index", "question_id", "question", "image_path", "missing_image")}, indent=2))
        batch_path = args.output_dir / "batches" / f"{run_name}_{args.judge_model}_batch.jsonl"
        build_batch(records[: args.preview], args.judge_model, batch_path)
        print(f"Dry-run batch preview written to {batch_path}")
        return
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set; skipping API demo.")
        return
    usable = [r for r in records if not r.get("missing_image")][: args.max_api_questions]
    if not args.sync:
        _write_jsonl(args.output_dir / "normalized" / f"{run_name}.jsonl", usable)
        batch_path = args.output_dir / "batches" / f"{run_name}_{args.judge_model}_batch.jsonl"
        build_batch(usable, args.judge_model, batch_path)
        client = make_openai_client()
        batch_file = client.files.create(file=batch_path.open("rb"), purpose="batch")
        job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"valid-difficulty {run_name}"},
        )
        batch_id_path = args.output_dir / "batches" / f"{run_name}_batch_id.txt"
        batch_id_path.write_text(job.id, encoding="utf-8")
        print(f"Submitted batch {job.id}; saved ID to {batch_id_path}")
        print(
            "Collect with: "
            f"python eval/question/valid_difficulty_eval.py collect "
            f"--run-name {run_name} "
            f"--batch-id-file {batch_id_path} "
            f"--models {' '.join(args.models)}"
        )
        return

    client = make_openai_client()
    judged = []
    for record in tqdm(usable, desc="Running synchronous API demo"):
        try:
            judged.append(call_one(client, record, args.judge_model))
        except Exception as exc:
            failed = dict(record)
            failed["api_res"] = None
            failed["scores"] = score_judge(None)
            failed["error"] = str(exc)
            judged.append(failed)
    result_path = args.output_dir / "results" / f"{run_name}_judged.jsonl"
    _write_jsonl(result_path, judged)
    summary = aggregate_rows(judged)
    write_summary(summary, args.output_dir / f"{run_name}_summary.csv")
    means_path = args.output_dir / f"{run_name}_dimension_means.csv"
    write_dimension_means(judged, run_name, means_path)
    print(f"Saved dimension means to {means_path}")
    print(f"Saved demo judged records to {result_path}")
    print(json.dumps(summary, indent=2))


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--run-name", default="demo")
    parser.add_argument("--judge-model", default="gpt-5-mini")
    parser.add_argument("--common-images", action="store_true", help="Only keep image indices present in every selected model.")
    parser.add_argument("--max-questions-per-model", type=int, default=None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Valid-difficulty evaluation for VLM question generation.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("normalize")
    add_common_args(p)
    p.set_defaults(func=command_normalize)

    p = sub.add_parser("build-batch")
    add_common_args(p)
    p.set_defaults(func=command_build_batch)

    p = sub.add_parser("submit")
    add_common_args(p)
    p.set_defaults(func=command_submit)

    p = sub.add_parser("build-sampled-batch")
    add_common_args(p)
    p.add_argument("--sampled-json", type=Path, required=True)
    p.set_defaults(func=command_build_sampled_batch)

    p = sub.add_parser("submit-sampled")
    add_common_args(p)
    p.add_argument("--sampled-json", type=Path, required=True)
    p.set_defaults(func=command_submit_sampled)

    p = sub.add_parser("status")
    p.add_argument("--batch-id", default="")
    p.add_argument("--batch-id-file", default="")
    p.set_defaults(func=command_status)

    p = sub.add_parser("collect")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--run-name", default="demo")
    p.add_argument("--batch-id", default="")
    p.add_argument("--batch-id-file", default="")
    p.add_argument("--normalized-jsonl", default="")
    p.add_argument("--models", nargs="*", default=[])
    p.set_defaults(func=command_collect)

    p = sub.add_parser("collect-sampled")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--run-name", default="demo")
    p.add_argument("--judge-model", default="gpt-5-mini")
    p.add_argument("--batch-id", default="")
    p.add_argument("--batch-id-file", default="")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.set_defaults(func=command_collect_sampled)

    p = sub.add_parser("aggregate")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--inputs", nargs="*", default=[])
    p.set_defaults(func=command_aggregate)

    p = sub.add_parser("demo")
    add_common_args(p)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--preview", type=int, default=6)
    p.add_argument("--max-api-questions", type=int, default=12)
    p.add_argument("--sync", action="store_true", help="Run immediate synchronous calls instead of submitting an OpenAI Batch job.")
    p.set_defaults(func=command_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
