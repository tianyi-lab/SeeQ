#!/usr/bin/env python3
"""Compute embedding diversity for qwen_qs_gene.py outputs."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def iter_questions(item: dict) -> list[str]:
    """Flatten generation groups into nonempty question strings, preserving order."""
    questions: list[str] = []
    for group in item.get("generations") or []:
        if isinstance(group, dict):
            group = [group]
        for row in group or []:
            if not isinstance(row, dict):
                continue
            question = str(row.get("gene_question") or "").strip()
            if question:
                questions.append(question)
    return questions


def cosine_distance_summary(embeddings: np.ndarray) -> dict[str, float]:
    """Summarize distinct embedding pairs; fewer than two items have zero diversity."""
    if len(embeddings) < 2:
        return {
            "mean_pairwise_cosine_distance": 0.0,
            "mean_pairwise_cosine_similarity": 1.0,
            "min_pairwise_cosine_distance": 0.0,
            "max_pairwise_cosine_distance": 0.0,
        }
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)
    sim = embeddings @ embeddings.T
    pairs = np.asarray([1.0 - sim[i, j] for i, j in combinations(range(len(embeddings)), 2)])
    return {
        "mean_pairwise_cosine_distance": float(np.mean(pairs)),
        "mean_pairwise_cosine_similarity": float(1.0 - np.mean(pairs)),
        "min_pairwise_cosine_distance": float(np.min(pairs)),
        "max_pairwise_cosine_distance": float(np.max(pairs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="qwen_qs_gene.py JSON output")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(input_path.read_text())
    rows = []
    all_questions = []
    question_offsets = []
    for item in data:
        questions = iter_questions(item)
        start = len(all_questions)
        all_questions.extend(questions)
        question_offsets.append((item, questions, start, len(all_questions)))

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.embedding_model, trust_remote_code=True)
    embeddings = np.asarray(
        model.encode(
            all_questions,
            batch_size=args.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )

    for item, questions, start, end in question_offsets:
        normalized = [q.lower().strip() for q in questions]
        unique_rate = len(set(normalized)) / len(normalized) if normalized else 0.0
        summary = cosine_distance_summary(embeddings[start:end])
        rows.append({
            "index": item.get("index"),
            "image_path": item.get("image_path"),
            "num_questions": len(questions),
            "unique_question_rate": unique_rate,
            **summary,
        })

    aggregate = {
        "input": str(input_path),
        "embedding_model": args.embedding_model,
        "num_images": len(rows),
        "num_questions": len(all_questions),
        "mean_embedding_diversity": float(np.mean([r["mean_pairwise_cosine_distance"] for r in rows])),
        "mean_unique_question_rate": float(np.mean([r["unique_question_rate"] for r in rows])),
        "mean_pairwise_cosine_similarity": float(np.mean([r["mean_pairwise_cosine_similarity"] for r in rows])),
    }

    stem = input_path.stem
    csv_path = out_dir / f"{stem}_qwen_embedding_diversity_per_image.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_dir / f"{stem}_qwen_embedding_diversity_summary.json"
    summary_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
