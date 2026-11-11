"""
RAG utilities for hard question generation.

Provides:
  - Sample bank management: load, append, novelty-filter, weighted sample
  - Image fact extraction (cached)
  - Difficulty score extraction from self_judge results
  - RAG-augmented prompt builders
"""

import json
import os
import re
import random
import hashlib
from typing import Dict, List, Optional, Tuple

from .analysis_utils import (
    _get_model_following_by_constraint,
    _merge_api_res_from_evaluation,
    _extract_scores_and_mfc,
)

# ── Retrieval prompt templates ────────────────────────────────────────────────
# Use direct placeholder replacement instead of .format(), because the prompt
# contains literal JSON braces.

broader_question_prompts = """You are a Question generator. Your objective is to rewrite a given Q&A into a genuinely harder version that requires more visual evidence or reasoning steps. Make it harder by increasing the DEPTH of reasoning required. Examples include evidence selection, object-role comparison, spatial relation reasoning, scene-function inference, and multi-object grounding. Do NOT make it harder by simply appending more sub-questions or extra clauses. Do NOT generate a question that asks the same thing as the original or is just a paraphrase. Do NOT start your rewritten question with the same first 5 words as the seed question. The question should not become broader, more subjective, or more generic. The added constraint must change the type or depth of reasoning required, not just expand the question's length.

Use the extracted visual facts below as grounding evidence. Only use facts that are clearly supported by the image.

Extracted visual facts:
{rag_facts}

Question: {seed_question}
Answer: {seed_answer}

Constraints:
- Require at least 2 concrete visible objects, regions, or relations.
- The question must contain a hidden intermediate inference.
- The intermediate inference step must not be given away in the question.
- Do NOT ask for subjective preference or broad commonsense.
- Do NOT merely paraphrase the seed question with synonyms.
- Ensure all generated data is consistent with the image content.
- The question must rely on visual information such that its answer would change or be impossible to determine without the image.
- Generate only ONE focused question, not multiple sub-questions.
- Make the SINGLE question harder by requiring deeper reasoning: multi-hop inference, counterfactual/causal reasoning, cross-region comparison, or quantitative/ordinal judgment.
- 'reasoning_steps' MUST contain at least 2 non-trivial, concrete operations, not just 'step 1' or 'step 2'.

Return the result in JSON format:
{'Rewritten question': 'one single syntactic question ending with ?', 'visible_evidence': ['at least two concrete visible objects/regions/relations required to answer, using the EXACT object names that appear in your question'], 'intermediate_inference': 'the hidden bridging inference required before the final answer', 'reasoning_steps': ['<concrete step, e.g. Locate the red chair>', '<concrete step, e.g. Compare its distance to the desk lamp vs the blue chair>'], 'why_harder_than_seed': 'why this requires more grounded reasoning than the seed',}"""

harder_question_prompts = """You are a Question generator. Your objective is to rewrite a given Q&A into a genuinely harder version that requires more visual evidence or reasoning steps. Make it harder by increasing the DEPTH of reasoning required. Examples include evidence selection, object-role comparison, spatial relation reasoning, scene-function inference, and multi-object grounding. Do NOT make it harder by simply appending more sub-questions or extra clauses. Do NOT generate a question that asks the same thing as the original or is just a paraphrase. Do NOT start your rewritten question with the same first 5 words as the seed question. The question should not become broader, more subjective, or more generic. The added constraint must change the type or depth of reasoning required, not just expand the question's length.

The following questions were generated for similar images and are rated as HARD:
{examples_text}

Rewrite the following Q&A so it is harder than the examples while staying tightly grounded in this image. It must require more reasoning steps, finer perceptual discrimination, or a deeper multi-hop chain.

Question: {seed_question}
Answer: {seed_answer}

Examples of GOOD rewritten questions:
Example 1:
Q: 'Which chair, the wooden chair or the metal chair, is positioned closer to the desk lamp?'
visible_evidence: ['wooden chair', 'metal chair', 'desk lamp']
reasoning_steps: ['Locate the wooden chair and measure its distance to the desk lamp', 'Locate the metal chair and measure its distance to the desk lamp', 'Compare the two distances']

Constraints:
- Do not reuse the same reasoning pattern as any example above.
- The added difficulty must come from deeper reasoning, not longer question text.
- Require at least 2 concrete visible objects, regions, or relations.
- The question must contain a hidden intermediate inference.
- Do NOT merely paraphrase the seed question with synonyms.
- Ensure all generated data is consistent with the image content.
- The question must rely on visual information such that its answer would change or be impossible to determine without the image.
- Generate only ONE focused question, not multiple sub-questions.
- 'reasoning_steps' MUST contain at least 2 non-trivial, concrete operations, not just 'step 1' or 'step 2'.

Return the result in JSON format:
{'Rewritten question': 'one single syntactic question ending with ?', 'visible_evidence': ['at least two concrete visible objects/regions/relations required to answer, using the EXACT object names that appear in your question'], 'intermediate_inference': 'the hidden bridging inference required before the final answer', 'reasoning_steps': ['<concrete step, e.g. Locate the red chair>', '<concrete step, e.g. Compare its distance to the desk lamp vs the blue chair>'], 'why_harder_than_seed': 'why this requires more grounded reasoning than the seed', 'final_answer_type': 'short phrase|entity|comparison|count|yes_no', }."""

_GENERIC_BANNED_PATTERNS = [
    (r"\bwhat happens if\b", "what_happens_if"),
    (r"\bwhat happens\b", "what_happens"),
    (r"\bwhat impact\b", "what_impact"),
    (r"\bwhat effect\b", "what_effect"),
    (r"\bwhich (?:is|one is|one|item|object|tool)\s+more important\b", "more_important"),
    (r"\blooks?\s+(?:nice|better)\b", "looks_nice"),
    # "how does X affect/impact/influence Y" — image-decoupled causal template
    (r"\bhow (?:does|do|would|might)\b.{0,60}\b(?:affect|impact|influence)\b", "how_does_affect"),
    (r"\bwhat (?:is the|are the|would be the)\s+(?:effect|impact|influence|role|significance)\s+of\b", "what_effect_of"),
    (r"\bhow (?:does|do|would|might)\b.{0,60}\b(?:change|alter|modify)\b.{0,40}\b(?:room|space|area|appearance|look)\b", "how_changes_appearance"),
]

_BROAD_SCENE_TERMS = {
    "mood",
    "atmosphere",
    "comfort",
    "usefulness",
    "functionality",
    "functional",
    "ambiance",
    "aesthetic",
    "aesthetics",
    "looks",
    "feel",
    "nicer",
    "nice",
    "influence",
    "role",
    "significance",
}

_COMPARISON_CUES = {
    "than", "versus", "vs", "compare", "comparison", "between",
    "better", "worse", "more", "less", "larger", "smaller",
}

_RELATION_CUES = {
    "left", "right", "above", "below", "near", "between", "behind",
    "in front", "closer", "further", "adjacent", "beside", "relative",
}

_NON_VISUAL_TERMS = {
    "productivity", "ambiance", "mood", "comfort", "usefulness",
    "importance", "significance", "beauty", "style", "meaning",
}


def _tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return re.findall(r"[a-z0-9']+", text.lower())


def normalize_question(text: str) -> str:
    """Normalize case and whitespace for sample-bank deduplication."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().lower().split())


def lexical_overlap_ratio(text_a: str, text_b: str) -> float:
    """Compute Jaccard overlap over distinct tokens longer than two characters."""
    toks_a = {tok for tok in _tokenize(text_a) if len(tok) > 2}
    toks_b = {tok for tok in _tokenize(text_b) if len(tok) > 2}
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / len(toks_a | toks_b)


def is_syntactic_question(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False
    if not text.endswith("?"):
        return False
    return bool(re.match(r"^(what|which|why|how|where|when|who|whom|whose|is|are|do|does|did|can|could|would|should)\b", text, flags=re.IGNORECASE))


def detect_generic_pattern_flags(question: str) -> List[str]:
    text = normalize_question(question)
    flags = [label for pattern, label in _GENERIC_BANNED_PATTERNS if re.search(pattern, text)]
    broad_hits = sorted({term for term in _BROAD_SCENE_TERMS if term in text})
    if broad_hits:
        flags.append("broad_scene_term")
    if any(term in text for term in _NON_VISUAL_TERMS):
        flags.append("non_visual_abstraction")
    return list(dict.fromkeys(flags))


def _as_list_of_strings(value) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _flatten_evidence_tokens(items: List[str]) -> List[str]:
    tokens: List[str] = []
    for item in items:
        tokens.extend(tok for tok in _tokenize(item) if len(tok) > 2)
    return tokens


def compute_specificity_score(
    question: str,
    *,
    visible_evidence: Optional[List[str]] = None,
    reasoning_steps: Optional[List[str]] = None,
) -> float:
    text = normalize_question(question)
    evidence = _as_list_of_strings(visible_evidence or [])
    steps = _as_list_of_strings(reasoning_steps or [])
    score = 0.0

    distinct_evidence = {ev.lower() for ev in evidence}
    score += min(len(distinct_evidence), 4) * 1.25
    score += min(len(steps), 4) * 0.75

    if any(cue in text for cue in _COMPARISON_CUES):
        score += 1.0
    if any(cue in text for cue in _RELATION_CUES):
        score += 0.75

    flags = detect_generic_pattern_flags(question)
    score -= 1.0 * len(flags)

    unique_tokens = {tok for tok in _flatten_evidence_tokens(evidence) if tok not in _BROAD_SCENE_TERMS}
    score += min(len(unique_tokens), 3) * 0.5

    return round(score, 3)


def validate_rag_question(
    *,
    question: str,
    seed_question: str,
    content: Optional[dict] = None,
    require_structured: bool = False,
) -> Dict[str, object]:
    content = content or {}
    visible_evidence = _as_list_of_strings(content.get("visible_evidence"))
    reasoning_steps = _as_list_of_strings(content.get("reasoning_steps"))
    generic_flags = detect_generic_pattern_flags(question)
    overlap = lexical_overlap_ratio(question, seed_question)
    specificity = compute_specificity_score(
        question,
        visible_evidence=visible_evidence,
        reasoning_steps=reasoning_steps,
    )

    result = {
        "valid": True,
        "failure_reason": "",
        "generic_pattern_flags": generic_flags,
        "visible_evidence_count": len({item.lower() for item in visible_evidence}),
        "reasoning_steps_count": len(reasoning_steps),
        "seed_overlap": round(overlap, 3),
        "specificity_score": specificity,
    }

    if not is_syntactic_question(question):
        result["valid"] = False
        result["failure_reason"] = "not_syntactic_question"
        return result

    if require_structured:
        if len({item.lower() for item in visible_evidence}) < 2:
            result["valid"] = False
            result["failure_reason"] = "insufficient_visible_evidence"
            return result
        if len(reasoning_steps) < 2:
            result["valid"] = False
            result["failure_reason"] = "insufficient_reasoning_steps"
            return result
        if not str(content.get("intermediate_inference", "")).strip():
            result["valid"] = False
            result["failure_reason"] = "missing_intermediate_inference"
            return result

    if overlap >= 0.82:
        result["valid"] = False
        result["failure_reason"] = "too_similar_to_seed"
        return result

    if generic_flags:
        result["valid"] = False
        result["failure_reason"] = "generic_shallow_pattern"
        return result

    if specificity < 2.0:
        result["valid"] = False
        result["failure_reason"] = "low_specificity"
        return result

    return result


# ── Difficulty score extraction ───────────────────────────────────────────────

def extract_difficulty_scores(dict_res: dict) -> Dict[str, float]:
    """Extract reasoning_difficulty and perception_difficulty_image from self_judge result."""
    api = _merge_api_res_from_evaluation(dict_res.get("evaluation", []))
    scores, _, _ = _extract_scores_and_mfc({"api_res": api})
    soft = api.get("soft_constraints", {}) or {}
    mfc = soft.get("model_following_capability", {}) or {}
    valid_constraints = [
        item for item in _get_model_following_by_constraint(mfc)
        if isinstance(item, dict) and str(item.get("constraint_text", "")).strip()
    ]
    if valid_constraints:
        model_following = sum(1 for item in valid_constraints if item.get("complies")) / len(valid_constraints)
    else:
        model_following = 0.0
    return {
        "reasoning_difficulty": float(scores.get("reasoning_difficulty", 0.0)),
        "perception_difficulty_image": float(scores.get("perception_difficulty_image", 0.0)),
        "model_following_capability": float(model_following),
    }


# ── Concept tag extraction ────────────────────────────────────────────────────

_TASK_KEYWORDS = {
    "compare", "comparison", "difference", "similar", "same",
    "spatial", "position", "location", "above", "below", "left", "right",
    "count", "many", "number", "quantity",
    "cause", "why", "because", "effect",
    "infer", "inference", "chain",
    "color", "shape", "texture", "size",
    "action", "doing", "activity",
    "relation", "relationship",
}

_STOP_WORDS = {
    "image", "picture", "photo", "question", "answer",
    "thing", "item", "object", "one", "two", "three",
    "this", "that", "these", "those", "with", "from",
}


def extract_concept_tags(question: str) -> List[str]:
    """
    Extract coarse concept tags from question text (no external NLP deps).
    Combines task-type keywords present in question and simple noun extraction.
    """
    text = question.lower()
    tags = [kw for kw in _TASK_KEYWORDS if kw in text]
    # nouns after articles
    nouns = re.findall(r'\b(?:the|a|an)\s+([a-z][a-z\-]+)', text)
    nouns = [n for n in nouns if n not in _STOP_WORDS and len(n) > 3]
    tags.extend(nouns[:10])
    return list(set(tags))


# ── Sample bank management ────────────────────────────────────────────────────

def load_sample_bank(path: str) -> List[dict]:
    """Load sample bank from JSONL file. Returns empty list if file missing."""
    if not path or not os.path.exists(path):
        return []
    bank = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    bank.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return curate_sample_bank(bank)


def append_to_bank_file(entry: dict, path: str):
    """Append a single entry to the bank JSONL file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def curate_sample_bank(bank: List[dict]) -> List[dict]:
    """Remove weak or generic entries from the bank before sampling."""
    curated = []
    seen_questions = set()
    for entry in bank:
        question = entry.get("question", "")
        norm_q = normalize_question(question)
        if not norm_q or norm_q in seen_questions:
            continue
        if not is_high_quality_bank_entry(entry):
            continue
        seen_questions.add(norm_q)
        curated.append(entry)
    return curated


def score_bank_entry(entry: dict, seed_question: str = "") -> float:
    """Weight difficulty, compliance, and specificity, penalizing overlap with the seed."""
    question = entry.get("question", "")
    scores = {
        "reasoning": float(entry.get("reasoning_difficulty", 0.0)),
        "perception": float(entry.get("perception_difficulty_image", 0.0)),
        "mfc": float(entry.get("model_following_capability", 0.0)),
        "specificity": float(entry.get("specificity_score", compute_specificity_score(question))),
    }
    overlap_penalty = lexical_overlap_ratio(seed_question, question) if seed_question else 0.0
    return max(
        0.1,
        2.0 * scores["reasoning"]
        + 1.25 * scores["perception"]
        + 2.0 * scores["mfc"]
        + 1.5 * scores["specificity"]
        - 4.0 * overlap_penalty,
    )


def sample_from_bank(bank: List[dict], k: int = 5, seed_question: str = "") -> List[dict]:
    """Sample k curated entries weighted by quality, compliance, specificity, and novelty."""
    if not bank:
        return []
    bank = curate_sample_bank(bank)
    if not bank:
        return []
    k = min(k, len(bank))
    weights = [score_bank_entry(entry=e, seed_question=seed_question) for e in bank]
    total = sum(weights)
    probs = [w / total for w in weights]
    indices = random.choices(range(len(bank)), weights=probs, k=k)
    return [bank[i] for i in indices]


def is_novel(concept_tags: List[str], bank: List[dict], overlap_threshold: float = 0.6) -> bool:
    """
    Return True if concept_tags are sufficiently novel vs all bank entries.
    Novel = no existing entry has Jaccard overlap >= overlap_threshold.
    """
    if not bank or not concept_tags:
        return True
    tag_set = set(concept_tags)
    for entry in bank:
        existing = set(entry.get("concept_tags", []))
        if not existing:
            continue
        union = tag_set | existing
        if union and len(tag_set & existing) / len(union) >= overlap_threshold:
            return False
    return True


def try_add_to_bank(
    question: str,
    answer: str,
    image_path: str,
    dict_res: dict,
    bank: List[dict],
    bank_path: str,
    min_reasoning_diff: float = 1.0,
    overlap_threshold: float = 0.6,
    metadata: Optional[dict] = None,
    filter_criteria: str = "default",
) -> bool:
    """
    Add question to bank if it has sufficient difficulty and is novel.
    Updates the in-memory bank list and appends to the JSONL file.
    Returns True if added.
    """
    metadata = metadata or {}
    scores = extract_difficulty_scores(dict_res)
    if filter_criteria == "image_perception_only":
        if scores["perception_difficulty_image"] < 1.0:
            return False
    elif (scores["reasoning_difficulty"] < min_reasoning_diff
            and scores["perception_difficulty_image"] < 1.0):
        return False

    if not validate_rag_question(
        question=question,
        seed_question=metadata.get("seed_question", ""),
        content=metadata.get("content", {}),
        require_structured=False,
    )["valid"]:
        return False

    concept_tags = extract_concept_tags(question)
    if not is_novel(concept_tags, bank, overlap_threshold):
        return False

    entry = {
        "question": question,
        "answer": answer,
        "image_path": image_path,
        "reasoning_difficulty": scores["reasoning_difficulty"],
        "perception_difficulty_image": scores["perception_difficulty_image"],
        "model_following_capability": scores["model_following_capability"],
        "concept_tags": concept_tags,
        "specificity_score": float(metadata.get("specificity_score", compute_specificity_score(question))),
        "why_hard": metadata.get("why_hard", ""),
    }
    bank.append(entry)
    append_to_bank_file(entry, bank_path)
    return True


def is_high_quality_bank_entry(entry: dict) -> bool:
    question = entry.get("question", "")
    validation = validate_rag_question(
        question=question,
        seed_question="",
        content={},
        require_structured=False,
    )
    stored_specificity = float(entry.get("specificity_score", validation["specificity_score"]))
    if not validation["valid"] and validation["failure_reason"] != "low_specificity":
        return False
    if float(entry.get("reasoning_difficulty", 0.0)) < 2.0:
        return False
    if stored_specificity < 1.5:
        return False
    return True


# ── Image fact extraction ─────────────────────────────────────────────────────

_FACT_EXTRACTION_PROMPT = (
    "Analyze this image carefully. Return a JSON with:\n"
    '{"objects": ["list of main visible objects"], '
    '"attributes": {"object_name": ["attr1", "attr2"]}, '
    '"spatial_relations": [{"subject": "obj1", "relation": "relation_type", "object": "obj2"}]}\n'
    "Only include clearly visible information. Return valid JSON only."
)


def get_image_facts(
    image,
    Dict_config_m,
    model_infer_fn,
    cache_dir: str,
    image_path: str,
) -> Optional[dict]:
    """
    Extract structured facts from image. Cache to avoid redundant inference.
    Returns dict with keys: objects, attributes, spatial_relations. None on failure.
    """
    cache_key = hashlib.md5(image_path.encode()).hexdigest()[:16]
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            pass

    try:
        result, _ = model_infer_fn(
            dict_input={"prompt": _FACT_EXTRACTION_PROMPT, "image": image},
            dict_config_m=Dict_config_m,
            max_len=1024,
            do_sample=False,
        )
        if not isinstance(result, dict) or "objects" not in result:
            return None
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(result, f)
        return result
    except Exception:
        return None


def _format_facts_text(facts: dict) -> str:
    """Format extracted facts dict into prompt-ready text."""
    lines = []
    objects = facts.get("objects", [])
    if objects:
        lines.append("Objects: " + ", ".join(str(o) for o in objects[:10]))
    attrs = facts.get("attributes", {})
    if attrs:
        attr_lines = [
            f"  {k}: {', '.join(str(v) for v in (list(vals.values()) if isinstance(vals, dict) else list(vals))[:3])}"
            for k, vals in list(attrs.items())[:5]
        ]
        lines.append("Attributes:\n" + "\n".join(attr_lines))
    rels = facts.get("spatial_relations", [])
    if rels:
        rel_lines = [
            f"  {r.get('subject','')} {r.get('relation','')} {r.get('object','')}"
            for r in rels[:5]
        ]
        lines.append("Spatial relations:\n" + "\n".join(rel_lines))
    return "\n".join(lines) if lines else "No facts extracted."


def _format_examples_text(sampled: List[dict]) -> str:
    """Format sampled bank entries into prompt-ready examples."""
    if not sampled:
        return "No examples available."
    lines = []
    for i, entry in enumerate(sampled, 1):
        q = entry.get("question", "")
        a = entry.get("answer", "")
        why_hard = entry.get("why_hard", "")
        if not why_hard:
            why_hard = (
                f"Requires about {int(round(float(entry.get('reasoning_difficulty', 0.0))))} "
                f"reasoning steps and grounded comparison across visible evidence."
            )
        lines.append(f"Example {i}:\n  Q: {q}\n  A: {a}\n  Why hard: {why_hard}")
    return "\n".join(lines)


def _dedupe_keep_order(items: List[str]) -> List[str]:
    """Deduplicate strings while preserving the original order."""
    seen = set()
    deduped = []
    for item in items:
        if not isinstance(item, str):
            continue
        item = item.strip().lower()
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _extract_fact_keywords(facts: dict) -> List[str]:
    """Extract training-friendly keywords from fact hints."""
    keywords: List[str] = []

    for obj in facts.get("objects", []):
        if isinstance(obj, str):
            keywords.append(obj)

    for obj_name, attrs in facts.get("attributes", {}).items():
        if isinstance(obj_name, str):
            keywords.append(obj_name)
        if isinstance(attrs, dict):
            attr_values = list(attrs.values())
        elif isinstance(attrs, list):
            attr_values = attrs
        else:
            attr_values = [attrs]
        for value in attr_values:
            if isinstance(value, str):
                keywords.append(value)

    for rel in facts.get("spatial_relations", []):
        if not isinstance(rel, dict):
            continue
        for key in ("subject", "relation", "object"):
            value = rel.get(key)
            if isinstance(value, str):
                keywords.append(value)

    return _dedupe_keep_order(keywords)


def _extract_example_keywords(sampled_bank: List[dict]) -> List[str]:
    """Extract training-friendly keywords from sampled example hints."""
    keywords: List[str] = []
    for entry in sampled_bank:
        keywords.extend(entry.get("concept_tags", []))
    return _dedupe_keep_order(keywords)


# ── Instruction builder ───────────────────────────────────────────────────────

def generate_rag_instructions(
    seed_question: str,
    seed_answer: str,
    facts: Optional[dict],
    sampled_bank: List[dict],
) -> List[dict]:
    """
    Build RAG-augmented instructions:
      - add_multihop_rag_facts_v1  (if facts available)
      - add_example_harder_v1      (if bank has entries)

    Returns a list of instruction records with prompt metadata and RAG hints.
    """
    instruction_records: List[dict] = []

    if facts:
        facts_text = _format_facts_text(facts)
        prompt = (
            broader_question_prompts
            .replace("{rag_facts}", facts_text)
            .replace("{seed_question}", seed_question)
            .replace("{seed_answer}", seed_answer)
        )
        instruction_records.append({
            "instruction": prompt,
            "constraint_tag": "rag_multihop_facts",
            "prompt_name": "add_multihop_rag_facts_v1",
            "rag_context": {
                "hint_type": "facts",
                "hint_text": facts_text,
                "hint_keywords": _extract_fact_keywords(facts),
                "hint_payload": facts,
            },
        })

    if sampled_bank:
        examples_text = _format_examples_text(sampled_bank)
        prompt = (
            harder_question_prompts
            .replace("{examples_text}", examples_text)
            .replace("{seed_question}", seed_question)
            .replace("{seed_answer}", seed_answer)
        )
        instruction_records.append({
            "instruction": prompt,
            "constraint_tag": "rag_example_harder",
            "prompt_name": "add_example_harder_v1",
            "rag_context": {
                "hint_type": "bank_examples",
                "hint_text": examples_text,
                "hint_keywords": _extract_example_keywords(sampled_bank),
                "hint_payload": sampled_bank,
            },
        })

    return instruction_records


def rank_seed_pairs_for_stage2(
    seed_pairs: List[Tuple[str, str, dict, List]],
    *,
    top_k: int = 8,
    filter_criteria: str = "default",
) -> List[Tuple[str, str, dict, List]]:
    scored = []
    for pair in seed_pairs:
        question, answer, met, constraints = pair
        scores = extract_difficulty_scores(met)
        specificity = compute_specificity_score(question)
        generic_penalty = len(detect_generic_pattern_flags(question))
        if filter_criteria == "image_perception_only":
            total = (
                3.0 * scores["perception_difficulty_image"]
                + 1.5 * specificity
                - 2.0 * generic_penalty
            )
        else:
            total = (
                2.0 * scores["reasoning_difficulty"]
                + 1.0 * scores["perception_difficulty_image"]
                + 1.5 * specificity
                - 2.0 * generic_penalty
            )
        scored.append((total, pair))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [pair for _, pair in scored[:max(1, min(top_k, len(scored)))]]
