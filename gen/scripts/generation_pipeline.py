"""
RAG-augmented question generation pipeline.

RAG/self-evolving question generation with two evolution passes per image:
  1. add_multihop_rag_facts_v1  - multi-hop via extracted image facts
  2. add_example_harder_v1      - harder-than-examples via sample bank

Accepted questions are added back to the sample bank (novelty-filtered),
bootstrapping the bank's difficulty ceiling over multiple rounds.

Usage (see gen/run_cmd_sat_3b.sh):
  export JSON_FILE=/path/to/questions.json
  python3 gen/scripts/generation_pipeline.py \\
      --model_type=qwen25_3b \\
      --postfix=sat \\
      --partition=4 --partition_id=0 \\
      --num_turn=2 \\
      --save_dir=output/data/pipeline_rag_sat_3b_part \\
      --rag_bank=output/rag/sample_questions.jsonl \\
      --fact_cache_dir=output/rag/facts
"""

import sys
import os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from gen.scripts.qwen_unified import *
import utils.gen.evol_utils as evol_utils
import utils.gen.selfjudge_utils as selfjudge_utils
import utils.gen.retrieval_utils as rag_utils
import traceback
from typing import Callable, List, Tuple, Dict, Any, Optional
import torch
import warnings
from collections import defaultdict
import json
import random
from tqdm import tqdm
from argparse import ArgumentParser

FILTER_MODE_FILTERED = "filtered"
FILTER_MODE_UNFILTERED = "unfiltered"
FILTER_MODE_CHOICES = (
    FILTER_MODE_FILTERED,
    FILTER_MODE_UNFILTERED,
)

EVOLUTION_MODE_EVOLVED = "evolved"
EVOLUTION_MODE_NONE = "none"
EVOLUTION_MODE_CHOICES = (
    EVOLUTION_MODE_EVOLVED,
    EVOLUTION_MODE_NONE,
)


def self_judge_unfiltered(qs, ans, image, Dict_config_m, constraints=[]):
    """Accept generated QA pairs without running the model-based filter."""
    return True, {
        "filter_mode": FILTER_MODE_UNFILTERED,
        "accepted_without_filtering": True,
        "question": qs,
        "answer": ans,
        "constraints": constraints,
    }


def is_multi_question(text: str) -> bool:
    """Return True if text appears to contain multiple questions."""
    import re
    if not isinstance(text, str):
        return False
    if text.count("?") > 1:
        return True
    pattern = r'(?:^|[,;]\s*(?:(?:and|or|but)\s+)?)(?:what|how|why|where|when|which|who)\b'
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return len(matches) > 1


def build_seed_rewrite_records(list_evol_pair, source: str = "seed_generation") -> List[dict]:
    """Represent seed QA pairs in the rewritten schema used by training prep."""
    records = []
    for seed_q, seed_a, seed_met in list_evol_pair:
        records.append({
            "seed_q": seed_q,
            "seed_a": seed_a,
            "rewritten": {
                "Rewritten question": seed_q,
                "Answer": seed_a,
            },
            "evolution": 0,
            "promptver": source,
            "constraint": [],
            "simplify": seed_q,
            "complex": seed_q,
            "instruction": "",
            "accepted": True,
            "validation_failure_reason": "",
            "filter_met": seed_met,
        })
    return records


def gen_and_judge(
    prompt: str,
    image,
    Dict_config_m,
    model_infer_fn,
    dict_qs_met: Dict[str, Any],
    dict_ans: Dict,
    selfjudge_fn,
):
    """Try up to four generations, simplifying and answering each before judging.

    Cache accepted metrics by original question and answers by simplified question;
    return (None, None) when no candidate passes.
    """
    max_retry = 3
    retry_idx = 0
    while retry_idx <= max_retry:
        out, _ = model_infer_fn(
            dict_input={"prompt": prompt, "image": image},
            dict_config_m=Dict_config_m,
            do_sample=True,
            max_len=4096,
        )
        retry_idx += 1

        if not (isinstance(out, dict) and out.get("Question") is not None):
            continue

        qs = out["Question"]
        if is_multi_question(qs):
            continue

        simp_prompt = selfjudge_utils.simplify_prompt()
        simplified, _ = model_infer_fn(
            dict_input={"prompt": simp_prompt + f"Question: {qs}"},
            dict_config_m=Dict_config_m,
            do_sample=True,
            max_len=512,
        )
        if "What effect" in simplified or "What happens" in simplified:
            continue
        if simplified in dict_ans:
            continue

        ans_prompt = selfjudge_utils.gene_answer()
        ans, _ = model_infer_fn(
            dict_input={"prompt": ans_prompt + f"Question: {simplified}", "image": image},
            dict_config_m=Dict_config_m,
            do_sample=True,
            max_len=2048,
        )
        dict_ans[simplified] = ans

        out["Simplified"] = simplified
        out["Answer"] = ans

        if qs in dict_qs_met:
            return out, dict_qs_met[qs]

        check_suc, filter_info = selfjudge_fn(
            qs=simplified,
            ans=ans,
            image=image,
            Dict_config_m=Dict_config_m,
            constraints=[],
        )
        if check_suc:
            dict_qs_met[qs] = filter_info
            return out, filter_info
    return None, None


def _run_evolution_stage(
    seed_pairs: List[Tuple[str, str, dict, List[dict]]],
    *,
    evolution_id: int,
    generate_instructions_fn,
    image,
    Dict_config_m,
    model_infer_fn,
    selfjudge_fn,
    dict_qs_met: Dict[str, Any],
    dict_ans: Dict,
    list_rewritten: List[dict],
    check_improvement_fn: Callable[[dict, dict], bool],
) -> List[Tuple[str, str, dict, List[dict]]]:
    next_pairs: List[Tuple[str, str, dict, List[dict]]] = []
    for seed_question, seed_answer, seed_met, prev_constraints in seed_pairs:
        try:
            if not seed_question or not seed_answer:
                continue

            list_instructions, list_constraints, list_prompt = generate_instructions_fn(
                image, seed_question, seed_answer, seed_met, prev_constraints
            )
            if not list_instructions:
                continue

            for instruction, constraints, promptver in zip(
                list_instructions, list_constraints, list_prompt
            ):
                content, _ = model_infer_fn(
                    dict_input={"prompt": instruction, "image": image},
                    dict_config_m=Dict_config_m,
                    max_len=4096,
                )
                if not isinstance(content, dict):
                    continue

                rew_q = content.get("Rewritten question")
                if rew_q is None:
                    continue

                if is_multi_question(rew_q):
                    list_rewritten.append({
                        "seed_q": seed_question,
                        "seed_a": seed_answer,
                        "rewritten": content,
                        "evolution": evolution_id,
                        "promptver": promptver,
                        "constraint": prev_constraints + [constraints],
                        "simplify": "",
                        "complex": rew_q,
                        "instruction": instruction,
                        "accepted": False,
                        "validation_failure_reason": "multi_question",
                    })
                    continue

                simp_prompt = selfjudge_utils.simplify_prompt()
                simplify, _ = model_infer_fn(
                    dict_input={"prompt": simp_prompt + f"Question: {rew_q}"},
                    dict_config_m=Dict_config_m,
                    do_sample=True,
                    max_len=512,
                )
                if simplify in dict_ans:
                    list_rewritten.append({
                        "seed_q": seed_question,
                        "seed_a": seed_answer,
                        "rewritten": content,
                        "evolution": evolution_id,
                        "promptver": promptver,
                        "constraint": prev_constraints + [constraints],
                        "simplify": simplify,
                        "complex": rew_q,
                        "instruction": instruction,
                        "accepted": False,
                        "validation_failure_reason": "duplicate_simplified_question",
                    })
                    continue

                ans_prompt = selfjudge_utils.gene_answer()
                ans, _ = model_infer_fn(
                    dict_input={"prompt": ans_prompt + f"Question: {simplify}", "image": image},
                    dict_config_m=Dict_config_m,
                    do_sample=True,
                    max_len=2048,
                )
                dict_ans[simplify] = ans

                rew_q_ori = rew_q
                rew_q = simplify
                rew_a = ans
                content["Rewritten question"] = simplify
                content["Answer"] = ans

                if not rew_q or not isinstance(rew_q, str):
                    continue
                if not rew_a or not isinstance(rew_a, str):
                    continue

                new_constraints = prev_constraints + [constraints]
                if rag_utils.lexical_overlap_ratio(rew_q, seed_question) >= 0.82:
                    list_rewritten.append({
                        "seed_q": seed_question,
                        "seed_a": seed_answer,
                        "rewritten": content,
                        "evolution": evolution_id,
                        "promptver": promptver,
                        "constraint": new_constraints,
                        "simplify": simplify,
                        "complex": rew_q_ori,
                        "instruction": instruction,
                        "accepted": False,
                        "validation_failure_reason": "lexical_overlap_too_high",
                    })
                    continue

                if rew_q in dict_qs_met:
                    rew_met = dict_qs_met[rew_q]
                    check_suc = True
                else:
                    check_suc, rew_met = selfjudge_fn(
                        qs=rew_q,
                        ans=rew_a,
                        image=image,
                        Dict_config_m=Dict_config_m,
                        constraints=new_constraints,
                    )
                    if check_suc:
                        dict_qs_met[rew_q] = rew_met

                if not check_suc:
                    list_rewritten.append({
                        "seed_q": seed_question,
                        "seed_a": seed_answer,
                        "rewritten": content,
                        "evolution": evolution_id,
                        "promptver": promptver,
                        "constraint": new_constraints,
                        "simplify": simplify,
                        "complex": rew_q_ori,
                        "instruction": instruction,
                        "accepted": False,
                        "validation_failure_reason": "selfjudge_failed",
                    })
                    continue

                if not check_improvement_fn(seed_met, rew_met):
                    list_rewritten.append({
                        "seed_q": seed_question,
                        "seed_a": seed_answer,
                        "rewritten": content,
                        "evolution": evolution_id,
                        "promptver": promptver,
                        "constraint": new_constraints,
                        "simplify": simplify,
                        "complex": rew_q_ori,
                        "instruction": instruction,
                        "accepted": False,
                        "validation_failure_reason": "difficulty_not_improved_enough",
                    })
                    continue

                next_pairs.append((rew_q, rew_a, rew_met, new_constraints))
                list_rewritten.append({
                    "seed_q": seed_question,
                    "seed_a": seed_answer,
                    "rewritten": content,
                    "evolution": evolution_id,
                    "promptver": promptver,
                    "constraint": new_constraints,
                    "simplify": simplify,
                    "complex": rew_q_ori,
                    "instruction": instruction,
                    "accepted": True,
                    "validation_failure_reason": "",
                })
            if len(next_pairs) >= 2:
                return next_pairs
        except Exception as e:
            print(e)
            torch.cuda.empty_cache()
            traceback.print_exc()
    return next_pairs

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "create_training_utils",
    os.path.join(os.path.dirname(__file__), "..", "create_training", "utils.py"),
)
_ct_utils = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ct_utils)
get_formatted_conversation = _ct_utils.get_formatted_conversation

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

FILTER_CRITERIA_DEFAULT = "default"
FILTER_CRITERIA_IMAGE_PERCEPTION_ONLY = "image_perception_only"
FILTER_CRITERIA_CHOICES = (
    FILTER_CRITERIA_DEFAULT,
    FILTER_CRITERIA_IMAGE_PERCEPTION_ONLY,
)


def _has_image_perception_improved(seed_met: dict, rew_met: dict, min_delta: float = 1.0) -> bool:
    """Compare visual grounding scores against the requested minimum increase."""
    seed_scores = rag_utils.extract_difficulty_scores(seed_met)
    rew_scores = rag_utils.extract_difficulty_scores(rew_met)
    perception_delta = (
        rew_scores.get("perception_difficulty_image", 0.0)
        - seed_scores.get("perception_difficulty_image", 0.0)
    )
    return perception_delta >= min_delta


def _build_improvement_fn(filter_criteria: str) -> Callable[[dict, dict], bool]:
    """Select the visual-only or combined difficulty gate for rewritten questions."""
    if filter_criteria == FILTER_CRITERIA_IMAGE_PERCEPTION_ONLY:
        return _has_image_perception_improved
    return lambda sm, rm: selfjudge_utils.has_difficulty_improved(seed_met=sm, rew_met=rm)


def _gen_seed_with_finetuned(
    image,
    Dict_config_m,
    Dict_config_rewrite,
    model_infer_fn,
    dict_qs_met,
    dict_ans,
    selfjudge_fn,
    max_retry=3,
):
    """Generate a seed question with the finetuned model (IQ prompt format),
    then simplify/answer/judge with Dict_config_rewrite (Qwen2.5)."""
    for _ in range(max_retry):
        conv = get_formatted_conversation("IQ", "", "")
        prompt = conv[0]["value"].replace("<image>\n", "", 1).replace('Task: IQ.\n', '').replace('Task: IQA.\n', '')
        prompt += evol_utils.gen_format

        raw, _ = model_infer_fn( dict_input={"prompt": prompt, "image": image}, dict_config_m=Dict_config_m, do_sample=True, max_len=512, )
        if isinstance(raw, str) and raw.startswith("Question:"):
            qs = raw[len("Question:"):].strip()
        elif isinstance(raw, dict) and raw.get("Question"):
            qs = raw["Question"]
        else:
            continue

        if not qs or is_multi_question(qs):
            continue
        simp_prompt = selfjudge_utils.simplify_prompt()
        simplified, _ = model_infer_fn(
            dict_input={"prompt": simp_prompt + f"Question: {qs}"},
            dict_config_m=Dict_config_rewrite,
            do_sample=True, max_len=512,
        )
        if not isinstance(simplified, str) or not simplified:
            continue
        if "What effect" in simplified or "What happens" in simplified:
            continue
        if simplified in dict_ans:
            continue

        ans_prompt = selfjudge_utils.gene_answer()
        ans, _ = model_infer_fn(
            dict_input={"prompt": ans_prompt + f"Question: {simplified}", "image": image},
            dict_config_m=Dict_config_rewrite,
            do_sample=True, max_len=2048,
        )
        dict_ans[simplified] = ans

        if qs in dict_qs_met:
            return {"Question": qs, "Simplified": simplified, "Answer": ans}, dict_qs_met[qs]

        check_suc, filter_info = selfjudge_fn(
            qs=simplified, ans=ans, image=image,
            Dict_config_m=Dict_config_rewrite,
            constraints=[],
        )
        if check_suc:
            dict_qs_met[qs] = filter_info
            return {"Question": qs, "Simplified": simplified, "Answer": ans}, filter_info

    return None, None


def _append_rewrite_record(
    list_rewritten: List[dict],
    *,
    seed_question: str,
    seed_answer: str,
    evolution_id: int,
    prompt_name: str,
    constraints: List,
    rag_context: dict,
    instruction: str,
    content: Optional[dict] = None,
    simplify: str = "",
    complex_q: str = "",
    accepted: bool = False,
    validation_failure_reason: str = "",
    generic_pattern_flags: Optional[List[str]] = None,
    specificity_score: Optional[float] = None,
    structure_preserved: Optional[bool] = None,
    structure_details: Optional[dict] = None,
):
    list_rewritten.append({
        "seed_q": seed_question,
        "seed_a": seed_answer,
        "rewritten": content or {},
        "evolution": evolution_id,
        "promptver": prompt_name,
        "constraint": constraints,
        "rag_context": rag_context,
        "simplify": simplify,
        "complex": complex_q,
        "instruction": instruction[:300] + "...",
        "accepted": accepted,
        "validation_failure_reason": validation_failure_reason,
        "generic_pattern_flags": generic_pattern_flags or [],
        "specificity_score": specificity_score,
        "structure_preserved": structure_preserved,
        "structure_details": structure_details or {},
    })


def _run_rag_evolution_stage(
    seed_pairs: List[Tuple[str, str, dict, List]],
    *,
    evolution_id: int,
    image,
    image_path: str,
    Dict_config_m,
    model_infer_fn,
    selfjudge_fn,
    dict_qs_met: Dict[str, Any],
    dict_ans: Dict,
    list_rewritten: List[dict],
    check_improvement_fn: Callable,
    bank: List[dict],
    bank_path: str,
    fact_cache_dir: str,
    filter_criteria: str = FILTER_CRITERIA_DEFAULT,
    filtering_enabled: bool = True,
) -> List[Tuple[str, str, dict, List]]:
    """
    RAG-augmented evolution stage.

    For each seed pair, generates RAG-augmented instructions (fact-chain and
    example-harder), runs the model, self-judges, checks difficulty improvement,
    and adds accepted questions to the sample bank.
    """
    next_pairs: List[Tuple[str, str, dict, List]] = []

    # Extract image facts once per image call (cached on disk)
    facts = rag_utils.get_image_facts(
        image=image,
        Dict_config_m=Dict_config_m,
        model_infer_fn=model_infer_fn,
        cache_dir=fact_cache_dir,
        image_path=image_path,
    )

    # Sample exemplars from bank
    for seed_question, seed_answer, seed_met, prev_constraints in seed_pairs:
        try:
            if not seed_question or not seed_answer:
                continue

            sampled_bank = rag_utils.sample_from_bank(bank, k=5, seed_question=seed_question)
            instruction_records = rag_utils.generate_rag_instructions(
                seed_question=seed_question,
                seed_answer=seed_answer,
                facts=facts,
                sampled_bank=sampled_bank,
            )

            if not instruction_records:
                continue

            for instruction_record in instruction_records:
                instruction = instruction_record["instruction"]
                constraint_tag = instruction_record["constraint_tag"]
                prompt_name = instruction_record["prompt_name"]
                rag_context = instruction_record["rag_context"]
                content, _ = model_infer_fn(
                    dict_input={"prompt": instruction, "image": image},
                    dict_config_m=Dict_config_m,
                    max_len=4096,
                )

                if not isinstance(content, dict):
                    continue

                rew_q = content.get("Rewritten question", None)
                if rew_q is None:
                    _append_rewrite_record(
                        list_rewritten,
                        seed_question=seed_question,
                        seed_answer=seed_answer,
                        evolution_id=evolution_id,
                        prompt_name=prompt_name,
                        constraints=prev_constraints + [constraint_tag],
                        rag_context=rag_context,
                        instruction=instruction,
                        content=content,
                        validation_failure_reason="missing_rewritten_question",
                    )
                    continue

                if is_multi_question(rew_q):
                    _append_rewrite_record(
                        list_rewritten,
                        seed_question=seed_question,
                        seed_answer=seed_answer,
                        evolution_id=evolution_id,
                        prompt_name=prompt_name,
                        constraints=prev_constraints + [constraint_tag],
                        rag_context=rag_context,
                        instruction=instruction,
                        content=content,
                        complex_q=rew_q,
                        validation_failure_reason="multi_question",
                    )
                    continue

                raw_validation = rag_utils.validate_rag_question(
                    question=rew_q,
                    seed_question=seed_question,
                    content=content,
                    require_structured=True,
                )
                new_constraints = prev_constraints + [constraint_tag]
                if not raw_validation["valid"]:
                    _append_rewrite_record(
                        list_rewritten,
                        seed_question=seed_question,
                        seed_answer=seed_answer,
                        evolution_id=evolution_id,
                        prompt_name=prompt_name,
                        constraints=new_constraints,
                        rag_context=rag_context,
                        instruction=instruction,
                        content=content,
                        complex_q=rew_q,
                        validation_failure_reason=str(raw_validation["failure_reason"]),
                        generic_pattern_flags=raw_validation["generic_pattern_flags"],
                        specificity_score=raw_validation["specificity_score"],
                    )
                    continue

                # Simplify
                simp_prompt = selfjudge_utils.simplify_prompt()
                simplify, _ = model_infer_fn(
                    dict_input={"prompt": simp_prompt + f"Question: {rew_q}"},
                    dict_config_m=Dict_config_m,
                    do_sample=True,
                    max_len=512,
                )
                structure_preserved, structure_details = selfjudge_utils.preserves_reasoning_structure(
                    rew_q,
                    simplify,
                    content=content,
                )
                if not structure_preserved:
                    _append_rewrite_record(
                        list_rewritten,
                        seed_question=seed_question,
                        seed_answer=seed_answer,
                        evolution_id=evolution_id,
                        prompt_name=prompt_name,
                        constraints=new_constraints,
                        rag_context=rag_context,
                        instruction=instruction,
                        content=content,
                        simplify=simplify,
                        complex_q=rew_q,
                        validation_failure_reason=str(structure_details.get("reason", "structure_not_preserved")),
                        generic_pattern_flags=raw_validation["generic_pattern_flags"],
                        specificity_score=raw_validation["specificity_score"],
                        structure_preserved=False,
                        structure_details=structure_details,
                    )
                    continue

                simp_validation = rag_utils.validate_rag_question(
                    question=simplify,
                    seed_question=seed_question,
                    content=content,
                    require_structured=True,
                )
                if not simp_validation["valid"]:
                    _append_rewrite_record(
                        list_rewritten,
                        seed_question=seed_question,
                        seed_answer=seed_answer,
                        evolution_id=evolution_id,
                        prompt_name=prompt_name,
                        constraints=new_constraints,
                        rag_context=rag_context,
                        instruction=instruction,
                        content=content,
                        simplify=simplify,
                        complex_q=rew_q,
                        validation_failure_reason=str(simp_validation["failure_reason"]),
                        generic_pattern_flags=simp_validation["generic_pattern_flags"],
                        specificity_score=simp_validation["specificity_score"],
                        structure_preserved=True,
                        structure_details=structure_details,
                    )
                    continue

                if simplify in dict_ans:
                    _append_rewrite_record(
                        list_rewritten,
                        seed_question=seed_question,
                        seed_answer=seed_answer,
                        evolution_id=evolution_id,
                        prompt_name=prompt_name,
                        constraints=new_constraints,
                        rag_context=rag_context,
                        instruction=instruction,
                        content=content,
                        simplify=simplify,
                        complex_q=rew_q,
                        validation_failure_reason="duplicate_simplified_question",
                        generic_pattern_flags=simp_validation["generic_pattern_flags"],
                        specificity_score=simp_validation["specificity_score"],
                        structure_preserved=True,
                        structure_details=structure_details,
                    )
                    continue

                # Generate answer
                ans_prompt = selfjudge_utils.gene_answer()
                ans, _ = model_infer_fn(
                    dict_input={"prompt": ans_prompt + f"Question: {simplify}", "image": image},
                    dict_config_m=Dict_config_m,
                    do_sample=True,
                    max_len=2048,
                )
                dict_ans[simplify] = ans

                rew_q_ori = rew_q
                rew_q = simplify
                rew_a = ans
                content["Rewritten question"] = simplify
                content["Answer"] = ans

                if not rew_q or not isinstance(rew_q, str):
                    continue
                if not rew_a or not isinstance(rew_a, str):
                    continue

                rew_met = {}

                if rew_q in dict_qs_met:
                    rew_met = dict_qs_met[rew_q]
                    check_suc = True
                else:
                    check_suc, rew_met = selfjudge_fn(
                        qs=rew_q,
                        ans=rew_a,
                        image=image,
                        Dict_config_m=Dict_config_m,
                        constraints=new_constraints,
                    )
                    if check_suc:
                        dict_qs_met[rew_q] = rew_met

                if not check_suc:
                    _append_rewrite_record(
                        list_rewritten,
                        seed_question=seed_question,
                        seed_answer=seed_answer,
                        evolution_id=evolution_id,
                        prompt_name=prompt_name,
                        constraints=new_constraints,
                        rag_context=rag_context,
                        instruction=instruction,
                        content=content,
                        simplify=simplify,
                        complex_q=rew_q_ori,
                        validation_failure_reason="selfjudge_failed",
                        generic_pattern_flags=simp_validation["generic_pattern_flags"],
                        specificity_score=simp_validation["specificity_score"],
                        structure_preserved=True,
                        structure_details=structure_details,
                    )
                    continue

                if not check_improvement_fn(seed_met, rew_met):
                    _append_rewrite_record(
                        list_rewritten,
                        seed_question=seed_question,
                        seed_answer=seed_answer,
                        evolution_id=evolution_id,
                        prompt_name=prompt_name,
                        constraints=new_constraints,
                        rag_context=rag_context,
                        instruction=instruction,
                        content=content,
                        simplify=simplify,
                        complex_q=rew_q_ori,
                        validation_failure_reason="difficulty_not_improved_enough",
                        generic_pattern_flags=simp_validation["generic_pattern_flags"],
                        specificity_score=simp_validation["specificity_score"],
                        structure_preserved=True,
                        structure_details=structure_details,
                    )
                    continue

                if filtering_enabled and not selfjudge_utils.has_required_model_following(rew_met, new_constraints):
                    _append_rewrite_record(
                        list_rewritten,
                        seed_question=seed_question,
                        seed_answer=seed_answer,
                        evolution_id=evolution_id,
                        prompt_name=prompt_name,
                        constraints=new_constraints,
                        rag_context=rag_context,
                        instruction=instruction,
                        content=content,
                        simplify=simplify,
                        complex_q=rew_q_ori,
                        validation_failure_reason="model_following_failed",
                        generic_pattern_flags=simp_validation["generic_pattern_flags"],
                        specificity_score=simp_validation["specificity_score"],
                        structure_preserved=True,
                        structure_details=structure_details,
                    )
                    continue

                # Add judged questions to the bank if novel. In unfiltered mode,
                # avoid polluting the shared RAG bank with unjudged generations.
                if filtering_enabled:
                    added = rag_utils.try_add_to_bank(
                        question=rew_q,
                        answer=rew_a,
                        image_path=image_path,
                        dict_res=rew_met,
                        bank=bank,
                        bank_path=bank_path,
                        metadata={
                            "seed_question": seed_question,
                            "content": content,
                            "specificity_score": simp_validation["specificity_score"],
                            "why_hard": content.get("why_harder_than_seed", ""),
                        },
                        filter_criteria=filter_criteria,
                    )
                    if added:
                        print(f"  [bank] Added novel question (bank size: {len(bank)})")

                next_pairs.append((rew_q, rew_a, rew_met, new_constraints))
                _append_rewrite_record(
                    list_rewritten,
                    seed_question=seed_question,
                    seed_answer=seed_answer,
                    evolution_id=evolution_id,
                    prompt_name=prompt_name,
                    constraints=new_constraints,
                    rag_context=rag_context,
                    instruction=instruction,
                    content=content,
                    simplify=simplify,
                    complex_q=rew_q_ori,
                    accepted=True,
                    generic_pattern_flags=simp_validation["generic_pattern_flags"],
                    specificity_score=simp_validation["specificity_score"],
                    structure_preserved=True,
                    structure_details=structure_details,
                )

            if len(next_pairs) >= 2:
                return next_pairs

        except Exception as e:
            print(e)
            torch.cuda.empty_cache()
            traceback.print_exc()

    return next_pairs


if __name__ == "__main__":
    ROOT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default=os.path.join(ROOT_FOLDER, ".cache", "hf"))
    parser.add_argument("--model_type", type=str, default="qwen25_3b")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--json_file", type=str, default=os.environ.get("JSON_FILE", ""))
    parser.add_argument("--gene_type", type=str, default="all")
    parser.add_argument("--save_dir", type=str, default="output/data/pipeline_rag_sat_3b_part")
    parser.add_argument("--postfix", type=str, default="sat")
    parser.add_argument("--partition", type=int, default=4)
    parser.add_argument("--partition_id", type=int, default=0)
    parser.add_argument("--num_turn", type=int, default=2)
    parser.add_argument("--rag_bank", type=str, default="output/rag/sample_questions.jsonl",
                        help="Shared sample bank JSONL (read at startup, written per-partition)")
    parser.add_argument("--rag_bank_out", type=str, default="",
                        help="Per-partition output bank file (default: rag_bank dir/bank_partN.jsonl)")
    parser.add_argument("--fact_cache_dir", type=str, default="output/rag/facts",
                        help="Directory for cached image fact JSON files")
    parser.add_argument("--filter_model_type", type=str, default="qwen25_3b",
                        help="Model type for filtering/self-judging (default: qwen25_3b base)")
    parser.add_argument("--filter_model_path", type=str, default="",
                        help="Explicit path to filter model (overrides --filter_model_type if set)")
    parser.add_argument("--filter_criteria", type=str, default=FILTER_CRITERIA_DEFAULT,
                        choices=FILTER_CRITERIA_CHOICES,
                        help="Difficulty filter criteria. Default keeps current reasoning/image-perception behavior.")
    parser.add_argument("--filter_mode", type=str, default=FILTER_MODE_FILTERED,
                        choices=FILTER_MODE_CHOICES,
                        help="Use 'unfiltered' to skip self-judge, difficulty-improvement, and model-following filters. Default: filtered.")
    parser.add_argument("--evolution_mode", type=str, default=EVOLUTION_MODE_EVOLVED,
                        choices=EVOLUTION_MODE_CHOICES,
                        help="Use 'none' to skip standard/RAG evolution and save seed QA only. Default: evolved.")
    parser.add_argument("--skip_evolution", action="store_true",
                        help="Alias for --evolution_mode none.")

    args = parser.parse_args()
    if not args.json_file:
        raise ValueError("Set --json_file or JSON_FILE to an input questions JSON path.")
    if args.skip_evolution:
        args.evolution_mode = EVOLUTION_MODE_NONE

    modeltype = args.model_type
    gene_type = args.gene_type
    save_dir = args.save_dir
    partition = args.partition
    partition_id = args.partition_id
    num_turn = args.num_turn
    postfix = args.postfix
    filter_criteria = args.filter_criteria
    filter_mode = args.filter_mode
    evolution_mode = args.evolution_mode
    filtering_enabled = filter_mode == FILTER_MODE_FILTERED

    # Load generation model
    if args.model_path == "":
        model_path = load_model_path_qwen(modeltype)
    else:
        model_path = args.model_path

    Dict_config_m = load_models_qwen(model_path=model_path)

    # Resolve per-partition bank output path
    rag_bank_out = args.rag_bank_out
    if not rag_bank_out:
        rag_bank_dir = os.path.dirname(os.path.abspath(args.rag_bank))
        rag_bank_out = os.path.join(rag_bank_dir, f"bank_part{modeltype}_{args.partition_id}.jsonl")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(args.fact_cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(rag_bank_out)), exist_ok=True)

    # Load input data
    with open(args.json_file, "r") as f:
        input_data = json.load(f)

    dict_definition, list_task, task_str = evol_utils.load_task_info()
    list_data = evol_utils.load_seed_data(input_data=input_data, postfix=postfix)

    dict_qs_met = load_exist_file(
        save_dir=save_dir,
        modeltype=f"{gene_type}_{modeltype}_gene{postfix}_filtermet_part{partition_id}",
        print_log=False,
    )
    if dict_qs_met is None:
        dict_qs_met: Dict[str, Any] = {}

    dict_ans = load_exist_file(
        save_dir=save_dir,
        modeltype=f"{gene_type}_{modeltype}_gene{postfix}_ans_part{partition_id}",
        print_log=False,
    )
    if dict_ans is None:
        dict_ans = {}

    # Load shared RAG sample bank
    bank = rag_utils.load_sample_bank(args.rag_bank)
    print(f"Loaded RAG sample bank: {len(bank)} entries from {args.rag_bank}")
    print(f"Per-partition bank output: {rag_bank_out}")
    print(f"Filter mode: {filter_mode}")
    print(f"Filter criteria: {filter_criteria}")
    print(f"Evolution mode: {evolution_mode}")

    list_booster = evol_utils.list_booster
    genqa_prompt_v1 = evol_utils.genqa_prompt_v1
    genqa_prompt_v2 = evol_utils.genqa_prompt_v2


    # Load filter model (may differ from generation model)
    if filtering_enabled:
        if args.filter_model_path:
            filter_model_path = args.filter_model_path
        else:
            filter_model_path = load_model_path_qwen(args.filter_model_type)

        if filter_model_path != model_path:
            print(f"Loading separate filter model: {filter_model_path}")
            Dict_config_filter = load_models_qwen(model_path=filter_model_path)
        else:
            Dict_config_filter = Dict_config_m
    else:
        Dict_config_filter = Dict_config_m

    # Wrapper: use the configured filter behavior, ignoring the passed Dict_config_m.
    def selfjudge_with_filter(qs, ans, image, Dict_config_m, constraints=[]):
        if not filtering_enabled:
            return self_judge_unfiltered(
                qs=qs, ans=ans, image=image,
                Dict_config_m=Dict_config_filter,
                constraints=constraints,
            )
        return selfjudge_utils.self_judge(
            qs=qs, ans=ans, image=image,
            Dict_config_m=Dict_config_filter,
            constraints=constraints,
        )

    # Partition data
    if partition > 1:
        n_data = len(list_data)
        if partition_id < 0 or partition_id >= partition or n_data == 0:
            list_data = []
        else:
            base, extra = divmod(n_data, partition)
            if partition_id < extra:
                start = partition_id * (base + 1)
                end = start + base + 1
            else:
                start = extra * (base + 1) + (partition_id - extra) * base
                end = start + base
            list_data = list_data[start:end]
            print(f"Partition {partition_id}: items {start}–{end} (total dataset: {n_data})")

    # Resume from existing results
    list_result = load_exist_file(
        save_dir=save_dir,
        modeltype=f"{gene_type}_{modeltype}_gene{postfix}_part{partition_id}",
        print_log=False,
    )
    if list_result and len(list_result) > 0:
        list_exist_img = {item["image_path"] for item in list_result}
        list_missing = [item for item in list_data if item["image_path"] not in list_exist_img]
        print(f"Resuming: {len(list_exist_img)} done, {len(list_missing)} remaining")
    else:
        list_missing = list_data
        list_result = []

    num_iter = 2

    _improvement_fn = _build_improvement_fn(filter_criteria) if filtering_enabled else lambda sm, rm: True

    def gen_instructions_stage(img, seed_q, seed_a, seed_met, prev_constraints):
        return evol_utils.generate_instructions_v2(
            image=img,
            seed_question=seed_q,
            seed_answer=seed_a,
            list_task=list_task,
            task_str=task_str,
            dict_definition=dict_definition,
            list_prev_const=[],
            Dict_config_m=Dict_config_filter,
            num_iter=num_iter,
        )

    for i, meta in enumerate(tqdm(list_missing)):
        try:
            image_path = meta["image_path"]
            image = load_image_qwen(image_path=image_path)

            dict_seed = defaultdict(list)
            list_evol_pair = []

            # ── Stage 0: Seed generation ──────────────────────────────────
            num_seed = 5 if gene_type == "all" else len(list_booster)
            for j in list_booster[:num_seed]:
                seed1, filt1 = gen_and_judge(
                    prompt=genqa_prompt_v1 + j,
                    image=image,
                    Dict_config_m=Dict_config_m,
                    model_infer_fn=model_infer_qwen,
                    dict_ans=dict_ans,
                    dict_qs_met=dict_qs_met,
                    selfjudge_fn=selfjudge_with_filter,
                )
                if seed1 is not None and seed1.get("Simplified") and seed1.get("Answer"):
                    list_evol_pair.append((seed1["Simplified"], seed1["Answer"], filt1))
                    dict_seed["v1"].append((seed1["Simplified"], seed1["Answer"]))

                seed2, filt2 = gen_and_judge(
                    prompt=genqa_prompt_v2 + j,
                    image=image,
                    Dict_config_m=Dict_config_m,
                    model_infer_fn=model_infer_qwen,
                    dict_ans=dict_ans,
                    dict_qs_met=dict_qs_met,
                    selfjudge_fn=selfjudge_with_filter,
                )
                if seed2 is not None and seed2.get("Simplified") and seed2.get("Answer"):
                    list_evol_pair.append((seed2["Simplified"], seed2["Answer"], filt2))
                    dict_seed["v2"].append((seed2["Simplified"], seed2["Answer"]))

            print(f"[{i}] Seeds: {len(list_evol_pair)}")
            write_to_file(save_dir=save_dir,
                          modeltype=f"{gene_type}_{modeltype}_gene{postfix}_filtermet_part{partition_id}",
                          result=dict_qs_met, print_log=False)
            write_to_file(save_dir=save_dir,
                          modeltype=f"{gene_type}_{modeltype}_gene{postfix}_ans_part{partition_id}",
                          result=dict_ans, print_log=False)

            list_rewritten: List[dict] = []

            if evolution_mode == EVOLUTION_MODE_NONE:
                list_rewritten = build_seed_rewrite_records(list_evol_pair)
                list_result.append({
                    "image_path": image_path,
                    "seed": dict_seed,
                    "rewritten": list_rewritten,
                })

                write_to_file(save_dir=save_dir,
                              modeltype=f"{gene_type}_{modeltype}_gene{postfix}_part{partition_id}",
                              result=list_result, print_log=True)
                write_to_file(save_dir=save_dir,
                              modeltype=f"{gene_type}_{modeltype}_gene{postfix}_filtermet_part{partition_id}",
                              result=dict_qs_met, print_log=False)
                write_to_file(save_dir=save_dir,
                              modeltype=f"{gene_type}_{modeltype}_gene{postfix}_ans_part{partition_id}",
                              result=dict_ans, print_log=False)
                print(f"[{i}] Evolution disabled; saved {len(list_rewritten)} seed QA records")
                continue

            seed_pairs_stage1 = [
                (sq, sa, sm, []) for (sq, sa, sm) in list_evol_pair
            ]

            # ── Stage 1: Standard evolution ───────────────────────────────
            list_loop_1 = _run_evolution_stage(
                seed_pairs=seed_pairs_stage1,
                evolution_id=1,
                generate_instructions_fn=gen_instructions_stage,
                image=image,
                Dict_config_m=Dict_config_filter,
                model_infer_fn=model_infer_qwen,
                selfjudge_fn=selfjudge_with_filter,
                dict_qs_met=dict_qs_met,
                dict_ans=dict_ans,
                list_rewritten=list_rewritten,
                check_improvement_fn=_improvement_fn,
            )
            print(f"[{i}] Standard evol-1: {len(list_loop_1)} questions")

            # ── Stage 1 RAG: RAG-augmented evolution on seeds ─────────────
            list_loop_1_rag = _run_rag_evolution_stage(
                seed_pairs=seed_pairs_stage1,
                evolution_id=1,
                image=image,
                image_path=image_path,
                Dict_config_m=Dict_config_filter,
                model_infer_fn=model_infer_qwen,
                selfjudge_fn=selfjudge_with_filter,
                dict_qs_met=dict_qs_met,
                dict_ans=dict_ans,
                list_rewritten=list_rewritten,
                check_improvement_fn=_improvement_fn,
                bank=bank,
                bank_path=rag_bank_out,
                fact_cache_dir=args.fact_cache_dir,
                filter_criteria=filter_criteria,
                filtering_enabled=filtering_enabled,
            )
            print(f"[{i}] RAG evol-1: {len(list_loop_1_rag)} questions")

            all_stage1 = list_loop_1 + list_loop_1_rag
            stage2_seed_pairs = rag_utils.rank_seed_pairs_for_stage2(
                all_stage1,
                top_k=8,
                filter_criteria=filter_criteria,
            ) if all_stage1 else []

            if num_turn > 1 and stage2_seed_pairs:
                # ── Stage 2: Standard evolution on all stage-1 outputs ────
                list_loop_2 = _run_evolution_stage(
                    seed_pairs=stage2_seed_pairs,
                    evolution_id=2,
                    generate_instructions_fn=gen_instructions_stage,
                    image=image,
                    Dict_config_m=Dict_config_filter,
                    model_infer_fn=model_infer_qwen,
                    selfjudge_fn=selfjudge_with_filter,
                    dict_qs_met=dict_qs_met,
                    dict_ans=dict_ans,
                    list_rewritten=list_rewritten,
                    check_improvement_fn=_improvement_fn,
                )
                print(f"[{i}] Standard evol-2: {len(list_loop_2)} questions")

                # ── Stage 2 RAG: RAG-augmented on stage-1 outputs ─────────
                list_loop_2_rag = _run_rag_evolution_stage(
                    seed_pairs=stage2_seed_pairs,
                    evolution_id=2,
                    image=image,
                    image_path=image_path,
                    Dict_config_m=Dict_config_filter,
                    model_infer_fn=model_infer_qwen,
                    selfjudge_fn=selfjudge_with_filter,
                    dict_qs_met=dict_qs_met,
                    dict_ans=dict_ans,
                    list_rewritten=list_rewritten,
                    check_improvement_fn=_improvement_fn,
                    bank=bank,
                    bank_path=rag_bank_out,
                    fact_cache_dir=args.fact_cache_dir,
                    filter_criteria=filter_criteria,
                    filtering_enabled=filtering_enabled,
                )
                print(f"[{i}] RAG evol-2: {len(list_loop_2_rag)} questions")

            list_result.append({
                "image_path": image_path,
                "seed": dict_seed,
                "rewritten": list_rewritten,
            })

            write_to_file(save_dir=save_dir,
                          modeltype=f"{gene_type}_{modeltype}_gene{postfix}_part{partition_id}",
                          result=list_result, print_log=True)
            write_to_file(save_dir=save_dir,
                          modeltype=f"{gene_type}_{modeltype}_gene{postfix}_filtermet_part{partition_id}",
                          result=dict_qs_met, print_log=False)
            write_to_file(save_dir=save_dir,
                          modeltype=f"{gene_type}_{modeltype}_gene{postfix}_ans_part{partition_id}",
                          result=dict_ans, print_log=False)

        except Exception as e:
            print(e)
            torch.cuda.empty_cache()
            traceback.print_exc()
            sys.exit(-1)
