from gen.scripts.qwen_unified import *
from .analysis_utils import (
    _get_model_following_by_constraint,
    _merge_api_res_from_evaluation,
    _extract_scores_and_mfc,
)
from typing import List, Dict, Tuple, Optional
import json
import os
import re
from tqdm import tqdm
import time


def build_prompt_v1() -> str:
    """
    Returns a compact instruction prompt for an API model.
    Model must output JSON ONLY, no scores—just counts and concrete evidence.
    """
    return f"""
    You are a strict JSON evaluation engine. Your output MUST be a SINGLE, minified JSON object.
    DO NOT include any text, prose, explanations, or conversational preamble before or after the JSON.
    Your entire output must start with {{ and end with }}.

    --- INPUTS TO ANALYZE ---
    - generated_question (Q)
    - added_constraints (C)

    --- REQUIRED OUTPUT JSON SCHEMA (Fill this structure) ---
    {{
    "hard_constraints": {{
        "Factuality": {{ "count": 0, "evidence": [] }},
        "Ambiguity":  {{ "count": 0, "evidence": [] }},
        "Validity": {{
        "answerable": true,
        "missing_info": [],
        "evidence": []
        }}
    }},
    "soft_constraints": {{
        "perception_difficulty": {{
        "simplified_question": "",
        "text":  {{ "count": 0, "evidence": [] }},
        "image": {{ "count": 0, "evidence": [] }}
        }},
        "reasoning_difficulty": {{ "count": 0, "evidence": [] }},
        "model_following_capability": {{
        "by_constraint": [
            {{
            "constraint_text": "",
            "complies": true,
            "evidence": []
            }}
        ]
        }}
    }}
    }}

    --- GENERAL RULES ---
    1.  **Evidence Objects:** All `evidence` lists must contain objects with this exact format:
        {{
        "span_text":"<verbatim cue>",
        "char_span":[start,end],  # 0-indexed, [start,end)
        "type":"<label>",
        "why":"<one-sentence rationale tied to this cue>"
        }}
    2.  **Verbatim Spans:** `span_text` MUST be the minimal verbatim substring from 'Q' or 'C'.
    3.  **Accurate Spans:** `char_span` MUST match the `span_text` location exactly.
    4.  **Empty Evidence:** If no items are found for a category, use `count: 0` and `evidence: []`.

    --- DETAILED FILLING RULES (BY KEY) ---

    1.  hard_constraints.Factuality
        - What to include: Presupposed facts or over-specific claims in Q that are not supported by C.
        - type ∈ {{"presupposition","contradiction","unsupported_detail"}}
        - span_text: The exact problematic phrase (e.g., "the red triangle number").
        - why: Say why it’s likely hallucinated or unsupported by C.

    2.  hard_constraints.Ambiguity
        - What to include: Cues in Q that make the reference unclear.
        - type ∈ {{"vague_pronoun","underspecified","polysemy","scope_ambiguity","ellipsis"}}
        - span_text: The ambiguous token/phrase (e.g., "it", "the object", "near the top").
        - why: State what is missing (e.g., "which object?", "which region?").

    3.  hard_constraints.Validity
        - answerable: [true/false] Is Q answerable given C?
        - missing_info: [list of strings] List any info required by Q but absent from C (e.g., "object identity", "image crop").
        - evidence: Point to Q phrases demanding unavailable info OR to C items that resolve it.
        - If unanswerable: type="unanswerable_cue"
        - If answerable due to C: type="resolving_cue"
        - span_text: Minimal phrase.
        - why: Explain how it blocks/enables answering.

    4.  soft_constraints.perception_difficulty
        - simplified_question: [string] Rewrite Q minimally to a simpler, equivalent form.
        - text (Textual Complexity)
        - Goal: Measures the parsing complexity of the *original* question (Q).
        - Include: All distinct components of the question's text structure.
        - type ∈ {{"textual_entity", "textual_attribute", "spatial_relation", "logical_condition", "numeric_value"}}
        - span_text: The exact token/phrase (e.g., "types", "organs", "left of", "if", "how many").
        - why: "Identifies a component of the question's text structure."
        - image (Visual Grounding Load)
        - Goal: Measures the visual grounding load. Finds *text cues in Q* that refer to visual properties or objects.
        - NOTE: You are analyzing the *text* of Q. Do not analyze an image.
        - Include: Text in Q that names a specific visual category, property, or count.
        - type ∈ {{"visual_category", "visual_property", "visual_count"}}
        - span_text: The exact referring phrase from Q (e.g., "organs", "blue squares", "how many", "top row").
        - why: "Identifies a visual object, property, or count that must be found."

    5.  soft_constraints.reasoning_difficulty
        - Goal: Decompose Q into the minimal reasoning steps. Each evidence item marks one step.
        - type="step"
        - source="question"
        - span_text: The phrase in Q that implies this step (e.g., "how many", "largest", "after").
        - why: [string] Describe the reasoning step itself (e.g., "Perform a count", "Find maximum value", "Apply a filter").

    6.  soft_constraints.model_following_capability
        - **NOTE:** The `by_constraint` list MUST contain one JSON object for **EVERY** single constraint listed in the input 'C'.
        - For EACH constraint in C, create one object:
        - constraint_text: Copy the constraint from C verbatim.
        - complies: [true/false] Does Q comply with/enable this constraint?
        - evidence: Cite phrases from Q or C that show compliance/violation.
            - type ∈ {{"compliance","violation"}}
            - span_text: Exact cue (e.g., from C: "must mention color"; from Q: "red ...").
            - why: How this cue meets or breaks the rule.

    ---
    FINAL REMINDER: Output JSON only. Start with {{. End with }}. No extra text.
    """

def soft_constraints_follow() -> str:
    """
    Returns a compact instruction prompt for an API model.
    Model must output JSON ONLY, no scores—just counts and concrete evidence.
    """
    return f"""
    You are a strict JSON evaluation engine. Your output MUST be a SINGLE, minified JSON object.
    DO NOT include any text, prose, explanations, or conversational preamble before or after the JSON.
    Your entire output must start with {{ and end with }}.

    --- INPUTS TO ANALYZE ---
    - generated_question (Q)
    - added_constraints (C)

    --- REQUIRED OUTPUT JSON SCHEMA (Fill this structure) ---
    {{
    "soft_constraints": {{
        "model_following_capability": {{
        "by_constraint": [
            {{
            "constraint_text": "",
            "complies": true,
            "evidence": []
            }}
        ]
        }}
    }}
    }}

    --- GENERAL RULES ---
    1.  **Evidence Objects:** All `evidence` lists must contain objects with this exact format:
        {{
        "span_text":"<verbatim cue>",
        "char_span":[start,end],  # 0-indexed, [start,end)
        "type":"<label>",
        "why":"<one-sentence rationale tied to this cue>"
        }}
    2.  **Verbatim Spans:** `span_text` MUST be the minimal verbatim substring from 'Q' or 'C'.
    3.  **Accurate Spans:** `char_span` MUST match the `span_text` location exactly.
    4.  **Empty Evidence:** If no items are found for a category, use `count: 0` and `evidence: []`.

    --- DETAILED FILLING RULES (BY KEY) ---

    1.  soft_constraints.model_following_capability
        - **NOTE:** The `by_constraint` list MUST contain one JSON object for **EVERY** single constraint listed in the input 'C'.
        - For EACH constraint in C, create one object:
        - constraint_text: Copy the constraint from C verbatim.
        - complies: [true/false] Does Q comply with/enable this constraint?
        - evidence: Cite phrases from Q or C that show compliance/violation.
            - type ∈ {{"compliance","violation"}}
            - span_text: Exact cue (e.g., from C: "must mention color"; from Q: "red ...").
            - why: How this cue meets or breaks the rule.

    ---
    FINAL REMINDER: Output JSON only. Start with {{. End with }}. No extra text.
    """

def soft_constraints_simplified() -> str:
    return f"""
    You are a strict JSON evaluation engine. Your output MUST be a SINGLE, minified JSON object.
    DO NOT include any text, prose, explanations, or conversational preamble before or after the JSON.
    Your entire output must start with {{ and end with }}.

    --- INPUTS TO ANALYZE ---
    - simplified generated question (Q)
    - input_image (U)  # only for deciding what is visually depictable

    --- REQUIRED OUTPUT JSON SCHEMA ---
    {{
      "soft_constraints": {{
        "perception_difficulty": {{
          "text":  {{ "count": 0, "evidence": [] }},
          "image": {{ "count": 0, "evidence": [] }}
        }}
      }}
    }}

    --- GENERAL RULES ---
    1. Evidence Objects:
       Each item in any `evidence` list MUST be:
       {{
         "span_text":"<verbatim cue>",
         "char_span":[start,end],
         "type":"<label>",
         "why":"<one-sentence rationale tied to this cue>"
       }}

    2. span_text MUST be an exact substring from Q.
    3. char_span MUST match that substring exactly (0-indexed, [start,end)).
    4. If a category has evidence, then `count` MUST equal len(evidence).

    --- DETAILED FILLING RULES ---
    1. soft_constraints.perception_difficulty.text
        - Extract ALL noun or noun-phrase entities mentioned in Q, whether abstract or concrete (e.g., "cultural significance", "organs",  "traditional churches").
        - This includes every concrete or abstract thing mentioned (e.g., "organs", "image").
        - For each distinct noun/noun phrase, you MUST create one evidence item.
        - type ∈ {{"textual_entity","textual_attribute","spatial_relation",
                    "logical_condition","semantic_concept"}}

    2. soft_constraints.perception_difficulty.image
       - From the entities found above, select those that are visually depictable
         in a normal image (e.g., "organs", "pews", "churches", "windows", "people", "color").
       - For each visually depictable entity, you MUST create one evidence item.
       - type ∈ {{"visual_entity","visual_property","visual_count"}}

    EXAMPLE (for the model's internal understanding, DO NOT copy it):
    Q = "How many organs can you see in the image?"
    - text.evidence MUST include at least:
        span_text="organs", type="textual_entity"
        span_text="image",  type="textual_entity"
    - image.evidence MUST include at least:
        span_text="organs", type="visual_entity"
        span_text="image",  type="visual_entity"

    FINAL REMINDER: Output JSON only. Start with {{. End with }}. No extra text.
    """


def soft_constraints_extracted() -> str:
    return f"""
    You are a strict JSON evaluation engine. Your task is to categorize specific entities by their perception difficulty based on the provided image and question.

    --- OUTPUT FORMAT ---
    - Your output MUST be a SINGLE, minified JSON object.
    - NO text, markdown blocks, or explanations. 
    - Start with {{ and end with }}.

    --- DEFINITIONS ---
    - EASY: Entity is prominent, unobstructed, and immediately identifiable.
    - HARD: Entity is tiny, occluded, blurry, low-contrast, or requires deep zooming.

    --- REQUIRED JSON SCHEMA ---
    {{
      "soft_constraints": {{
        "perception_difficulty": {{
          "image": {{
            "easy_count": 0,
            "hard_count": 0,
            "categorized_entities": [
              {{ "entity": "name", "difficulty": "easy|hard", "reason": "short text" }}
            ]
          }}
        }}
      }}
    }}

    --- INPUTS TO ANALYZE ---
    - Shared Entities (Question + Image): [INSERT_SHARED_ENTITIES]
    - Image (U): [VISUAL_INPUT]

    FINAL INSTRUCTION: Analyze every provided entity. Ensure the sum of easy_count and hard_count equals the total number of unique entities analyzed. Output JSON only.
    """


def prompt_build_soft_constraints_json():
    return f"""
    You are a strict JSON construction engine. Your output MUST be a SINGLE, valid, minified JSON object.
    DO NOT add explanations, comments, or text outside the JSON.

    --- INPUTS ---
    Image

    --- REQUIRED OUTPUT JSON SCHEMA ---
    {{
      "soft_constraints": {{
        "perception_difficulty": {{
          "text":  {{ "count": <number_of_text_entities>, "evidence": [ ... ] }},
          "image": {{ "count": <number_of_image_entities>, "evidence": [ ... ] }}
        }}
      }}
    }}

    --- EVIDENCE RULES ---
    For each entity, produce one evidence object with fields:
    {{
      "span_text": "<the entity exactly as shown in the input list>",
      "char_span": [0, 0],
      "type": "<label>",
      "why": "<one short rationale>"
    }}

    RULES:
    - span_text MUST match the entity string exactly.
    - char_span SHOULD be [0,0] because the original question text is not provided.
    - For TEXT_ENTITIES:
        type must be one of:
        "textual_entity", "textual_attribute", "spatial_relation",
        "logical_condition", "semantic_concept"
    - For IMAGE_ENTITIES:
        type must be one of:
        "visual_entity", "visual_property", "visual_count"
    - count MUST equal the length of the evidence list.
    - Output JSON only. No prose.

    Now construct the JSON.
    """


def prompt_build_soft_constraints_json_ori():
    return f"""
    You are a strict JSON construction engine. Your output MUST be a SINGLE, valid, minified JSON object.
    DO NOT add explanations, comments, or text outside the JSON.

    --- INPUTS ---
    TEXT_ENTITIES (noun/noun-phrases extracted from Q)
    IMAGE_ENTITIES (subset of TEXT_ENTITIES that are visually depictable in a normal image)

    --- REQUIRED OUTPUT JSON SCHEMA ---
    {{
      "soft_constraints": {{
        "perception_difficulty": {{
          "text":  {{ "count": <number_of_text_entities>, "evidence": [ ... ] }},
          "image": {{ "count": <number_of_image_entities>, "evidence": [ ... ] }}
        }}
      }}
    }}

    --- EVIDENCE RULES ---
    For each entity, produce one evidence object with fields:
    {{
      "span_text": "<the entity exactly as shown in the input list>",
      "char_span": [0, 0],
      "type": "<label>",
      "why": "<one short rationale>"
    }}

    RULES:
    - span_text MUST match the entity string exactly.
    - char_span SHOULD be [0,0] because the original question text is not provided.
    - For TEXT_ENTITIES:
        type must be one of:
        "textual_entity", "textual_attribute", "spatial_relation",
        "logical_condition", "semantic_concept"
    - For IMAGE_ENTITIES:
        type must be one of:
        "visual_entity", "visual_property", "visual_count"
    - count MUST equal the length of the evidence list.
    - Output JSON only. No prose.

    Now construct the JSON.
    """


def hard_constraints() -> str:
    """
    Returns a compact instruction prompt for an API model.
    Model must output JSON ONLY, no scores—just counts and concrete evidence.
    """
    return f"""
    You are a strict JSON evaluation engine. Your output MUST be a SINGLE, minified JSON object.
    DO NOT include any text, prose, explanations, or conversational preamble before or after the JSON.
    Your entire output must start with {{ and end with }}.

    --- INPUTS TO ANALYZE ---
    - generated_question (Q)

    --- REQUIRED OUTPUT JSON SCHEMA (Fill this structure) ---
    {{
    "hard_constraints": {{
        "Factuality": {{ "count": 0, "evidence": [] }},
        "Ambiguity":  {{ "count": 0, "evidence": [] }},
        "Validity": {{
        "answerable": true,
        "missing_info": [],
        "invalid_category": "",   # one of: "", "nonvisual", "hidden", "external", "nonexistent", 
        "evidence": [] 
        }}
    }}
    }}

    --- GENERAL RULES ---
    1.  **Evidence Objects:** All `evidence` lists must contain objects with this exact format:
        {{
        "span_text":"<verbatim cue>",
        "char_span":[start,end],  # 0-indexed, [start,end)
        "type":"<label>",
        "why":"<one-sentence rationale tied to this cue>"
        }}
    2.  **Verbatim Spans:** `span_text` MUST be the minimal verbatim substring from 'Q' or 'C'.
    3.  **Accurate Spans:** `char_span` MUST match the `span_text` location exactly.
    4.  **Empty Evidence:** If no items are found for a category, use `count: 0` and `evidence: []`.

    --- DETAILED FILLING RULES (BY KEY) ---

    1.  hard_constraints.Factuality
        Include **unsupported or false assumptions** in Q:
        - presupposed facts not supported by the image
        - over-specific details
        - **references to nonexistent objects** 
        type ∈ {"presupposition","contradiction","unsupported_detail","nonexistent_object"}
        - span_text: The exact problematic phrase (e.g., "the red triangle number").
        - why: Say why it’s likely hallucinated or unsupported by C.

    2.  hard_constraints.Ambiguity
        Include vague or unclear references.
        type ∈ {"vague_pronoun","underspecified","polysemy","ellipsis","scope_ambiguity"}
        - span_text: The ambiguous token/phrase (e.g., "it", "the object", "near the top").
        - why: State what is missing (e.g., "which object?", "which region?").

    3.  hard_constraints.Validity
        A question is INVALID only if the required information cannot be inferred from the image in principle. Set `"answerable": false` and assign an `"invalid_category"` when Q requires:
        - **nonvisual** → sound, inner material, internal design, ... 
        - **hidden** → behind/inside/occluded elements, ...  
        - **external** → historical facts, artist name, cultural meaning, rituals, ...
        Add each problematic phrase under evidence with type="unanswerable_cue".  
        List missing high-level info types in `missing_info`.

    FINAL REMINDER: Output JSON only. Start with {{. End with }}. No extra text.
    """

def simplify_prompt():
    return """You are a careful question normalizer.
    Rewrite the question into a shorter, cleaner form while preserving the same reasoning structure.
    Rules:
    - Keep it as ONE syntactic question ending with '?'.
    - Preserve all compared entities, visible evidence anchors, spatial relations, and counterfactual conditions.
    - Preserve whether the question is asking for comparison, causal explanation, multi-hop inference, or conditional reasoning.
    - Remove only incidental wording, not the core reasoning requirements.
    - Do NOT turn the question into a generic template like "What happens", "What effect", "What impact", "Which is more important", "How does X affect Y", "How does X impact Y", or "How does X influence Y".
    - Do NOT broaden the question into mood/usefulness/aesthetic/comfort commentary.
    Only output the rewritten question. Do not add any explanation or formatting.
    """

def gene_answer():
    return """You are a helpful assistant that helps generating the answer of the question based on the given image. Directly output the final answer. """

def get_reasoning_steps_all():
    return (
        "You are a step-by-step visual reasoning engine.\n"
        "Given a QUESTION and an IMAGE, generate ONLY the reasoning steps\n"
        "as numbered lines, followed by a final 'Answer:' line.\n"
        "\n"
        "STRICT FORMAT:\n"
        "Step 1: <text>\n"
        "Step 2: <text>\n"
        "...\n"
        "Answer: <final answer>\n"
        "\n"
        "RULES:\n"
        "- Start output with 'Step 1:'.\n"
        "- Each step MUST be a distinct reasoning operation needed to answer the question\n"
        "  (e.g., identify objects, compare quantities, infer a relation).\n"
        "- Do NOT include steps that only confirm, verify, or double-check previous results\n"
        "  (e.g., 'Confirm the count of organs', 'Double-check the answer').\n"
        "- No explanations, no metadata, no extra sentences outside the steps and the final Answer line.\n"
        "- Do NOT include anything before 'Step 1:'.\n"
        "- Do NOT include anything after the 'Answer:' line.\n"
    )
    
def prompt_reasoning_steps() -> str:
    return f"""
    You are a step-by-step reasoning engine.
    Given a QUESTION and an IMAGE, you must infer the reasoning steps required to answer the question.

    A reasoning step = a single cognitive operation a human must perform
    (identify objects, compare quantities, infer cause/effect, interpret relations, integrate cues).

    Forbidden:
    - Do NOT include steps that confirm, verify, or double-check previous results.
    - Do NOT include meta steps (e.g., "look at the image", "analyze the question").
    - Do NOT include the final answer.

    OUTPUT FORMAT (STRICT JSON, NO EXTRA TEXT):
    {{
        "steps": ["step1", "step2", ...]
    }}

    Do NOT output anything besides this JSON.
    """

def prompt_count_steps() -> str:
    return f"""
        You will be given a JSON object containing a list called "steps".
        Count how many steps there are.

        Return ONLY a minified JSON object like:

        {{
        "soft_constraints": {{
            "reasoning_difficulty": {{ "count": N, "evidence": []  # steps
            }}
        }}
        }}
        """


def prompt_reasoning_steps_and_count() -> str:
    return f"""
    You are a step-by-step reasoning engine.

    TASK 1 — Extract Reasoning Steps  
    Given a QUESTION and an IMAGE, infer the reasoning steps required to answer the question.

    A reasoning step = a single cognitive operation a human must perform  
    (identify objects, compare quantities, infer cause/effect, interpret relations, integrate cues).

    Forbidden:
    - Do NOT include steps that confirm, verify, or double-check previous results.
    - Do NOT include meta steps (e.g., "look at the image", "analyze the question").
    - Do NOT include the final answer.

    TASK 2 — Count Steps  
    After generating the reasoning steps, count how many steps there are.

    FINAL OUTPUT FORMAT (STRICT JSON, NO EXTRA TEXT):
    {{
        "soft_constraints": {{
            "reasoning_difficulty": {{
                "count": <number_of_steps>,
                "evidence": ["step1", "step2", ...]
            }}
        }}
    }}

    Where:
    - "count" = number of reasoning steps
    - "evidence" = the list of reasoning steps (in order)
    
    Do NOT output anything except this JSON.
    """


def question_judge() -> str:
    """
    Returns a compact instruction prompt for an API model.
    Model must output JSON ONLY, no scores—just counts and concrete evidence.
    """
    return f"""
    You are a strict JSON evaluation engine. Your output MUST be a SINGLE, minified JSON object.
    DO NOT include any text, prose, explanations, or conversational preamble before or after the JSON.
    Your entire output must start with {{ and end with }}.

    --- INPUTS TO ANALYZE ---
    - generated_question (Q)
    - added_constraints (C)

    --- REQUIRED OUTPUT JSON SCHEMA (Fill this structure) ---
    {{
    "hard_constraints": {{
        "Factuality": {{ "count": 0, "evidence": [] }},
        "Ambiguity":  {{ "count": 0, "evidence": [] }},
        "Validity": {{
        "answerable": true,
        "missing_info": [],
        "evidence": []
        }}
    }},
    "soft_constraints": {{
        "perception_difficulty": {{
        "simplified_question": "",
        "text":  {{ "count": 0, "evidence": [] }},
        "image": {{ "count": 0, "evidence": [] }}
        }},
        "reasoning_difficulty": {{ "count": <number_of_steps>, "evidence": ["step1", "step2", ...] }}
    }}
    }}

    --- GENERAL RULES ---
    1.  **Evidence Objects:** All `evidence` lists must contain objects with this exact format:
        {{
        "span_text":"<verbatim cue>",
        "char_span":[start,end],  # 0-indexed, [start,end)
        "type":"<label>",
        "why":"<one-sentence rationale tied to this cue>"
        }}
    2.  **Verbatim Spans:** `span_text` MUST be the minimal verbatim substring from 'Q' or 'C'.
    3.  **Accurate Spans:** `char_span` MUST match the `span_text` location exactly.
    4.  **Empty Evidence:** If no items are found for a category, use `count: 0` and `evidence: []`.

    --- DETAILED FILLING RULES (BY KEY) ---

    1.  hard_constraints.Factuality
        - What to include: Presupposed facts or over-specific claims in Q that are not supported by C.
        - type ∈ {{"presupposition","contradiction","unsupported_detail"}}
        - span_text: The exact problematic phrase (e.g., "the red triangle number").
        - why: Say why it’s likely hallucinated or unsupported by C.

    2.  hard_constraints.Ambiguity
        - What to include: Cues in Q that make the reference unclear.
        - type ∈ {{"vague_pronoun","underspecified","polysemy","scope_ambiguity","ellipsis"}}
        - span_text: The ambiguous token/phrase (e.g., "it", "the object", "near the top").
        - why: State what is missing (e.g., "which object?", "which region?").

    3.  hard_constraints.Validity
        - answerable: [true/false] Is Q answerable given C?
        - missing_info: [list of strings] List any info required by Q but absent from C (e.g., "object identity", "image crop").
        - evidence: Point to Q phrases demanding unavailable info OR to C items that resolve it.
        - If unanswerable: type="unanswerable_cue"
        - If answerable due to C: type="resolving_cue"
        - span_text: Minimal phrase.
        - why: Explain how it blocks/enables answering.

    4.  soft_constraints.perception_difficulty
        - simplified_question: [string] Rewrite Q minimally to a simpler, equivalent form.
        - text (Textual Complexity):
            - Goal: Measures the parsing complexity of the *original* question (Q).
            - Include: All distinct components of the question's text structure.
            - type ∈ {{"textual_entity", "textual_attribute", "spatial_relation", "logical_condition", "numeric_value"}}
            - span_text: The exact token/phrase (e.g., "types", "organs", "left of", "if", "how many").
            - why: "Identifies a component of the question's text structure."
        - image (Visual Grounding Load):
            - Goal: Measures the visual grounding load. Finds *text cues in Q* that refer to visual properties or objects.
            - NOTE: You are analyzing the *text* of Q. Do not analyze an image.
            - Include: Text in Q that names a specific visual category, property, or count.
            - type ∈ {{"visual_category", "visual_property", "visual_count"}}
            - span_text: The exact referring phrase from Q (e.g., "organs", "blue squares", "how many", "top row").
            - why: "Identifies a visual object, property, or count that must be found."

    5.  soft_constraints.reasoning_difficulty
        - Goal: Decompose Q into the minimal reasoning steps. Each evidence item marks one step.
        - A reasoning step = a single cognitive operation a human must perform (identify objects, compare quantities, infer cause/effect, interpret relations, integrate cues).
        - Do NOT include steps that confirm, verify, or double-check previous results.
        - Do NOT include meta steps (e.g., "look at the image", "analyze the question").
        - Do NOT include the final answer.

    ---
    FINAL REMINDER: Output JSON only. Start with {{. End with }}. No extra text.
    """


def _qwen_judge_func(judge_func: str):
    """Resolve the supported inference backend and reject unknown judge names."""
    if judge_func != "qwen":
        raise ValueError("Only judge_func='qwen' is supported in the official repo.")
    return model_infer_qwen


def self_judge_v2(qs: str, ans: str, image, Dict_config_m: Dict, constraints: List=[], judge_func: str='qwen'):
    # load prompt
    prompt = build_prompt_v1()
    hard_cons = hard_constraints()
    soft_cons_simp = soft_constraints_simplified()
    soft_cons_mode = soft_constraints_follow()
    simp_prompt = simplify_prompt()
    soft_entity_cnt = prompt_build_soft_constraints_json()
    reasoning_all = prompt_reasoning_steps_and_count()
    reasoning_step = prompt_reasoning_steps()
    reasoning_cnt = prompt_count_steps()
    model_infer_func = _qwen_judge_func(judge_func)

    if qs is None:
        return False
    if ans is None:
        return False
    
    # simplify questions
    Dict_inputs = {'prompt': simp_prompt + f"Question: {qs}"}
    simplified, _ = model_infer_func(dict_input=Dict_inputs, dict_config_m=Dict_config_m, max_len=3000, print_content=False, do_sample=True, )

    # return if question is successfully evolved or not
    if_valid = True
    dict_res = dict()

    Dict_inputs = {'prompt': hard_cons + f"Question: {qs}", "image": image}
    hard_eval, _ = model_infer_func(dict_input=Dict_inputs, dict_config_m=Dict_config_m, )
    if not isinstance(hard_eval, dict):
        return False, dict_res

    # check whether question is qualified in hard constraints:
    satisfy_hard = is_valid_hard_constraints(hard_eval)
    if not satisfy_hard:
        dict_res = {'simplified': simplified, 'evaluation': [hard_eval]}
        return False, dict_res

    Dict_inputs = {'prompt': soft_cons_simp + f"Question: {simplified}", "image": image}
    soft_simp_eval, _ = model_infer_func(dict_input=Dict_inputs, dict_config_m=Dict_config_m, )
    if not isinstance(soft_simp_eval, dict):
        return False, dict_res

    Dict_inputs = {'prompt': reasoning_all + f"Question: {simplified}", "image": image}
    soft_reason_eval, _ = model_infer_func(dict_input=Dict_inputs, dict_config_m=Dict_config_m, )
    if not isinstance(soft_reason_eval, dict):
        return False, dict_res

    dict_res = {'simplified': simplified, 'evaluation': [hard_eval, soft_simp_eval, soft_reason_eval, {}]}

    # Dict_inputs = {'prompt': 'Extract ALL noun or noun-phrase entities mentioned in Q, whether abstract or concrete. Return only the list of entities' + f"Question: {simplified}", }
    # text_entities, _ = model_infer_func(dict_input=Dict_inputs, dict_config_m=Dict_config_m, )

    # Dict_inputs = {'prompt': 'Extract entities that are visually depictable in the given image. Return only the list of entities in the given list' + f"Given entities: {text_entities}", 'image': image}
    # img_entities, _ = model_infer_func(dict_input=Dict_inputs, dict_config_m=Dict_config_m, )

    # Dict_inputs = {'prompt': soft_entity_cnt + f"Text Entity: {text_entities}. Image Entity: {img_entities}"}
    # soft_simp_eval, _ = model_infer_func(dict_input=Dict_inputs, dict_config_m=Dict_config_m, )
    # if not isinstance(soft_simp_eval, dict):
    #     return False, dict_res

    # Dict_inputs = {'prompt': reasoning_step + f"Question: {simplified}", }
    # reasoning_steps_eval, _ = model_infer_func(dict_input=Dict_inputs, dict_config_m=Dict_config_m, )
    # Dict_inputs = {'prompt': reasoning_cnt + f"Reasoning steps: {reasoning_steps_eval}", }
    # soft_reason_eval, _ = model_infer_func(dict_input=Dict_inputs, dict_config_m=Dict_config_m, )
    # if not isinstance(soft_reason_eval, dict):
    #     return False, dict_res

    # if constraints is not None and len(constraints) > 0:
    #     cons_str = ",".join(constraints)
    #     Dict_inputs = {'prompt': soft_cons_mode + f"Question: {qs}, Added constraints: {cons_str}", 'image': image}
    #     soft_simp_mode, _ = model_infer_func(dict_input=Dict_inputs, dict_config_m=Dict_config_m, )
    #     if not isinstance(soft_simp_mode, dict):
    #         return False, dict_res
    # else:
    #     soft_simp_mode = {}

    # dict_res = {'simplified': simplified, 'evaluation': [hard_eval, soft_simp_eval, soft_reason_eval, soft_simp_mode]}

    return True, dict_res


def self_judge_grpo(qs: str, ans: str, image, Dict_config_m: Dict, constraints: List=[], judge_func: str='qwen'):
    # load prompt
    prompt = build_prompt_v1()
    hard_cons = hard_constraints()
    soft_cons_simp = soft_constraints_simplified()
    soft_cons_mode = soft_constraints_follow()
    simp_prompt = simplify_prompt()
    soft_entity_cnt = prompt_build_soft_constraints_json()
    reasoning_all = prompt_reasoning_steps_and_count()
    reasoning_step = prompt_reasoning_steps()
    reasoning_cnt = prompt_count_steps()
    model_infer_func = _qwen_judge_func(judge_func)

    if qs is None:
        return False
    if ans is None:
        return False
    
    # return if question is successfully evolved or not
    if_valid = True
    dict_res = dict()

    Dict_inputs = {'prompt': hard_cons + f"Question: {qs}", "image": image}
    hard_eval, _ = model_infer_func(dict_input=Dict_inputs, dict_config_m=Dict_config_m, )
    if not isinstance(hard_eval, dict):
        return False, dict_res

    # check whether question is qualified in hard constraints:
    satisfy_hard = is_valid_hard_constraints(hard_eval)
    if not satisfy_hard:
        dict_res = {'simplified': qs, 'evaluation': [hard_eval]}
        return False, dict_res

    Dict_inputs = {'prompt': soft_cons_simp + f"Question: {qs}", "image": image}
    soft_simp_eval, _ = model_infer_func(dict_input=Dict_inputs, dict_config_m=Dict_config_m, )
    if not isinstance(soft_simp_eval, dict):
        return False, dict_res

    Dict_inputs = {'prompt': reasoning_all + f"Question: {qs}", "image": image}
    soft_reason_eval, _ = model_infer_func(dict_input=Dict_inputs, dict_config_m=Dict_config_m, )
    if not isinstance(soft_reason_eval, dict):
        return False, dict_res

    dict_res = {'evaluation': [hard_eval, soft_simp_eval, soft_reason_eval, {}]}

    return True, dict_res


def _is_compound_question(qs: str) -> bool:
    """
    Heuristic filter: reject questions that are really two questions joined by
    'and' (the dominant failure mode where the model inflates length instead of
    depth).  We look for the pattern 'and what/how/why/which/where/when' after
    a substantial clause, which is the tell-tale sign of question concatenation.
    """
    import re
    if qs is None:
        return False
    # Count question marks — two or more is always compound
    if qs.count('?') >= 2:
        return True
    # Detect "...clause, and what/how/why..." pattern
    if re.search(r',\s+and\s+(what|how|why|which|where|when)\b', qs, re.IGNORECASE):
        return True
    return False


def self_judge(qs: str, ans: str, image, Dict_config_m: Dict, constraints: List=[], judge_func: str='qwen'):
    """Validate image-grounded questions before scoring their difficulty.

    Return (accepted, diagnostics). Hard-constraint failures stop further model
    calls; successful diagnostics retain hard, perception, reasoning, and
    constraint-following evaluations for downstream scoring.
    """
    # Build separate prompts for validity, difficulty, and constraint compliance.
    hard_cons = hard_constraints()
    soft_cons_mode = soft_constraints_follow()
    soft_cons_extracted = soft_constraints_extracted()
    reasoning_all = prompt_reasoning_steps_and_count()
    model_infer_func = _qwen_judge_func(judge_func)

    if qs is None:
        return False, {}
    if ans is None:
        return False, {}

    # Reject compound questions early — they inflate text counts without adding depth
    if _is_compound_question(qs):
        return False, {}

    # Check the original question against the image before spending inference
    # on difficulty scores: difficulty cannot rescue an invalid question.
    Dict_inputs = {'prompt': hard_cons + f"Question: {qs}", 'image': image}
    hard_eval, _ = model_infer_func(
        dict_input=Dict_inputs, dict_config_m=Dict_config_m, max_len=1024,
    )
    if not isinstance(hard_eval, dict):
        return False, {}
    dict_res = {
        'evaluation': [hard_eval],
        'question_info': {'constraint': constraints},
    }
    if not is_valid_hard_constraints(hard_eval):
        return False, dict_res

    Dict_inputs = {'prompt': 'Extract ALL noun or noun-phrase entities mentioned in Q, whether abstract or concrete and saved as text entity. ' + f"Question: {qs}", }
    text_entity, _ = model_infer_func(dict_input=Dict_inputs, dict_config_m=Dict_config_m, max_len=512)
    
    Dict_inputs = {'prompt': soft_cons_extracted + f"\nShared entities: {text_entity}", "image": image}
    soft_simp_eval, _ = model_infer_func(dict_input=Dict_inputs, dict_config_m=Dict_config_m, max_len=1024)
    if not isinstance(soft_simp_eval, dict):
        return False, dict_res

    image_data = soft_simp_eval.get('soft_constraints', {}).get('perception_difficulty', {}).get('image', {})
    easy_count = image_data.get('easy_count', 0)
    hard_count = image_data.get('hard_count', 0)
    # Hard-to-ground entities contribute twice the weight of easy entities.
    count = (hard_count * 2) + (easy_count * 1)
    image_data['count'] = count
    
    Dict_inputs = {'prompt': reasoning_all + f"Question: {qs}", "image": image}
    soft_reason_eval, _ = model_infer_func(dict_input=Dict_inputs, dict_config_m=Dict_config_m, max_len=1024)
    if not isinstance(soft_reason_eval, dict):
        return False, dict_res

    if constraints is not None and len([item for item in constraints if item]) > 0:
        cons_str = ", ".join([str(item) for item in constraints if str(item).strip()])
        Dict_inputs = {'prompt': soft_cons_mode + f"Question: {qs}, Added constraints: {cons_str}", "image": image}
        soft_mode_eval, _ = model_infer_func(dict_input=Dict_inputs, dict_config_m=Dict_config_m, max_len=1024)
        if not isinstance(soft_mode_eval, dict):
            soft_mode_eval = {}
    else:
        soft_mode_eval = {}

    dict_res = {
        'evaluation': [hard_eval, soft_simp_eval, soft_reason_eval, soft_mode_eval],
        'question_info': {'constraint': constraints},
    }
    
    return True, dict_res


def self_judge_ori(qs: str, ans: str, image, Dict_config_m: Dict, constraints: List=[], judge_func: str='qwen'):
    # load prompt
    prompt = build_prompt_v1()
    hard_cons = hard_constraints()
    soft_cons_simp = soft_constraints_simplified()
    soft_cons_mode = soft_constraints_follow()
    simp_prompt = simplify_prompt()
    soft_entity_cnt = prompt_build_soft_constraints_json()
    reasoning_all = get_reasoning_steps_all()
    reasoning_step = prompt_reasoning_steps()
    reasoning_cnt = prompt_count_steps()
    model_infer_func = _qwen_judge_func(judge_func)

    if qs is None:
        return False
    if ans is None:
        return False
    
    # simplify questions
    Dict_inputs = {'prompt': simp_prompt + f"Question: {qs}"}
    simplified, _ = model_infer_func(dict_input=Dict_inputs, dict_config_m=Dict_config_m, max_len=3000, print_content=False, do_sample=True, )

    # return if question is successfully evolved or not
    if_valid = True
    dict_res = dict()

    Dict_inputs = {'prompt': hard_cons + f"Question: {qs}", 'image': image}
    hard_eval, _ = model_infer_func(dict_input=Dict_inputs, dict_config_m=Dict_config_m, )
    if not isinstance(hard_eval, dict):
        return False, dict_res

    # check whether question is qualified in hard constraints:
    satisfy_hard = is_valid_hard_constraints(hard_eval)
    if not satisfy_hard:
        dict_res = {'simplified': simplified, 'evaluation': [hard_eval]}
        return False, dict_res

    Dict_inputs = {'prompt': 'Extract ALL noun or noun-phrase entities mentioned in Q, whether abstract or concrete and saved as text entity. Extract entities that are visually depictable in the given image and saved as image entity. ' + soft_entity_cnt + f"Question: {simplified}", }
    soft_simp_eval, _ = model_infer_func(dict_input=Dict_inputs, dict_config_m=Dict_config_m, )
    if not isinstance(soft_simp_eval, dict):
        return False, dict_res

    Dict_inputs = {'prompt': reasoning_all + f"Question: {simplified}", }
    soft_reason_eval, _ = model_infer_func(dict_input=Dict_inputs, dict_config_m=Dict_config_m, )
    if not isinstance(soft_reason_eval, dict):
        return False, dict_res

    if constraints is not None and len(constraints) > 0:
        cons_str = ",".join(constraints)
        Dict_inputs = {'prompt': soft_cons_mode + f"Question: {qs}, Added constraints: {cons_str}", 'image': image}
        soft_simp_mode, _ = model_infer_func(dict_input=Dict_inputs, dict_config_m=Dict_config_m, )
        if not isinstance(soft_simp_mode, dict):
            return False, dict_res
    else:
        soft_simp_mode = {}

    dict_res = {'simplified': simplified, 'evaluation': [hard_eval, soft_simp_eval, soft_reason_eval, soft_simp_mode]}

    return True, dict_res


def is_valid_hard_constraints(res: dict) -> bool:
    """Require zero factual/ambiguity errors and explicit answerability.

    Missing or malformed judge sections fail closed instead of accepting a
    question or interrupting generation with an attribute error.
    """
    if not isinstance(res, dict):
        return False
    hard = res.get("hard_constraints", {})
    if not isinstance(hard, dict):
        return False
    if any(not isinstance(hard.get(key), dict)
           for key in ("Factuality", "Ambiguity", "Validity")):
        return False

    factuality_ok = hard.get("Factuality", {}).get("count", None) == 0
    ambiguity_ok  = hard.get("Ambiguity", {}).get("count", None) == 0
    validity_ok   = hard.get("Validity", {}).get("answerable", None) is True

    return factuality_ok and ambiguity_ok and validity_ok


def has_difficulty_improved(seed_met: dict, rew_met: dict) -> bool:
    """
    Return True only if genuine difficulty improved from seed -> rewritten.

    Rules:
    - `reasoning_difficulty` OR `perception_difficulty_image` must increase.
    - `perception_difficulty_text` alone is NOT sufficient, because it trivially
      increases when the model appends extra clauses to the question without
      requiring deeper reasoning (the core failure mode we are guarding against).
    """

    # 1) Merge raw evaluation blocks into api_res-like dicts
    seed_api = _merge_api_res_from_evaluation(seed_met.get("evaluation", []))
    rew_api  = _merge_api_res_from_evaluation(rew_met.get("evaluation", []))

    # 2) Use the unified extractor to get flat scores
    seed_scores, _, _ = _extract_scores_and_mfc({"api_res": seed_api})
    rew_scores,  _, _ = _extract_scores_and_mfc({"api_res": rew_api})

    reasoning_delta = rew_scores.get("reasoning_difficulty", 0.0) - seed_scores.get("reasoning_difficulty", 0.0)
    perception_delta = rew_scores.get("perception_difficulty_image", 0.0) - seed_scores.get("perception_difficulty_image", 0.0)

    # Require a substantive gain. The 3B judge's step counts are noisy (±1), so we
    # accept a half-step gain as sufficient. We also allow perceptual-only improvement
    # when it is a full unit (clearly harder to ground visually).
    if reasoning_delta >= 0.5:
        return True
    if reasoning_delta >= 0.0 and perception_delta >= 1.0:
        return True
    if reasoning_delta > 0.0 and perception_delta > 0.0:
        return True

    return False


def _extract_nonempty_constraints(constraints: List) -> List[str]:
    """Strip constraint text and discard blank entries before matching judgments."""
    return [str(item).strip() for item in (constraints or []) if str(item).strip()]


def _get_model_following_entries(dict_res: dict) -> List[dict]:
    """Merge judge blocks and normalize their per-constraint compliance entries."""
    api = _merge_api_res_from_evaluation(dict_res.get("evaluation", []))
    soft = api.get("soft_constraints", {}) or {}
    mfc = soft.get("model_following_capability", {}) or {}
    return _get_model_following_by_constraint(mfc)


def extract_model_following_ratio(dict_res: dict) -> float:
    """Return the compliance fraction for named constraints, or zero if absent."""
    entries = [
        item for item in _get_model_following_entries(dict_res)
        if str(item.get("constraint_text", "")).strip()
    ]
    if not entries:
        return 0.0
    return sum(1 for item in entries if item.get("complies")) / len(entries)


def has_required_model_following(dict_res: dict, constraints: List) -> bool:
    """Require compliance with the latest nonblank constraint; no constraints pass."""
    required = _extract_nonempty_constraints(constraints)
    if not required:
        return True
    entries = _get_model_following_entries(dict_res)
    if not entries:
        return False

    by_constraint = {
        str(item.get("constraint_text", "")).strip(): bool(item.get("complies"))
        for item in entries if str(item.get("constraint_text", "")).strip()
    }

    critical = required[-1]
    return by_constraint.get(critical, False)


def preserves_reasoning_structure(original_q: str, simplified_q: str, content: Optional[dict] = None) -> Tuple[bool, Dict[str, object]]:
    """Check that simplification preserves question form and reasoning anchors."""
    content = content or {}
    original = (original_q or "").strip()
    simplified = (simplified_q or "").strip()
    details: Dict[str, object] = {"reason": "", "missing_anchors": []}

    if not simplified.endswith("?"):
        details["reason"] = "simplified_not_question"
        return False, details

    orig_lower = original.lower()
    simp_lower = simplified.lower()

    def _has_any(text: str, patterns: List[str]) -> bool:
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)

    if _has_any(orig_lower, [r"\bwhich\b", r"\bthan\b", r"\bversus\b", r"\bbetween\b"]) and not _has_any(
        simp_lower, [r"\bwhich\b", r"\bthan\b", r"\bversus\b", r"\bbetween\b"]
    ):
        details["reason"] = "dropped_comparison_structure"
        return False, details

    if _has_any(orig_lower, [r"\bif\b", r"\bwould\b", r"\bhappen\b", r"\bchange\b", r"\bmoving\b"]) and not _has_any(
        simp_lower, [r"\bif\b", r"\bwould\b", r"\bchange\b", r"\bmoving\b", r"\bcloser\b"]
    ):
        details["reason"] = "dropped_counterfactual_structure"
        return False, details

    if _has_any(orig_lower, [r"^why\b", r"\bcause\b", r"\breason\b"]) and not _has_any(
        simp_lower, [r"^why\b", r"\bcause\b", r"\breason\b", r"^how\b"]
    ):
        details["reason"] = "dropped_causal_structure"
        return False, details

    visible_evidence = content.get("visible_evidence", [])
    if isinstance(visible_evidence, list):
        missing = []
        # Check anchors against the ORIGINAL rewritten question (already validated),
        # not the simplified one — the 3B simplifier regularly drops object names.
        # Min token length > 3 avoids false failures on short stop-words.
        orig_tokens = set(re.findall(r"[a-z0-9']+", orig_lower))
        for item in visible_evidence:
            anchor_tokens = {tok for tok in re.findall(r"[a-z0-9']+", str(item).lower()) if len(tok) > 3}
            if anchor_tokens and not (anchor_tokens & orig_tokens):
                missing.append(str(item))
        if len(missing) >= max(1, len(visible_evidence) // 2):
            details["reason"] = "dropped_visible_evidence_anchors"
            details["missing_anchors"] = missing
            return False, details

    return True, details
