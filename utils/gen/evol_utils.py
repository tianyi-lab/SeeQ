from typing import Dict, List, Tuple, Optional
import json
import os
import re
from tqdm import tqdm
import random
from gen.scripts.qwen_unified import *

ROOT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

question_constraints = (
    "The question must rely on visual information such that its answer would change or be impossible to determine without the image. "
    "The answer should be deterministic and answerable with a few sentences. "
    "You should generate only ONE focused question, not multiple sub-questions. "
    "IMPORTANT: Do NOT create compound or multi-part questions by joining sub-questions with 'and' or 'what'. "
    "A question like 'How does X affect Y, and what in the image shows Z?' is FORBIDDEN — it is two easy questions concatenated, not a harder one. "
    "Instead, make the SINGLE question harder by requiring deeper reasoning: "
    "multi-hop inference, counterfactual/causal reasoning, cross-region comparison, or quantitative/ordinal judgment. "
    "The answer must be DIFFERENT from the original answer. "
)

# question_condition = (
#     "You are a Question generator. Your objective is to rewrite a given Q&A into a genuinely harder version. "
#     "Make it harder by increasing the DEPTH of reasoning required — for example: "
#     "(1) require comparison of two or more objects/regions; "
#     "(2) require visual perception or reasoning related to relationship recognition of two or more objects/regions; "
#     # "(2) require counterfactual or causal reasoning ('why', 'what would happen if', 'what caused'); "
#     "(3) require multi-hop inference (derive an attribute from another inferred attribute); "
#     "Do NOT make it harder by simply appending more sub-questions or extra clauses. "
#     "Do NOT generate a question that asks the same things as the original or is just a paraphrase. "
#     "The added constraint must change the type or depth of reasoning required, not just expand the question's length. "
# )

question_condition = (
    "You are a Question generator. Your objective is to rewrite a given Q&A into a genuinely harder version that require more visual evidence or reasoning steps. "
    "Make it harder by increasing the DEPTH of reasoning required — for example: "
    "(1) Evidence selection: Which visible evidence best supports that this area is used for work?"
    "(2) Object-role comparison: Which object is more central to the workspace function, the laptop or the apple?"
    "(3) Spatial relation: What does the chair’s placement relative to the desk suggest about the room’s use?"
    "(4) Scene-function inference: What visible objects indicate that this is a study/work area?"
    "(5) Multi-object grounding: How do the laptop, chair, and notebook together suggest the purpose of this space?"
    "Do NOT make it harder by simply appending more sub-questions or extra clauses. "
    "Do NOT generate a question that asks the same things as the original or is just a paraphrase. "
    "The question should not become broader, more subjective, or more generic."
    "The added constraint must change the type or depth of reasoning required, not just expand the question's length. "
    "Do NOT generate questions using the template 'How does X affect/impact/influence Y' — "
    "these produce world-knowledge questions that do not require looking at the image. "
)

gen_format = "Return result in a json format: {'Question': 'xxx', 'Thinkings': 'Your selection process'}\n"
rewrite_format = "Return the result in a JSON format: {'Rewritten question': 'xxx', 'Added information': 'xxxx', 'Modification': '<what modification is done>'}\n"

# question_constraints = "The question must rely on visual information such that its answer would change or be impossible to determine without the image. The question should be able to answer with a short, specific phrase or sentence rather than a detailed analysis / explanation / description. The added constraints should be involved in question itself, not only the background description. The new answer should be DIFFERENT with the original answer. "

list_booster = ["Be creative", "Be different", "Be smart", "Be weird", "Don’t ask the first thing you think of", "Be creative and don’t ask the first thing you think of"]

genqa_prompt_v1 = "List n1 topics that you can answer questions about. Choose a topic uniformly from this list, and state it. Then write 60 subtopics about the chosen topic. Then choose a subtopic uniformly from this list, and state it. Then write a question that is not about the subtopic, but can only be answered with expertise in the subtopic. Then write the answer. Both the question and answer should be long. The name of the subtopic should not appear in the question, and none of the words in subtopic should be reused in the question. " + question_constraints + gen_format

genqa_prompt_v2 = "You will be given a list of objects in the image. Based on these objects and the image, list n1 objects that you can answer questions about. Choose a object uniformly from this list, and state it. Then write 10 related attributes about the chosen object. Then choose a attributes uniformly from this list, and state it. Then write a question that is not about the attributes, but can only be answered with expertise in the attributes. Then write the answer. Both the question and answer should be long. The name of the attributes should not appear in the question, and none of the words in attributes should be reused in the question. " + question_constraints + gen_format

generate_prompt_v1 = "You will be given an image. Based on the image, generate a question that use this image can answer. " + question_constraints + gen_format

generate_prompt_v2 = "You will be given an image. Based on the image, generate a question that focus on the objects, actions, and spatial relationships shown in this image. " + question_constraints + gen_format

envolve_mmevol_v1 = question_condition + """This new created Q&A should belong to the same domain as the given Q&A but be even more rare. The difficulty level of the created Q&A should be similar to that of the given Q&A. Prioritize questions with definite answers. If a question can be resolved with only a few solving steps, it can bereformulated to explicitly request additional solving steps. It is essential to avoid making the #Rewritten Q&Aoverly verbose. ## Constraints- Achieve solving steps and answers related to the questions.- Ensure all generated data is consistent with the image content. - Double-check provided descriptions against the image content. - """ + question_constraints + rewrite_format

add_object_v1_random = question_condition + """You will be given a list of objects in the image. You SHOULD complicate the given Q&A using the following method, but not limited to it:
First, select 5 visual objects that you can ask questions about. Choose one object uniformly from this list and state it. Then write 10 related attributes about the chosen object. Choose one attribute uniformly from this list and state it. In the rewritten problem, include 1–2 new visual objects and their attributes while avoiding making the problem unnecessarily lengthy. If a problem can be solved in just a few steps, rewrite the problem by adding new constraints and requirements to increase the number of reasoning steps.
## Constraints
- Achieve solving steps and answers related to the question.
- Ensure all generated data is consistent with the image content.
- Double-check provided descriptions against the image content.
- """ + question_constraints + rewrite_format

add_object_v2_random = question_condition + """You will be given a list of objects in the image. You SHOULD complicate the given Q&A using the following method, but not limited to: First, select 5 visual objects that you can ask questions about. Choose a object uniformly from this list, and state it. Then write 10 related atributes about the chosen object. Then choose a atributes uniformly from this list, and state it. In the rewritten problem, include 1-2 new visual object & itr attributes whileavoiding making the problem unnecessarily lengthy. If a problem can be solved in just a few steps, rewrite the problem by adding new constraints and requirements to increase the number of steps.## Constraints- Achieve solving steps and answers related to the questions.- Ensure all generated data is consistent with the image content. - Double-check provided descriptions against the image content. - """ + question_constraints + rewrite_format

add_object_v1_specific = question_condition + """You will be given a visual object in the image and the relationship of the new object and the previous question. You SHOULD rewrite a new Q&A with the original one and the new object based on the relationship and image. The new Q&A should have the same or higher reasoning difficulty. ## Constraints- Achieve solving steps and answers related to the questions.- Ensure all generated data is consistent with the image content. - Double-check provided descriptions against the image content. - """ + question_constraints + rewrite_format

add_task_v1_random = question_condition + """You will be given a list of tasks that test different capabilities. You SHOULD complicate the given Q&A using the following method, but not limited to it: In the rewritten problem, make sure the question and its reasoning steps test one task capability in the list, while avoiding making the problem unnecessarily lengthy. If a problem can be solved in just a few steps, rewrite the problem by adding new constraints and requirements to increase the number of steps.
## Constraints
- Achieve solving steps and answers related to the question.
- Ensure all generated data is consistent with the image content.
- Double-check provided descriptions against the image content.
- """ + question_constraints + rewrite_format

add_task_v1_specific = question_condition + """You will be given a specific task that test perception or reasoning capabilities. You SHOULD complicate the given Q&A using the following method, but not limited to it: In the rewritten problem, make sure the question and its reasoning steps test the task capability in the list, while avoiding making the problem unnecessarily lengthy. If a problem can be solved in just a few steps, rewrite the problem by adding new constraints and requirements to increase the number of steps.
## Constraints
- Achieve solving steps and answers related to the question.
- Ensure all generated data is consistent with the image content.
- Double-check provided descriptions against the image content.
- """ + question_constraints + rewrite_format

add_area_v1_random = question_condition + """You will be given a list of positions in the image. You SHOULD complicate the given Q&A using the following method, but not limited to it: In the rewritten problem, make sure the question and its reasoning steps relies on the information from the selected area of the image, while avoiding making the problem unnecessarily lengthy. If a problem can be solved in just a few steps, rewrite the problem by adding new constraints and requirements to increase the number of reasoning steps.
## Constraints
- Achieve solving steps and answers related to the question.
- Ensure all generated data is consistent with the image content.
- Double-check provided descriptions against the image content.
- """ + question_constraints + rewrite_format


add_area_v1_specific = question_condition + """You will be given a location. You SHOULD complicate the given Q&A using the following method, but not limited to it: In the rewritten problem, make sure the question and its reasoning steps relies on the information from the given area of the image, while avoiding making the problem unnecessarily lengthy. If a problem can be solved in just a few steps, rewrite the problem by adding new constraints and requirements to increase the number of reasoning steps.
## Constraints
- Achieve solving steps and answers related to the question.
- Ensure all generated data is consistent with the image content.
- Double-check provided descriptions against the image content.
- """ + question_constraints + rewrite_format

# --- Deep-reasoning evolution prompts ---
add_causal_v1 = question_condition + """Rewrite the given Q&A so that the question requires CAUSAL or COUNTERFACTUAL reasoning.
The rewritten question must ask WHY something happened, WHAT CAUSED an observed state, or WHAT WOULD CHANGE if a visible element were different.
Do NOT just append a second question. Transform the original question into one that requires causal inference over the image.
## Constraints
- The answer must require multi-step causal inference, not direct observation.
- The new answer must differ from the original answer.
- Ensure all generated data is consistent with the image content.
- Double-check provided descriptions against the image content.
- """ + question_constraints + rewrite_format

add_comparison_v1 = question_condition + """Rewrite the given Q&A so that the question requires CROSS-OBJECT or CROSS-REGION COMPARISON.
The rewritten question must ask the model to compare two or more distinct objects, regions, or attributes visible in the image and make a judgment (e.g., which is larger, which has more X, how do they differ in Y).
Do NOT just append a second question. Fuse the comparison into one focused question.
## Constraints
- The question must name or reference at least two distinct entities to compare.
- The answer must require identifying both entities and evaluating their difference/similarity.
- The new answer must differ from the original answer.
- Ensure all generated data is consistent with the image content.
- Double-check provided descriptions against the image content.
- """ + question_constraints + rewrite_format

add_multihop_v1 = question_condition + """Rewrite the given Q&A so that the question requires MULTI-HOP INFERENCE.
The rewritten question must require the model to first infer an intermediate visual fact, and then use that intermediate fact to answer the final question. The final answer cannot be obtained without completing the intermediate step.
Do NOT just append a second question. Design one question whose answer requires two or more inference hops.
## Constraints
- The question should not give away the intermediate fact directly.
- The answer must demonstrate the multi-hop chain.
- The new answer must differ from the original answer.
- Ensure all generated data is consistent with the image content.
- Double-check provided descriptions against the image content.
- """ + question_constraints + rewrite_format



def generate_instructions_v2(image, seed_question: str, seed_answer: str, list_task: List, task_str: str, dict_definition: Dict,  list_prev_const: List, Dict_config_m: Dict, num_iter: int=3):

    list_instructions = []
    list_constraints = []
    list_prompt = []

    # simplify questions
    Dict_inputs = {'prompt': "Find 3 NEW objects that is relavent but NOT included in the given QA pair based on the image. Return the result in a list of pair format: [(new object, relationship with original QA), (...), ] " + f"Question: {seed_question} \t Answer: {seed_answer}", "image": image, }
    related_obj, _ = model_infer_qwen(dict_input=Dict_inputs, dict_config_m=Dict_config_m, max_len=1024, print_content=False, do_sample=True, )
    try:
        pattern = r"\(([^,]+),\s*([^)]+)\)"
        list_matches = re.findall(pattern, related_obj)
    except Exception as e:
        list_matches = []
    
    seed_question_norm = re.sub(r"[^a-z0-9\s]", " ", seed_question.lower())
    seed_question_norm = " ".join(seed_question_norm.split())
    filtered_matches = []
    for item in list_matches:
        obj = item[0].strip().strip("'\"")
        if not obj:
            continue
        obj_norm = re.sub(r"[^a-z0-9\s]", " ", obj.lower())
        obj_norm = " ".join(obj_norm.split())
        if obj_norm and obj_norm not in seed_question_norm:
            filtered_matches.append((obj, item[1].strip()))

    # rewrite object (breadth strategy: add a new related object)
    for _ in range(min(len(filtered_matches), num_iter)):
        temp_pair = random.choice(filtered_matches)
        list_instructions.append(add_object_v1_specific + f"\nQuestion: {seed_question}\t Answer:{seed_answer}\n object: {temp_pair[0]} \n relation with previous question: {temp_pair[1]}")
        list_constraints.append(temp_pair[0])
        list_prompt.append('add_object_v1_specific')

    # task (breadth strategy: add a new task type)
    list_instructions.append(add_task_v1_random + f"\nQuestion: {seed_question}\t Answer:{seed_answer}\n List of tasks: {task_str}")
    list_constraints.append('')
    list_prompt.append('add_task_v1_random')

    for _ in range(num_iter):
        list_temp = [item for item in list_task if item not in list_prev_const]
        selected_task = random.choice(list_temp)
        list_instructions.append(add_task_v1_specific + f"\nQuestion: {seed_question}\t Answer:{seed_answer}\n Task: {selected_task}, definition: {dict_definition[selected_task]}")
        list_constraints.append(selected_task)
        list_prompt.append('selected_task')

    # deep-reasoning strategies: causal, comparison, multi-hop
    deep_strategies = [
        # (add_causal_v1,      'causal',     'add_causal_v1'),
        (add_comparison_v1,  'comparison', 'add_comparison_v1'),
        (add_multihop_v1,    'multihop',   'add_multihop_v1'),
    ]
    for prompt_tmpl, constraint_tag, prompt_tag in random.sample(deep_strategies, k=min(num_iter, len(deep_strategies))):
        list_instructions.append(prompt_tmpl + f"\nQuestion: {seed_question}\t Answer:{seed_answer}")
        list_constraints.append(constraint_tag)
        list_prompt.append(prompt_tag)

    return list_instructions, list_constraints, list_prompt


def load_task_info():
    """Load task definitions and return their lookup, ordered names, and prompt text."""
    task_path = os.path.join(ROOT_FOLDER, "config", "task_definition.json")
    with open(task_path, 'r') as f:
        list_task_ori = json.load(f)
    list_task = [item['task_name'] for item in list_task_ori]
    task_str = '\n'.join(list_task)
    dict_definition = {item['task_name']: item['definition'] for item in list_task_ori}
    return dict_definition, list_task, task_str


def load_seed_data(postfix: str, input_data, ):
    """Normalize dataset-specific image and seed-question fields; skip multi-image SAT items."""
    list_data = []

    if postfix == 'cvb':
        cases_root = os.environ.get("CASES_STUDY_ROOT", os.path.join(ROOT_FOLDER, "cases_study"))
        for i, list_meta in enumerate(tqdm(input_data)):
            image_path = list_meta['image_path']
            image_path = os.path.join(cases_root, image_path)
            image_id = image_path.split('/')[-1].split('.')[0]

            list_seed_q = [list_meta['questions']]
            list_data.append({'image_path': image_path, 'question': list_seed_q})

    elif postfix in ('qinstruct', 'hypersim', 'geo170k', 'sqa', ):
        for i, list_meta in enumerate(tqdm(input_data)):
            image_path = list_meta['image_path']
            image_id = image_path.split('/')[-1].split('.')[0]

            list_seed_q = []
            list_data.append({'image_path': image_path, 'question': list_seed_q})
    elif postfix in ('sat', ):
        for i, list_meta in enumerate(tqdm(input_data)):
            image_path = list_meta['image_paths']
            if len(image_path) > 1:
                continue

            list_seed_q = []
            list_data.append({'image_path': image_path[0], 'question': list_seed_q})
    else:

        for i, (image_path, list_meta) in enumerate(tqdm(input_data.items())):
            image_id = image_path.split('/')[-1].split('.')[0]
            caption = list_meta[0]
            list_qa = list_meta[1:]
            list_seed_q = [item['question'] for item in list_qa]
            list_data.append({'image_path': image_path, 'question': list_seed_q})
    return list_data
