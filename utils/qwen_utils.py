import ast
import base64
import json
import os
import time
import traceback
import warnings
from io import BytesIO
from pathlib import Path
from pprint import pprint
from typing import Dict

CACHE_DIR = os.environ.get("CACHE_DIR", str(Path(__file__).resolve().parents[1] / ".cache" / "hf"))
os.environ.setdefault("HF_HOME", CACHE_DIR)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(CACHE_DIR, "datasets"))
os.environ.setdefault("HF_MODULES_CACHE", os.path.join(CACHE_DIR, "modules"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(CACHE_DIR, "transformers"))

try:
    from vllm import LLM, SamplingParams
except Exception:
    LLM = None
    SamplingParams = None
    print("No vllm installed!")

import torch
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration

from utils.infer_utils import load_exist_file, make_json_safe, write_to_file

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"


def _image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getbuffer()).decode("ascii")


def _load_image_rgb(image_pil=None, image_path=None, bbox=None):
    if image_path is not None:
        image = Image.open(image_path).convert("RGB")
    elif image_pil is not None:
        image = image_pil.convert("RGB")
    else:
        raise ValueError("No image provided!")

    if bbox is not None:
        draw = ImageDraw.Draw(image)
        x1, y1, x2, y2 = map(int, bbox)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)

    return image


def load_image(image_pil=None, image_path=None, bbox=None):
    image = _load_image_rgb(image_pil=image_pil, image_path=image_path, bbox=bbox)
    return image, _image_to_base64(image)


def load_image_qwen(image_pil=None, image_path=None, bbox=None):
    image = _load_image_rgb(image_pil=image_pil, image_path=image_path, bbox=bbox)
    return _image_to_base64(image)


def _processor_path(model_path: str) -> str:
    if not os.path.isdir(model_path):
        return model_path
    if os.path.exists(os.path.join(model_path, "preprocessor_config.json")):
        return model_path
    parent = os.path.dirname(model_path)
    if os.path.exists(os.path.join(parent, "preprocessor_config.json")):
        return parent
    return model_path


def load_models(model_path: str, use_vllm: bool = False):
    if use_vllm:
        if LLM is None:
            raise ImportError("vllm is not installed, but use_vllm=True was requested.")
        model = LLM(model=model_path, gpu_memory_utilization=0.9, trust_remote_code=True, max_model_len=65536)
    elif "Qwen3" in model_path or "qwen3" in model_path:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

    processor = AutoProcessor.from_pretrained(_processor_path(model_path))
    dict_config_m = {"model": model, "processor": processor, "use_vllm": use_vllm}
    if hasattr(processor, "tokenizer"):
        dict_config_m["tokenizer"] = processor.tokenizer
    return dict_config_m


def load_models_qwen(model_path: str):
    return load_models(model_path=model_path, use_vllm=False)


def prepare_input(prompt: str, processor, image=None, system_prompt=None):
    messages = []

    if system_prompt is not None:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

    if image is None:
        messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
    else:
        if not isinstance(image, list):
            image = [image]
        content = [{"type": "image", "image": img} for img in image]
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs, messages


def prepare_prompt(prompt: str, tokenizer, processor, image=None, enable_thinking=True):
    return prepare_input(prompt=prompt, processor=processor, image=image)


def prepare_input_vllm(prompt: str, processor, image=None):
    if image is None:
        messages = [{"role": "user", "content": prompt}]
    else:
        if not isinstance(image, list):
            image = [image]
        content = [{"type": "image", "image": img} for img in image]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _video_inputs, _video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    return [{"prompt": text, "multi_modal_data": {"image": image_inputs}}], messages


def process_output(generation_output, processor, input_ids, update_ans_ids: bool = False):
    outputs = generation_output.sequences.detach().cpu()
    outputs_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, outputs)]
    return processor.tokenizer.batch_decode(outputs_trimmed, skip_special_tokens=True)[0]


def load_model_path(modeltype):
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    if modeltype == "qwen25_7b":
        model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    elif modeltype == "qwen25_3b":
        model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    elif modeltype == "qwen3_4b":
        model_path = "Qwen/Qwen3-VL-4B-Instruct"
    elif modeltype == "qwen3_2b":
        model_path = "Qwen/Qwen3-VL-2B-Instruct"
    elif modeltype == "qwen3_8b":
        model_path = "Qwen/Qwen3-VL-8B-Instruct"
    elif modeltype == "qwen3_32b":
        model_path = "Qwen/Qwen3-VL-32B-Instruct"
    print(f"Evaluating model: {model_path}")
    return model_path


def load_model_path_qwen(modeltype):
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    if modeltype == "qwen25_7b":
        model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    elif modeltype == "qwen25_3b":
        model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    elif modeltype == "qwen3_4b":
        model_path = "Qwen/Qwen3-VL-4B-Instruct"
    elif modeltype == "qwen3_8b":
        model_path = "Qwen/Qwen3-VL-8B-Instruct"
    elif modeltype == "qwen3_32b":
        model_path = "Qwen/Qwen3-32B"
    print(f"Evaluating model: {model_path}")
    return model_path


def _decode_generated_content(generated_ids, inputs, processor):
    """Remove input prompt tokens before decoding each generated continuation."""
    output_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def model_infer(
    dict_input: Dict,
    dict_config_m: Dict,
    max_len: int = 10000,
    print_content: bool = False,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k=20,
    return_dict_in_generate: bool = False,
    output_hidden_states: bool = False,
    **kwargs,
):
    """Run Transformers or vLLM inference and return parsed content plus metadata."""
    try:
        model = dict_config_m["model"]
        processor = dict_config_m["processor"]
    except Exception as e:
        print(e)
        traceback.print_exc()
        raise

    prompt = dict_input.get("prompt", "")
    system_prompt = dict_input.get("system_prompt", None)
    image = dict_input.get("image", None)
    dict_output = {"content": "", "thinking": ""}

    if not dict_config_m.get("use_vllm", False):
        inputs, _conversation = prepare_input(system_prompt=system_prompt, prompt=prompt, image=image, processor=processor)
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_len,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
            )

        content = _decode_generated_content(generated_ids, inputs, processor)

        if return_dict_in_generate:
            with torch.no_grad():
                outputs_forward = model(
                    **inputs,
                    return_dict_in_generate=return_dict_in_generate,
                    output_hidden_states=output_hidden_states,
                )
            dict_output["last_hidden"] = outputs_forward.hidden_states[-1].detach().cpu()
    else:
        if SamplingParams is None:
            raise ImportError("vllm is not installed, but dict_config_m['use_vllm'] is True.")
        start_time = time.time()
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_len, top_p=top_p, top_k=top_k)
        inputs, _messages = prepare_input_vllm(prompt=prompt, image=image, processor=processor)
        outputs = model.generate(inputs, sampling_params=sampling_params)
        content = ""
        for output in outputs:
            content = output.outputs[0].text
        print(time.time() - start_time)

    content = postprocess_output(content)
    if print_content:
        if isinstance(content, str):
            print(content)
        else:
            pprint(content)

    dict_output["content"] = content
    return dict_output


def model_infer_qwen(
    dict_input: Dict,
    dict_config_m: Dict,
    max_len: int = 10000,
    print_content: bool = False,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k=20,
):
    """Expose inference as the (content, thinking) tuple expected by generation and judging."""
    output = model_infer(
        dict_input=dict_input,
        dict_config_m=dict_config_m,
        max_len=max_len,
        print_content=print_content,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    return output["content"], output.get("thinking", "")


def postprocess_output(output: str):
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
