from functools import lru_cache

import torch
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from .config import Settings


def _compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@lru_cache(maxsize=1)
def _build_cached_llm(model_name: str, max_new_tokens: int) -> HuggingFacePipeline:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for 4-bit quantization. "
            "Install PyTorch CUDA and verify torch.cuda.is_available() is True."
        )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=_compute_dtype(),
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    return HuggingFacePipeline(pipeline=text_generation_pipeline)


def build_llm(settings: Settings) -> HuggingFacePipeline:
    return _build_cached_llm(settings.model_name, settings.max_new_tokens)
