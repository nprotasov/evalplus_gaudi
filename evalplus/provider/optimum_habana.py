from hf import HuggingFaceDecoder


import copy
import time
from typing import List
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import habana_frameworks.torch.hpu as torch_hpu

from evalplus.provider.base import DecoderBase
from evalplus.provider.utility import (
    extra_eos_for_direct_completion,
    make_raw_chat_prompt,
)

import habana_frameworks.torch.core as htcore
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

from optimum.habana.transformers.trainer import _is_peft_model

adapt_transformers_to_gaudi()


class OptimumHabana():
    def __init__(
        self,
        name: str,
        device: str,
        gguf_file: str = None,
        **kwargs,
    ):
        self.device = torch.device(device)
        self.dtype = "bfloat16"
        self.attn_implementation = "flash_attention_2"

        self.lazy_mode = False
        self.hpu_graphs = False
        self.torch_compile = True

        kwargs = {
            # "device_map": device_map,
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": getattr(torch, self.dtype),
            # "attn_implementation": attn_implementation,  # "eager", "flash_attention_2", "sdpa"
            "gguf_file": gguf_file
        }
        print(f"{type(self).__name__} {kwargs=}")
        self.model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        self.model = self.model.eval().to(device)

        if self.torch_compile:
            self.model = self._get_torch_compiled_model(self.model)
        else:
            from habana_frameworks.torch.hpu import wrap_in_hpu_graph
            self.model = wrap_in_hpu_graph(self.model)

            if _is_peft_model(self.model):
                self.model.base_model = wrap_in_hpu_graph(
                    self.model.base_model)
                if self.model.peft_type == "ADAPTION_PROMPT":
                    self.model.base_model.model = wrap_in_hpu_graph(
                        self.model.base_model.model)

        self.generation_config = copy.deepcopy(self.model.generation_config)
        self.generation_config.use_cache = True
        self.generation_config.attn_softmax_bf16 = True
        self.generation_config.reuse_cache = True
        self.generation_config.use_flash_attention = True
        self.generation_config.flash_attention_recompute = True
        self.generation_config.flash_attention_causal_mask = True
        self.generation_config.flash_attention_fast_softmax = True
        self.generation_config.trust_remote_code = True
        self.generation_config.reduce_recompile = True
        self.generation_config.clear_hpu_graphs_cache = True
        self.generation_config.reduce_recompile = True

    def generate(self, input_tokens, tokenizer, **kwargs):
        torch_hpu.synchronize()
        outputs = self.model.generate(
            input_tokens,
            tokenizer=tokenizer,
            hpu_graphs=self.hpu_graphs,
            lazy_mode=self.lazy_mode,
            generation_config=self.generation_config,
            **kwargs
        )
        torch_hpu.synchronize()
        return outputs

    def _get_torch_compiled_model(model):
        # for gpt_bigcode, mpt, bloom, gpt2 model_type
        if hasattr(model, "transformer"):
            model.transformer = torch.compile(
                model.transformer, backend="hpu_backend", options={"keep_input_mutations": True}
            )
        # for gpt_neox
        elif hasattr(model, "gpt_neox"):
            model.gpt_neox = torch.compile(model.gpt_neox, backend="hpu_backend", options={
                                           "keep_input_mutations": True})
        # for llama, mistral, mixtral, qwen2
        elif hasattr(model, "model"):
            model.model = torch.compile(model.model, backend="hpu_backend", options={
                                        "keep_input_mutations": True})
        else:
            model = torch.compile(model, backend="hpu_backend", options={
                                  "keep_input_mutations": True})
        return model
