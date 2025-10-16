import asyncio
import torch
import os
from vllm import AsyncLLMEngine, LLM, SamplingParams
from transformers import AutoTokenizer
import threading
import queue
from .decoder import tokens_decoder_sync


import special_tokens as st


class OrpheusOfflineModel:
    def __init__(self, model_name, dtype=torch.bfloat16, tokenizer='canopylabs/orpheus-3b-0.1-ft'):
        self.model_name = model_name
        self.dtype = dtype
        self.model = LLM(model_name)
        self.available_voices = ["Timur", "Aiganysh"]
        self.available_tones = ["neutral", "strict"]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def _format_prompt(self, text, voice, tone):
        adapted_prompt = f"{voice}: <{tone}> {text}"
        prompt_tokens = self.tokenizer.encode(adapted_prompt,
                                              add_special_tokens=True)
        start_token = torch.tensor([[st.SOH]], dtype=torch.int64)
        end_tokens = torch.tensor([[st.EOT, st.EOH, st.SOA, st.SOS]], dtype=torch.int64)
        all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
        prompt_string = self.tokenizer.decode(all_input_ids[0])
        return prompt_string

    def generate_tokens_sync(self, text, voice, tone, request_id="req-001", temperature=0.6, top_p=0.8,
                             max_tokens=1200, repetition_penalty=1.3):
        prompt_string = self._format_prompt(text, voice, tone)
        print(prompt_string)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
            repetition_penalty=repetition_penalty,
        )

        output = self.model.generate(
            [prompt_string],
            sampling_params=sampling_params,
            request_id=request_id,
        )





