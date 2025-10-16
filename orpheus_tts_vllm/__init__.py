import asyncio
import torch
import os
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer
import threading
import queue
from .decoder import tokens_decoder_sync


class OrpheusModel:
    def __init__(self, model_name, dtype=torch.bfloat16, tokenizer='canopylabs/orpheus-3b-0.1-ft',
                 **engine_kwargs):
        self.model_name = self._map_model_params(model_name)
        self.dtype = dtype
        self.engine_kwargs = engine_kwargs  # vLLM engine kwargs
        self.engine = self._setup_engine()
        self.available_voices = ["Timur", "Aiganysh"]
        self.available_voices = ["Timur", "Aiganysh"]

        # Use provided tokenizer path or default to model_name
        tokenizer_path = tokenizer if tokenizer else model_name
        self.tokenizer = self._load_tokenizer(tokenizer_path)

    def _setup_engine(self):
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            dtype=self.dtype,
            **self.engine_kwargs
        )

        return AsyncLLMEngine.from_engine_args(engine_args)

    def _format_prompt(self, prompt, voice="tara"):
            if voice:
                adapted_prompt = f"{voice}: {prompt}"
                prompt_tokens = self.tokenizer(adapted_prompt, return_tensors="pt")
                start_token = torch.tensor([[ 128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = self.tokenizer.decode(all_input_ids[0])
                return prompt_string
            else:
                prompt_tokens = self.tokenizer(prompt, return_tensors="pt")
                start_token = torch.tensor([[ 128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = self.tokenizer.decode(all_input_ids[0])
                return prompt_string