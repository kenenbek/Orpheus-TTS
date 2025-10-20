import torch
from vllm import AsyncLLMEngine, LLM, SamplingParams
from transformers import AutoTokenizer
from snac import SNAC

from . import special_tokens as ST
import scipy.io.wavfile as wavfile


class OrpheusOfflineModel:
    def __init__(self, model_path, dtype=torch.bfloat16, tokenizer='canopylabs/orpheus-3b-0.1-ft',
                 device="cuda"):
        self.model_path = model_path
        self.dtype = dtype
        self.device = device
        self.vllm_model = LLM(model=model_path,
                              max_model_len=8192,
                              dtype=dtype,
                              device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(self.device)

    def prepare_prompts(self, text):
        prompts = [f"Timur: <neutral> {text}",
                   f"Timur: <strict> {text}",
                   f"Aiganysh: <neutral> {text}",
                   f"Aiganysh: <strict> {text}"]

        all_input_ids = []
        for prompt in prompts:
            input_ids = self.tokenizer(prompt,
                                       add_special_tokens=True,
                                       return_tensors="pt").input_ids
            all_input_ids.append(input_ids)

        start_token = torch.tensor([[ST.SOH]], dtype=torch.int64)
        end_tokens = torch.tensor([[ST.EOT, ST.EOH, ST.SOA, ST.SOS]], dtype=torch.int64)

        all_modified_input_ids = []
        for input_ids in all_input_ids:
            modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)  # SOH SOT Text EOT EOH
            all_modified_input_ids.append(modified_input_ids)

        input_ids = all_modified_input_ids
        return input_ids

    def generate(self, text, request_id="req-001"):
        input_ids, attention_mask = self.prepare_prompts(text)
        sampling_params = SamplingParams(
            n=1,  # num_return_sequences
            temperature=0.6,
            top_p=0.95,
            max_tokens=1200, #max_new_tokens
            stop_token_ids=[ST.EOS], #eos_token_id
            repetition_penalty=1.1,
            detokenize=False,
        )

        with torch.no_grad():
            generated_ids = self.vllm_model.generate(
                prompt_token_ids=input_ids,
                sampling_params=sampling_params,
            )
            return generated_ids

    def parse_output_as_speech(self, generated_ids):
        # @title Parse Output as speech
        token_to_find = ST.SOS
        token_to_remove = ST.EOS

        token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

        if len(token_indices[1]) > 0:
            last_occurrence_idx = token_indices[1][-1].item()
            cropped_tensor = generated_ids[:, last_occurrence_idx + 1:]
        else:
            cropped_tensor = generated_ids

        mask = cropped_tensor != token_to_remove

        processed_rows = []

        for row in cropped_tensor:
            masked_row = row[row != token_to_remove]
            processed_rows.append(masked_row)

        code_lists = []

        for row in processed_rows:
            row_length = row.size(0)
            new_length = (row_length // 7) * 7
            trimmed_row = row[:new_length]
            trimmed_row = [t - 128266 for t in trimmed_row]
            code_lists.append(trimmed_row)

        def redistribute_codes(code_list):
            layer_1 = []
            layer_2 = []
            layer_3 = []
            for i in range((len(code_list) + 1) // 7):
                layer_1.append(code_list[7 * i])
                layer_2.append(code_list[7 * i + 1] - 4096)
                layer_3.append(code_list[7 * i + 2] - (2 * 4096))
                layer_3.append(code_list[7 * i + 3] - (3 * 4096))
                layer_2.append(code_list[7 * i + 4] - (4 * 4096))
                layer_3.append(code_list[7 * i + 5] - (5 * 4096))
                layer_3.append(code_list[7 * i + 6] - (6 * 4096))
            codes = [torch.tensor(layer_1).unsqueeze(0),
                     torch.tensor(layer_2).unsqueeze(0),
                     torch.tensor(layer_3).unsqueeze(0)]
            audio_hat = self.snac_model.decode(codes)
            return audio_hat

        my_samples = []
        for i, code_list in enumerate(code_lists):
            samples = redistribute_codes(code_list)
            my_samples.append(samples)

            samples_np = samples.detach().squeeze().to("cpu").numpy()
            wavfile.write(f'audio_{i}.wav', 24000, samples_np)

    def pipeline(self, text):
        ids = self.prepare_prompts(text)
        generated_ids = self.generate(ids)
        self.parse_output_as_speech(generated_ids)