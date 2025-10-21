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
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cpu")
        self.start_token = [ST.SOH]
        self.end_tokens = [ST.EOT, ST.EOH, ST.SOA, ST.SOS]

    def prepare_prompts(self, text):
        prompts = [f"Timur: <neutral> {text}",
                   f"Timur: <strict> {text}",
                   f"Aiganysh: <neutral> {text}",
                   f"Aiganysh: <strict> {text}"]

        input_ids = self.tokenizer(prompts,
                                   add_special_tokens=True,
                                   padding=False,
                                   truncation=False,
                                   return_tensors=None)["input_ids"]

        input_ids = [self.start_token + ids + self.end_tokens for ids in input_ids]
        return input_ids

    def generate_ids(self, input_ids, sample_rate=22050):
        sampling_params = SamplingParams(
            n=1,  # num_return_sequences
            temperature=0.6,
            top_p=0.95,
            max_tokens=1200, #max_new_tokens
            stop_token_ids=[ST.EOS], #eos_token_id
            repetition_penalty=1.1,
            detokenize=False,
        )
        generated_ids = self.vllm_model.generate(
            prompt_token_ids=input_ids,
            sampling_params=sampling_params,
        )
        return generated_ids

    def _redistribute_and_decode(self, normalized_tokens):
        """
        Helper function to demultiplex the flat list of codes into the
        three layers required by the SNAC vocoder and decode to audio.
        """
        layer_1, layer_2, layer_3 = [], [], []

        # The number of 7-token blocks
        num_blocks = len(normalized_tokens) // 7

        for i in range(num_blocks):
            base_idx = 7 * i
            layer_1.append(normalized_tokens[base_idx])
            layer_2.append(normalized_tokens[base_idx + 1] - 4096)
            layer_3.append(normalized_tokens[base_idx + 2] - (2 * 4096))
            layer_3.append(normalized_tokens[base_idx + 3] - (3 * 4096))
            layer_2.append(normalized_tokens[base_idx + 4] - (4 * 4096))
            layer_3.append(normalized_tokens[base_idx + 5] - (5 * 4096))
            layer_3.append(normalized_tokens[base_idx + 6] - (6 * 4096))

        # Convert the Python lists to the required tensor format for the vocoder
        codes = [
            torch.tensor(layer_1).unsqueeze(0),
            torch.tensor(layer_2).unsqueeze(0),
            torch.tensor(layer_3).unsqueeze(0)
        ]

        with torch.no_grad():  # Good practice when running inference
            audio_hat = self.snac_model.decode(codes)
        return audio_hat

    def parse_output_as_speech(self, generated_ids):
        sos_token = ST.SOS
        eos_token = ST.EOS
        codec_offset = 128266
        batch_audio = []
        for output in generated_ids:
            # The generated tokens are a tuple in output.outputs[0].token_ids
            generated_tokens = output.outputs[0].token_ids

            # 1. Find the last occurrence of the SOS token and slice after it.
            # This separates the generated audio codes from any prefix/prompt tokens.
            try:
                # Find the index of the last SOS token by searching the reversed tuple
                last_sos_idx = len(generated_tokens) - 1 - generated_tokens[::-1].index(sos_token)
                cropped_tokens = generated_tokens[last_sos_idx + 1:]
            except ValueError:
                # If no SOS token is found, use the entire sequence
                cropped_tokens = generated_tokens

            # 2. Filter out all EOS tokens using a list comprehension.
            filtered_tokens = [token for token in cropped_tokens if token != eos_token]

            # 3. Trim the sequence to the nearest multiple of 7.
            # The codec expects a flat list of codes in groups of 7.
            num_blocks = len(filtered_tokens) // 7
            if num_blocks == 0:
                # If there are not enough tokens to form a single block, skip.
                # You might want to return a silent tensor or handle this differently.
                batch_audio.append(torch.zeros((1, 0)))  # Example: empty audio
                continue

            trimmed_length = num_blocks * 7
            trimmed_tokens = filtered_tokens[:trimmed_length]

            # 4. Normalize the tokens by subtracting the offset.
            # This is also done efficiently with a list comprehension.
            normalized_tokens = [t - codec_offset for t in trimmed_tokens]

            # 5. Redistribute the flat list into layers and decode to audio.
            audio_tensor = self._redistribute_and_decode(normalized_tokens)
            batch_audio.append(audio_tensor)

        return batch_audio

    def generate(self, text, sample_rate=22050):
        ids = self.prepare_prompts(text)
        generated_ids = self.generate_ids(ids)
        batch_audio = self.parse_output_as_speech(generated_ids)
        return batch_audio