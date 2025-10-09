import logging
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("finetune.log")]
)

logger = logging.getLogger(__name__)


class DataCollatorForOrpheus:
    def __init__(self, pad_token_id: int, label_pad_token_id: int = -100):
        self.pad_token_id = int(pad_token_id)
        self.label_pad_token_id = int(label_pad_token_id)

    def __call__(self, features):
        # Ensure lists
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        attn = [f["attention_mask"] for f in features]

        lengths = [len(x) for x in input_ids]
        max_len = max(lengths)

        def pad_list(lst, pad_value, to_len):
            return lst + [pad_value] * (to_len - len(lst))

        batch_input_ids = [pad_list(x, self.pad_token_id, max_len) for x in input_ids]
        batch_labels = [pad_list(x, self.label_pad_token_id, max_len) for x in labels]

        if any(a is None for a in attn):
            attn = [[1] * len(x) for x in input_ids]
        batch_attn = [pad_list(a, 0, max_len) for a in attn]

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attn, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


