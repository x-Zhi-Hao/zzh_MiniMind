from torch.utils.data import Dataset
import torch
import os
import random
import json
from datasets import load_dataset


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 使用 HuggingFace datasets 的惰性加载，避免一次性读入大文件
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # Step 1：tokenize 原始文本，留出首尾各 1 个 token 的位置给 BOS/EOS
        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=self.max_length - 2,  # 预留 BOS + EOS 的位置
            truncation=True,
        ).input_ids

        # Step 2：拼接 BOS + token序列 + EOS，构成完整序列
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

        # Step 3：右侧用 PAD 补齐到 max_length，保证 batch 内等长
        input_ids = tokens + [self.tokenizer.pad_token_id] * (
            self.max_length - len(tokens)
        )
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Step 4：labels 与 input_ids 完全相同，但 PAD 位置置 -100，
        #         CrossEntropyLoss 会自动忽略 -100，不计入 loss
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        # ！修正：返回 attention_mask，使 attention 层能屏蔽 padding token
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return input_ids, labels, attention_mask


def pre_processing_chat(conversations, add_system_ratio=0.2):
    """Optionally add a system prompt so SFT sees both chat styles."""
    system_prompts = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "You are a helpful AI assistant.",
    ]
    if conversations and conversations[0].get("role") != "system":
        if random.random() < add_system_ratio:
            return [
                {"role": "system", "content": random.choice(system_prompts)}
            ] + conversations
    return conversations


class SFTDataset(Dataset):
    """Conversation SFT dataset that computes loss only on assistant replies.

    Expected JSONL format:
    {"conversations": [{"role": "user", "content": "..."}, ...]}
    """

    def __init__(self, data_path, tokenizer, max_length=340):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=data_path, split="train")
        self.assistant_start_ids = tokenizer(
            f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
        ).input_ids
        self.assistant_end_ids = tokenizer(
            f"{tokenizer.eos_token}\n", add_special_tokens=False
        ).input_ids

    def __len__(self):
        return len(self.samples)

    def _make_prompt(self, conversations):
        # The SFT corpus contains a small tool-calling subset.  In that subset
        # ``tools`` and ``tool_calls`` are JSON-encoded strings, whereas
        # ``apply_chat_template`` expects Python lists/dicts.  Decode only
        # those structural fields; user/assistant text stays untouched.
        messages = []
        for conversation in conversations:
            message = dict(conversation)
            for field in ("functions", "tools", "tool_calls"):
                value = message.get(field)
                if isinstance(value, str):
                    try:
                        message[field] = json.loads(value)
                    except json.JSONDecodeError:
                        # Keep malformed optional metadata as-is.  Normal
                        # conversational samples do not use these fields.
                        pass
            messages.append(message)

        tools = (
            messages[0].get("tools") or messages[0].get("functions")
            if messages
            and messages[0].get("role") == "system"
            else None
        )
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
        )

    def _make_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        start_marker_length = len(self.assistant_start_ids)
        end_marker_length = len(self.assistant_end_ids)

        while i < len(input_ids):
            if input_ids[i : i + start_marker_length] != self.assistant_start_ids:
                i += 1
                continue

            start = i + start_marker_length
            end = start
            while end < len(input_ids):
                if input_ids[end : end + end_marker_length] == self.assistant_end_ids:
                    break
                end += 1

            for position in range(start, min(end + end_marker_length, self.max_length)):
                labels[position] = input_ids[position]
            i = end + end_marker_length if end < len(input_ids) else len(input_ids)

        return labels

    def __getitem__(self, index):
        conversations = pre_processing_chat(self.samples[index]["conversations"])
        prompt = self._make_prompt(conversations)
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self._make_labels(input_ids)
        attention_mask = (
            torch.tensor(input_ids, dtype=torch.long) != self.tokenizer.pad_token_id
        ).long()
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            attention_mask,
        )
