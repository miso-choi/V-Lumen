"""Batch collation with an answer-only loss mask (image and prompt tokens get label -100)."""

from __future__ import annotations

from qwen_vl_utils import process_vision_info


def make_answer_only_collate(processor, max_len: int, device: str):
    def collate(batch_messages):
        full_texts, prompt_texts, images = [], [], []
        for msgs in batch_messages:
            full_texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
            prompt = msgs[:-1] if msgs[-1]["role"] == "assistant" else msgs
            prompt_texts.append(processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True))
            imgs, *_ = process_vision_info(msgs)
            if imgs:
                images.extend(imgs)

        inputs = processor(text=full_texts, images=images or None, return_tensors="pt",
                           padding=True, truncation=True, max_length=max_len)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100
        for i, (msgs, prompt_text) in enumerate(zip(batch_messages, prompt_texts)):
            imgs, *_ = process_vision_info(msgs)
            prompt_len = processor(text=prompt_text, images=imgs or None, return_tensors="pt",
                                   truncation=True, max_length=max_len)["input_ids"].shape[1]
            labels[i, :prompt_len] = -100
        inputs["labels"] = labels
        return inputs

    return collate
