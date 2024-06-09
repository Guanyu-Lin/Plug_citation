import json
import torch
import os
import numpy as np
from transformers import LlamaTokenizer

import random
# from transformers import LlamaTokenizer, T5Config
from tools import shift_tokens_right


class ASQAFormatter:
    def __init__(self, config, mode, *args, **params):
        self.max_len = config.getint("train", "max_len")
        self.ans_max_len = config.getint("train", "ans_max_len")
        self.model_type = config.get("model", "model_type")
        self.mode = mode
        if self.model_type == "PostT5" and self.mode != "test":
            self.max_len = 256
        self.plmpath = os.path.join(config.get("model", "pretrained_model_path"), config.get("model", "pretrained_model"))
        self.tokenizer = LlamaTokenizer.from_pretrained(os.path.join(self.plmpath, "tokenizer"))

    def generate_input(self, question, context):
        if self.model_type == "PostT5" and self.mode != "test":
            return " ".join(["question:", question.lstrip()])
        else:
            return " ".join(["question:", question.lstrip(), "context:", "\n".join([c["text"].lstrip() for c in context])])

    def preprocess_squad_batch(self, examples):
        inputs = [self.generate_input(qa["question"] + "<extra_id_0>" if "<extra_id_0>" not in qa["question"] else qa["question"], qa["context"]) for qa in examples]
        targets = []
        for qa in examples:
            targets.append("<extra_id_0>" + random.choice(qa["answers"]))

        return inputs, targets

    def process(self, data):
        inputs, targets = self.preprocess_squad_batch(data)
        model_inputs = self.tokenizer(inputs, max_length=self.max_len, padding="max_length", truncation=True)

        labels = self.tokenizer(text_target=targets, max_length=self.ans_max_len, padding="max_length", truncation=True)

        if self.mode == "train":
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

            model_inputs["decoder_input_ids"] = shift_tokens_right(torch.LongTensor(labels["input_ids"]), 0, 0)
            model_inputs["labels"] = labels["input_ids"]
            model_inputs["decoder_attention_mask"] = labels["attention_mask"]

        for key in model_inputs:
            model_inputs[key] = torch.LongTensor(model_inputs[key])

        # print(self.tokenizer.decode(model_inputs["input_ids"][0]))
        # print(self.tokenizer.decode(model_inputs["decoder_input_ids"][0]))
        # print("===" * 10)
        if "labels" in model_inputs:
            model_inputs["labels"][:, 0] = -100

        model_inputs["answers"] = [{" ".join(ans.split()[:512]) for ans in doc["answers"]} for doc in data]

        return model_inputs


class LlamaGeneration:
    def __init__(self, model, tokenizer):
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        # import pdb
        # pdb.set_trace()

    def _convert_to_tensors(self, input_text):
        model_inputs = {}
        input_ids = self.tokenizer.encode(input_text)

        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = [1] * len(model_inputs["input_ids"])

        for key in model_inputs:
            model_inputs[key] = torch.tensor(model_inputs[key]).int().unsqueeze(0)

        return model_inputs

    def _process_texts(self, text_list):
        input_tensors = list(map(self._convert_to_tensors, text_list))
        keys = set(input_tensors[0].keys())
        padded = {}
        for key in keys:
            padded[key] = pad(input_tensors, key, padding_side='left').cuda()
        return padded

    def generate(self, text_list, **kwargs):
        model_inputs = self._process_texts(text_list)

        with torch.inference_mode():
            result = self._decode(model_inputs, **kwargs)
        return result

    def _decode(self, model_inputs, **kwargs):
        raise NotImplementedError("_decode is not implemented.")


def pad(orig_items, key, padding_value=0, padding_side="left"):
    items = []
    if isinstance(orig_items[0][key], list):
        assert isinstance(orig_items[0][key][0], torch.Tensor)
        for it in orig_items:
            for tr in it[key]:
                items.append({key: tr})
    else:
        assert isinstance(orig_items[0][key], torch.Tensor)
        items = orig_items

    batch_size = len(items)
    shape = items[0][key].shape
    dim = len(shape)
    assert dim <= 3
    max_length = max(item[key].shape[-1] for item in items)
    min_length = min(item[key].shape[-1] for item in items)
    dtype = items[0][key].dtype

    if dim == 1:
        return torch.cat([item[key] for item in items], dim=0)
    elif dim == 2:
        if max_length == min_length:
            return torch.cat([item[key] for item in items], dim=0)
        tensor = torch.zeros((batch_size, max_length), dtype=dtype) + padding_value
    else:
        tensor = torch.zeros((batch_size, max_length, shape[-1]), dtype=dtype) + padding_value

    for i, item in enumerate(items):
        if dim == 2:
            if padding_side == "left":
                tensor[i, -len(item[key][0]):] = item[key][0].clone()
            else:
                tensor[i, :len(item[key][0])] = item[key][0].clone()
        elif dim == 3:
            if padding_side == "left":
                tensor[i, -len(item[key][0]):, :] = item[key][0].clone()
            else:
                tensor[i, :len(item[key][0]), :] = item[key][0].clone()

    return tensor


class ASQAPlugDFormatter:
    def __init__(self, config, mode, *args, **params):
        self.max_len = config.getint("train", "max_len")
        self.ctx_len = config.getint("train", "ctx_len")
        self.ans_max_len = config.getint("train", "ans_max_len")
        self.mode = mode
        # self.plmpath = os.path.join(config.get("model", "pretrained_model_path"), config.get("model", "pretrained_model"))
        self.plmpath = config.get("model", "pretrained_model_path")

        # self.tokenizer = LlamaTokenizer.from_pretrained("checkpoint/PLMs/t5-large/tokenizer")
        self.tokenizer = LlamaTokenizer.from_pretrained(os.path.join(self.plmpath, "tokenizer"))
        self.tokenizer.pad_token_id = 0

    def _convert_to_tensors(self, input_text):
        model_inputs = {}
        input_ids = self.tokenizer.encode(input_text)

        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = [1] * len(model_inputs["input_ids"])

        for key in model_inputs:
            model_inputs[key] = torch.tensor(model_inputs[key]).int().unsqueeze(0)

        return model_inputs

    def process(self, data):
        if self.mode == "train":
            input_ids = []
            ctx_ids = []
            attention_mask = []
            ctx_attention_mask = []
            labels = []
            length = []
            for ins in data:
                input = self.tokenizer.encode(ins["question"])
                ctx = self.tokenizer.encode(ins["context"])
                ctx = ctx[:self.ctx_len]
                ctx_ids.append(ctx + [self.tokenizer.pad_token_id] * (self.ctx_len - len(ctx)))

                ctx_attention_mask.append([1] * len(ctx) + [0] * (self.ctx_len - len(ctx)))

                output = self.tokenizer.encode(ins["answers"].strip(), add_special_tokens=False)
                output.append(self.tokenizer.eos_token_id)

                ids = input + output
                lab = [-100] * (len(input) - 1) + output
                lab.append(-100)

                length.append(len(ids))
                assert len(ids) == len(lab)
                ids, lab = ids[:self.max_len], lab[:self.max_len]
                input_ids.append(ids + [self.tokenizer.pad_token_id] * (self.max_len - len(ids)))
                labels.append(lab + [-100] * (self.max_len - len(ids)))
                attention_mask.append([1] * len(ids) + [0] * (self.max_len - len(ids)))
            model_inputs = {
                "ctx_input_ids": torch.LongTensor(ctx_ids),
                "ctx_attention_mask": torch.LongTensor(ctx_attention_mask),
            }
            model_inputs["input_ids"] = torch.LongTensor(input_ids)
            model_inputs["attention_mask"] = torch.LongTensor(attention_mask)
            model_inputs["labels"] = torch.LongTensor(labels)
            model_inputs["length"] = torch.LongTensor(length)

        elif self.mode == "valid":
            query = [d["question"] for d in data]
            pure_question = [d["pure_question"] for d in data]
            qa_pairs = [d["qa_pairs"] for d in data]
            ctxs = [d["context"] for d in data]
            targets = [d["answers"] for d in data]

            input_tensors = list(map(self._convert_to_tensors, query))
            keys = set(input_tensors[0].keys())
            padded_query_info = {}
            for key in keys:
                padded_query_info[key] = pad(input_tensors, key, padding_side='left').cuda()

            input_tensors = list(map(self._convert_to_tensors, ctxs))
            keys = set(input_tensors[0].keys())
            padded_ctx_info = {}
            for key in keys:
                padded_ctx_info[key] = pad(input_tensors, key, padding_side='left').cuda()

            input_tensors = list(map(self._convert_to_tensors, targets))
            keys = set(input_tensors[0].keys())
            padded_target_info = {}
            for key in keys:
                padded_target_info[key] = pad(input_tensors, key, padding_side='left').cuda()

            model_inputs = {
                "que_input_ids": padded_query_info["input_ids"],
                "que_attention_mask": padded_query_info["attention_mask"],
                "ctx_input_ids": padded_ctx_info["input_ids"],
                "ctx_attention_mask": padded_ctx_info["attention_mask"]
            }

            model_inputs["answers"] = targets
            model_inputs["query"] = query
            model_inputs["pure_question"] = pure_question
            model_inputs["qa_pairs"] = qa_pairs

        return model_inputs
