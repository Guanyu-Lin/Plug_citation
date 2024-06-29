import json
import torch
import os
import numpy as np
from transformers import LlamaTokenizer

import random
# from transformers import LlamaTokenizer, T5Config
from tools import shift_tokens_right

class OpenQAFormatter:
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
            model_inputs["labels"][:,0] = -100

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



class OpenQAPlugDFormatter:
    def __init__(self, config, mode, *args, **params):
        self.max_len = config.getint("train", "max_len")
        self.ctx_len = config.getint("train", "ctx_len")
        self.ans_max_len = config.getint("train", "ans_max_len")
        self.mode = mode
        # self.plmpath = os.path.join(config.get("model", "pretrained_model_path"), config.get("model", "pretrained_model"))
        self.plmpath = config.get("model", "pretrained_model_path")
        # import pdb
        # pdb.set_trace()
        # print()
        # self.tokenizer = LlamaTokenizer.from_pretrained("checkpoint/PLMs/t5-large/tokenizer")
        self.tokenizer = LlamaTokenizer.from_pretrained(os.path.join(self.plmpath, "tokenizer"))
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # print(self.tokenizer.pad_token)
        # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # pdb.set_trace()
        # print(repr(self.tokenizer.pad_token))
    def process(self, data):
        query = [d["question"] for d in data]
        # ctxs = ["\n".join(["%s\t%s" % (c["wikipedia_title"], c["text"].lstrip()) for c in d["context"]]) for d in data]
        ctxs = ["\n".join([c["text"].lstrip() for c in d["context"]]) for d in data]
        targets = [random.choice(d["answers"]) for d in data]

        query_info = self.tokenizer(query)

        ctx_info = self.tokenizer(ctxs)

        labels = self.tokenizer(text_target=targets, max_length=self.ans_max_len, padding="max_length", truncation=True)
        # print(query)
        # print(query_info["input_ids"])
        model_inputs = {
            "que_input_ids": query_info["input_ids"],
            "que_attention_mask": query_info["attention_mask"],
            "ctx_input_ids": ctx_info["input_ids"],
            "ctx_attention_mask": ctx_info["attention_mask"]
        }
        if self.mode == "train":
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

            model_inputs["decoder_input_ids"] = shift_tokens_right(torch.LongTensor(labels["input_ids"]), 0, 0)
            model_inputs["labels"] = labels["input_ids"]
            model_inputs["decoder_attention_mask"] = labels["attention_mask"]

        for key in model_inputs:
            model_inputs[key] = torch.LongTensor(model_inputs[key])
        if "labels" in model_inputs:
            model_inputs["labels"][:,0] = -100

        model_inputs["answers"] = [{ans for ans in doc["answers"]} for doc in data]

        return model_inputs

