# from transformers import LlamaTokenizer
from model_center.tokenizer import LlamaTokenizer

import torch
from torch import nn
from model.metric import squad_metric, squad_train_metric, compute_qa, compute_mauve
import bmtrain as bmt
import os
from ..T5Adapter.T5Adapter import T5Adapter
from ..PlugD.PlugD import PlugD,HyperPlugD
from model_center.model.config import LlamaConfig

class Seq2Seq(nn.Module):
	def __init__(self, config, gpu_list, *args, **params):
		super(Seq2Seq, self).__init__()
		# self.plmpath = os.path.join(config.get("model", "pretrained_model_path"), config.get("model", "pretrained_model"))
		self.plmpath = config.get("model", "pretrained_model_path")
		self.llamaconfig = LlamaConfig.from_pretrained(self.plmpath)

		self.model_type = config.get("model", "model_type")
		if self.model_type == "t5" or self.model_type == "PostT5":
			self.model = T5Adapter(config)
		elif self.model_type == "PlugD" or self.model_type == "PostPlugD":
			self.model = PlugD(config)
		elif self.model_type == "HyperPlugD" or self.model_type == "PostHyperPlugD":
			self.model = HyperPlugD(config)
		else:
			raise ValueError("model_type has not been defined")

		self.ans_len = config.getint("train", "ans_max_len")
		self.loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)
		self.tokenizer = LlamaTokenizer.from_pretrained(os.path.join(self.plmpath, "tokenizer"))

		stop = ["\n", "\n\n"]
		stop = [] if stop is None else stop
		stop = list(set(stop + ["\n", "Ċ", "ĊĊ", "<0x0A>"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
		self.stop_token_ids = list(set([self.tokenizer._convert_token_to_id(stop_token) for stop_token in stop] + [self.tokenizer.eos_token_id]))
		# print([self.tokenizer._convert_token_to_id(stop_token) for stop_token in stop])
		# if "llama" in args.model.lower():
		self.stop_token_ids.remove(self.tokenizer.unk_token_id) # add padding 0?
		# import pdb
		# pdb.set_trace()
		self.RL = ("ELI5" in config.get("output", "model_name"))

	def forward(self, data, config, gpu_list, pre_result, mode, save_score=None):
		device = data["input_ids"].device if "input_ids" in data else data["ctx_input_ids"].device
		if mode == "train":
			if self.model_type in ["PostPlugD", "PostHyperPlugD"]:
				logits = self.model(data, no_ctx=True)
			else:
				logits = self.model(data, save_score=save_score)
			vocab_size = logits.shape[-1]

			loss = self.loss_func(logits.view(-1, vocab_size), data["labels"].view(-1))
			# predict = torch.argmax(logits, dim = 2)
			# pre_result = squad_train_metric(predict, data["labels"], pre_result)
			return {"loss": loss} 
		else:
			if self.model_type in ["PostPlugD", "PostHyperPlugD"] and mode == "valid":
				answer = self.model.generate_greedy(data, max_length=self.ans_len, no_ctx=True)
			else:
   
				answer = self.model.generate_random_sample(data, self.stop_token_ids, max_length=self.ans_len,save_score=save_score)
				result_text = list(map(self.tokenizer.decode, answer))
				# answer, result_text = self.model.generate_greedy(data, self.tokenizer, max_length=self.ans_len)
				# result_text = ["hello"]
			loss = torch.tensor(0.0).to(device)
		# import pdb
		# pdb.set_trace()
		if pre_result is None:
			pre_result = [{"answer": ans, "output": out, "question":  ques} for ans, out, ques, pair in zip(data["answers"], result_text, data["pure_question"])]
		else:
			pre_result.extend([{"answer": ans, "output": out, "question":  ques} for ans, out, ques, pair in zip(data["answers"], result_text, data["pure_question"])])
			# acc_result = compute_qa(answer, data["answers"], data["query"], acc_result, RL=self.RL)
			# acc_result = compute_mauve(result_text, data["answers"], data["query"], acc_result)
		return {"loss": loss, "pre_result": pre_result}

