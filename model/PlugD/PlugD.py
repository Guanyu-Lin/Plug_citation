# from ..Basic.DeltaT5 import DeltaT5, DeltaT5OnlyEnc
from ..Llama.Llama import Llama
from .generation_utils import BeamHypotheses, apply_repetition_penalty, top_k_top_p_filtering, pad
import torch.nn.functional as F

import math

import torch
from torch import nn
import bmtrain as bmt
import os
from model_center.model import LlamaConfig

from ..Basic.layers import Linear, MLP,KaimingLinear
from ..layer.blocks import FFNBlock
import json
from tools import reduce,print_rank
import time

class MapNet(torch.nn.Module):
	def __init__(self, hidden_size, mid_ratio=2):
		super().__init__()
		print("map net with mid dim=", int(hidden_size * mid_ratio))
		self.encoder = MLP(hidden_size, hidden_size, dim_mid=int(hidden_size * mid_ratio), init_std=0.02, length_scale=False, bias=False)
	def forward(self, doc_rep, mask):
		delta = self.encoder(doc_rep)
		return delta + doc_rep


def freeze_module(module: nn.Module, except_para=None):
	for name, param in module.named_parameters():
		if 'input_embedding' in name or ( not except_para is None and except_para in name):
			# import pdb
			# pdb.set_trace()
			continue
		param.requires_grad = False

class TaskPlug(nn.Module):
	def __init__(self, hidden_size, plm_config: LlamaConfig, bottleneck_dim: int = 32):
		super().__init__()

		self.hidden_size = hidden_size
		self.plm_config = plm_config
		self.bottleneck_dim = bottleneck_dim
		
		self.enc_adapter_A = Linear(1, 2 * self.plm_config.num_layers * self.plm_config.dim_model * self.bottleneck_dim, init_std=0.01)
		self.enc_adapter_B = Linear(1, 2 * self.plm_config.num_layers * self.plm_config.dim_model * self.bottleneck_dim, init_std=0.01)
		self.dec_adapter_A = Linear(1, 2 * self.plm_config.num_layers * self.plm_config.dim_model * self.bottleneck_dim, init_std=0.01)
		self.dec_adapter_B = Linear(1, 2 * self.plm_config.num_layers * self.plm_config.dim_model * self.bottleneck_dim, init_std=0.01)

		bmt.init_parameters(self.enc_adapter_A)
		bmt.init_parameters(self.enc_adapter_B)
		bmt.init_parameters(self.dec_adapter_A)
		bmt.init_parameters(self.dec_adapter_B)

	def get_adapter(self, device):
		# return None, None
		enc_adapters_A = self.enc_adapter_A(torch.ones(1, 1, dtype=torch.half, device=device)).view(self.plm_config.num_layers, 2, 1, self.plm_config.dim_model, self.bottleneck_dim)
		enc_adapters_B = self.enc_adapter_B(torch.ones(1, 1, dtype=torch.half, device=device)).view(self.plm_config.num_layers, 2, 1, self.plm_config.dim_model, self.bottleneck_dim)
		enc_adapters = torch.cat([enc_adapters_A, enc_adapters_B], dim=1)

		dec_adapters_A = self.dec_adapter_A(torch.ones(1, 1, dtype=torch.half, device=device)).view(self.plm_config.num_layers, 2, 1, self.plm_config.dim_model, self.bottleneck_dim)
		dec_adapters_B = self.dec_adapter_B(torch.ones(1, 1, dtype=torch.half, device=device)).view(self.plm_config.num_layers, 2, 1, self.plm_config.dim_model, self.bottleneck_dim)
		dec_adapters = torch.cat([dec_adapters_A, dec_adapters_B], dim=1)

		return enc_adapters, dec_adapters



class FFNTaskPlug(nn.Module):
	def __init__(self, hidden_size, plm_config: LlamaConfig, bottleneck_dim: int = 32):
		super().__init__()

		self.hidden_size = hidden_size
		self.plm_config = plm_config
		self.bottleneck_dim = bottleneck_dim
		
		self.enc_adapter_A = Linear(1, 1 * self.plm_config.num_layers * self.plm_config.dim_model * self.bottleneck_dim, init_std=0.01)
		self.enc_adapter_B = Linear(1, 1 * self.plm_config.num_layers * self.plm_config.dim_model * self.bottleneck_dim, init_std=0.01)
		self.dec_adapter_A = Linear(1, 1 * self.plm_config.num_layers * self.plm_config.dim_model * self.bottleneck_dim, init_std=0.01)
		self.dec_adapter_B = Linear(1, 1 * self.plm_config.num_layers * self.plm_config.dim_model * self.bottleneck_dim, init_std=0.01)

		bmt.init_parameters(self.enc_adapter_A)
		bmt.init_parameters(self.enc_adapter_B)
		bmt.init_parameters(self.dec_adapter_A)
		bmt.init_parameters(self.dec_adapter_B)

	def get_adapter(self, device):
		# return None, None
		enc_adapters_A = self.enc_adapter_A(torch.ones(1, 1, dtype=torch.half, device=device)).view(self.plm_config.num_layers, 1, 1, self.plm_config.dim_model, self.bottleneck_dim)
		enc_adapters_B = self.enc_adapter_B(torch.ones(1, 1, dtype=torch.half, device=device)).view(self.plm_config.num_layers, 1, 1, self.plm_config.dim_model, self.bottleneck_dim)
		enc_adapters = torch.cat([enc_adapters_A, enc_adapters_B], dim=1)

		dec_adapters_A = self.dec_adapter_A(torch.ones(1, 1, dtype=torch.half, device=device)).view(self.plm_config.num_layers, 1, 1, self.plm_config.dim_model, self.bottleneck_dim)
		dec_adapters_B = self.dec_adapter_B(torch.ones(1, 1, dtype=torch.half, device=device)).view(self.plm_config.num_layers, 1, 1, self.plm_config.dim_model, self.bottleneck_dim)
		dec_adapters = torch.cat([dec_adapters_A, dec_adapters_B], dim=1)

		# return None, None
		return enc_adapters, dec_adapters


class PlugD(nn.Module):
	def __init__(self, config, pretrain=False, doc_pos=True):
		super(PlugD, self).__init__()

		# plm_path = os.path.join(config.get("model", "pretrained_model_path"), config.get("model", "pretrained_model"))
		plm_path = config.get("model", "pretrained_model_path")
		# import pdb
		# pdb.set_trace()
		self.plm_config = LlamaConfig.from_pretrained(plm_path)

		# self.plm_config = LlamaConfig.from_pretrained(plm_path)

		self.hidden_size = self.plm_config.dim_model
		self.layerth = config.getint("train", "layerth")


		self.backbone = Llama.from_pretrained(plm_path) # doc_pos
		self.doc_plug_mapper = MapNet(self.hidden_size)
		# self.task_plug = FFNTaskPlug(self.hidden_size, self.plm_config, bottleneck_dim=config.getint("train", "bottleneck_dim"))

		self.pretrain = pretrain
		# if not pretrain:
		# 	freeze_module(self.backbone, except_para="plug")
		bmt.init_parameters(self.doc_plug_mapper)

	def generate_doc_plug(self, input_ids, attention_mask, save_score=None):
		with torch.inference_mode():
			# import pdb
			# pdb.set_trace()
			return_plug = {"layerth": self.layerth}
			last_hidden = self.backbone(
					input_ids = input_ids,
					attention_mask = attention_mask,
					return_plug = return_plug, save_score=save_score)

		doc_plug = self.doc_plug_mapper(last_hidden.last_hidden_state.clone(), attention_mask)
		# doc_plug = last_hidden.last_hidden_state.clone()
		return doc_plug, last_hidden
	
	# def generate_doc_plug_train(self, input_ids, attention_mask):
	#	  last_hidden = self.backbone(
	#			  input_ids = input_ids,
	#			  attention_mask = attention_mask,
	#			  only_encoder=True,
	#		  )
	#	  # doc_plug = self.doc_plug_mapper(last_hidden.last_hidden_state, attention_mask)
	#	  doc_plug = last_hidden.last_hidden_state
	#	  return doc_plug, last_hidden


	def forward(self, data, no_ctx=False, save_score=None):
		# if self.pretrain:
		#	  enc_adapters, dec_adapters = None, None
		# else:
		#	  enc_adapters, dec_adapters = self.task_plug.get_adapter(data["ctx_input_ids"].device)
		if no_ctx:
			parameters, deltas, data["ctx_attention_mask"] = None, None, None
		elif self.pretrain:
			parameters, _ = self.generate_doc_plug_train(data["ctx_input_ids"], data["ctx_attention_mask"])
			deltas = {"type": "prefix", "prefix_num": data["ctx_input_ids"].size(1), "layerth": self.layerth}
		else:
			# if save_score is not None:
			# 	parameters, _ = self.generate_doc_plug(data["ctx_input_ids"], data["ctx_attention_mask"], save_score+"plug")
			# else:
			parameters, _ = self.generate_doc_plug(data["ctx_input_ids"], data["ctx_attention_mask"], None)

			deltas = {"type": "prefix", "prefix_num": data["ctx_input_ids"].size(1), "layerth": self.layerth}
		# print("==" * 20)
		# parameters, deltas = None, None
		# here torch.cat([data["ctx_attention_mask"], data["que_attention_mask"]], dim=-1)

		output = self.backbone(
			input_ids=data["input_ids"],
			attention_mask=data["attention_mask"],
			pfxatt_mask=data["ctx_attention_mask"],
			# decoder_input_ids=data["decoder_input_ids"],
			# decoder_length=data["decoder_length"] if "decoder_length" in data else None,
			# decoder_attention_mask=data["decoder_attention_mask"] if "decoder_attention_mask" in data else None,
			deltas=deltas, parameters=parameters,
			# pfxatt_mask=data["ctx_attention_mask"],
			# enc_adapters=enc_adapters, dec_adapters=dec_adapters,
			# enc_lora=enc_adapters, dec_lora=dec_adapters,
			save_score=save_score,
		)

		return output.logits

	def generate_random_sample(self, data, stop_token_ids, max_length=20, top_k=0, top_p=0.95, temperature=1, repetition_window=None, repetition_penalty=1.0, no_ctx=False, save_score=None):

		max_length += 1
		if no_ctx:
			parameters, deltas, data["ctx_attention_mask"] = None, None, None
		else:
			if save_score is not None:
				parameters, _ = self.generate_doc_plug(data["ctx_input_ids"], data["ctx_attention_mask"], save_score+"plug")
			else:
				parameters, _ = self.generate_doc_plug(data["ctx_input_ids"], data["ctx_attention_mask"], None)

			deltas = {"type": "prefix", "prefix_num": data["ctx_input_ids"].size(1), "layerth": self.layerth}

		batch_size, device = data["ctx_input_ids"].size(0), data["ctx_input_ids"].device

		input_ids = data["que_input_ids"]
		  
		# import pdb
		# pdb.set_trace()
		# attention_mask = torch.cat([data["ctx_attention_mask"] * 0, data["que_attention_mask"]], dim = -1) 
		attention_mask = data["que_attention_mask"]    
		pred_start_index = input_ids.size(-1)
		past_key_values = None
		done = [False for _ in range(batch_size)]
		results = [None for _ in range(batch_size)]
		for i in range(max_length):
			if i == 0:
				out = self.backbone(
					input_ids=input_ids,
					attention_mask=attention_mask,
					deltas = deltas,
					parameters = parameters,
					pfxatt_mask=data["ctx_attention_mask"],
					use_cache = True,
					return_dict = True,
					save_score=save_score,
				)
				logits, past_key_values, score = out.logits, out.past_key_values, out.attentions
			else:
				out = self.backbone(
					input_ids=input_ids[:, -1:],
					attention_mask=attention_mask,	
					deltas = deltas,
					parameters = parameters,
					pfxatt_mask=data["ctx_attention_mask"],
					past_key_values=past_key_values,
					use_cache = True,
					return_dict = True,
				)
				logits, past_key_values = out.logits, out.past_key_values

			logits = logits[:, -1, :]


			if i == 0:
				logits[:, stop_token_ids] = -float("inf")

			apply_repetition_penalty(
				logits,
				batch_size,
				1,
				input_ids,
				repetition_penalty,
				pred_start_index,
				input_ids.size(-1) - 1,
				repetition_window,
			)

			logits = logits / temperature
			logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

			probs = F.softmax(logits, dim=-1)
			next_token = torch.multinomial(probs, num_samples=1)
			
		   

			for idx in range(batch_size):
				# import pdb
				# pdb.set_trace()
				if not done[idx] and (
					next_token[idx].item() in stop_token_ids or i == max_length - 1
				):

					done[idx] = True
					results[idx] = input_ids[idx, pred_start_index:].clone().cpu().tolist()  # type: ignore # noqa: E501

			if sum(done) == batch_size:
				break

			# update input ids
			input_ids = torch.cat([input_ids, next_token], dim=-1)
			attention_mask = torch.cat(
				[attention_mask, torch.ones((attention_mask.size(0), 1), dtype=torch.int, device=attention_mask.device)],
				dim=-1,
			)

		
		return results


class HyperPlugD(nn.Module):
	def __init__(self, config, pretrain=False, doc_pos=True):
		super(HyperPlugD, self).__init__()

		plm_path = os.path.join(config.get("model", "pretrained_model_path"), config.get("model", "pretrained_model"))
		self.plm_config = LlamaConfig.from_pretrained(plm_path)

		self.hidden_size = self.plm_config.dim_model
		self.layerth = config.getint("train", "layerth")

		self.doc_plug_encoder = DeltaT5OnlyEnc.from_pretrained(plm_path)
		self.backbone = DeltaT5.from_pretrained(plm_path, doc_pos=doc_pos)
		self.doc_plug_mapper = MapNet(self.hidden_size)
		# self.task_plug = FFNTaskPlug(self.hidden_size, self.plm_config, bottleneck_dim=config.getint("train", "bottleneck_dim"))

		self.pretrain = pretrain
		if not pretrain:
			freeze_module(self.doc_plug_encoder)
		bmt.init_parameters(self.doc_plug_mapper)

	def generate_doc_plug(self, input_ids, attention_mask):
		with torch.inference_mode():
			last_hidden = self.doc_plug_encoder(
					input_ids = input_ids,
					attention_mask = attention_mask,
				)
		doc_plug = self.doc_plug_mapper(last_hidden.clone(), attention_mask)
		return doc_plug, last_hidden

	def generate_doc_plug_train(self, input_ids, attention_mask):

		last_hidden = self.doc_plug_encoder(
				input_ids = input_ids,
				attention_mask = attention_mask,
			)
		doc_plug = self.doc_plug_mapper(last_hidden, attention_mask)
		return doc_plug, last_hidden
		# return None, last_hidden
	
	def cal_dec(self, hiddens, mask, dec_inp, dec_mask):
		batch, hidden_num, hidden_size = hiddens.size()
		mask_num = mask.size(1)
		if mask_num != hidden_num:
			mask = torch.max(mask.view(batch, -1, self.nto1), dim=2)[0]
		output, logits = self.backbone(
			encoder_outputs=hiddens,
			attention_mask=mask,
			decoder_input_ids=dec_inp,
			decoder_attention_mask=dec_mask,
		)
		return logits

	def forward(self, data, no_ctx=False):
		# if self.pretrain:
		# enc_adapters, dec_adapters = None, None
		# else:
		#	  enc_adapters, dec_adapters = self.task_plug.get_adapter(data["ctx_input_ids"].device)
		if no_ctx:
			parameters = None
			deltas = None
		else:
			if self.pretrain:
				parameters, _ = self.generate_doc_plug_train(data["ctx_input_ids"], data["ctx_attention_mask"])
			else:
				parameters, _ = self.generate_doc_plug(data["ctx_input_ids"], data["ctx_attention_mask"])
			deltas = {"type": "prefix", "prefix_num": data["ctx_input_ids"].size(1), "layerth": self.layerth}

		# parameters, deltas = None, None
		output, logits = self.backbone(
			input_ids=data["que_input_ids"],
			attention_mask=data["que_attention_mask"],
			decoder_input_ids=data["decoder_input_ids"],
			decoder_length=data["decoder_length"] if "decoder_length" in data else None,
			decoder_attention_mask=data["decoder_attention_mask"] if "decoder_attention_mask" in data else None,
			deltas=deltas, parameters=parameters,
			pfxatt_mask=data["ctx_attention_mask"],
			# enc_adapters=enc_adapters, dec_adapters=dec_adapters,
			# enc_lora=enc_adapters, dec_lora=dec_adapters,
		)
		return logits

	def generate_greedy(self, data, gen_length=20, no_ctx=False):
		# enc_adapters, dec_adapters = None, None
		# enc_adapters, dec_adapters = self.task_plug.get_adapter(data["ctx_input_ids"].device)
		if no_ctx:
			parameters = None
			deltas = None
		else:
			parameters, _ = self.generate_doc_plug(data["ctx_input_ids"], data["ctx_attention_mask"])
			deltas = {"type": "prefix", "prefix_num": data["ctx_input_ids"].size(1), "layerth": self.layerth}

		batch, device = data["ctx_input_ids"].size(0), data["ctx_input_ids"].device

		dec_input_ids = torch.zeros(batch, gen_length + 4, dtype=torch.long).to(device)
		length = torch.LongTensor([gen_length + 4] * batch).to(device)
		position =	0 # batch

		predict, logits = self.backbone(
			input_ids = data["que_input_ids"],
			attention_mask = data["que_attention_mask"],
			decoder_input_ids = dec_input_ids,
			decoder_length=length,
			deltas = deltas,
			parameters = parameters,
			pfxatt_mask=data["ctx_attention_mask"],
			# enc_adapters = enc_adapters, dec_adapters = dec_adapters,
			# enc_lora=enc_adapters, dec_lora=dec_adapters,
		)

		encoder_outputs = predict.encoder_last_hidden_state

		answer = [torch.argmax(logits[:, position], dim=-1)]
		end = (answer[-1] == 1)
		# past_key_values, att_mask = predict.past_key_values, data["decoder_attention_mask"]
		for i in range(gen_length):
			if not end.all():
				position += 1
				dec_input_ids[:, position] = answer[-1]
			predict, logits = self.backbone(
				encoder_outputs=encoder_outputs,
				attention_mask = data["que_attention_mask"],
				decoder_input_ids=dec_input_ids,
				decoder_length=length,
				pfxatt_mask=data["ctx_attention_mask"],
				deltas = deltas,
				parameters = parameters,

				# enc_adapters = enc_adapters, dec_adapters = dec_adapters,
				# enc_lora=enc_adapters, dec_lora=dec_adapters,
			)
			if not end.all():
				answer.append(torch.argmax(logits[:, position], dim=-1))
				# answer[-1][end] = 1
				end = end | (answer[-1] == 1)
			all_end = reduce(end.all().int(), "sum")
			if all_end == bmt.world_size():
				break
		answer = torch.cat([a.unsqueeze(1) for a in answer], dim=1).contiguous()
		return answer
