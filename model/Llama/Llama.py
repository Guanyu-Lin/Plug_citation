# coding=utf-8
# Copyright 2022 The OpenBMB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from model_center.utils import check_web_and_convert_path
import bmtrain as bmt
from typing import *
# from transformers.modeling_outputs import BaseModelOutput

import os
import torch
from typing import Optional, List
from .basemodel import BaseModel,BaseModelOutput
from ..layer import Embedding, Encoder, Linear, RotaryEmbeddingESM
from .config import LlamaConfig

class Llama(BaseModel):
	
	_CONFIG_TYPE = LlamaConfig
	
	def __init__(self, config: LlamaConfig):
		
		super().__init__()
		
		self.config = config
		# Embedding Layer
		# pdb.set_trace()
		self.input_embedding = Embedding(
			vocab_size = config.vocab_size, 
			embedding_size = config.dim_model,
			length_scale = config.length_scale,
			dtype = config.dtype,
			int8 = config.int8,
			init_mean = config.emb_init_mean,
			init_std = config.emb_init_std,
		)
		
		self.position_bias = RotaryEmbeddingESM(
			dim = config.dim_head,
			dtype = config.dtype,
		)
		
		self.encoder = Encoder(
			num_layers = config.num_layers,
			dim_model = config.dim_model, 
			dim_ff = config.dim_ff,
			num_heads = config.num_heads,
			num_heads_kv = config.num_heads_kv,
			dim_head = config.dim_head,
			dtype = config.dtype, 
			int8 = config.int8,
			norm_eps = config.norm_eps, 
			norm_init_var = config.norm_init_var,
			norm_bias = config.norm_bias,
			att_init_mean = config.att_init_mean, 
			att_init_std = config.att_init_std,
			att_bias = config.att_bias,
			att_mask_value = float(config.att_mask_value),
			pos_bias_type = config.pos_bias_type,
			ffn_init_mean = config.ffn_init_mean, 
			ffn_init_std = config.ffn_init_std,
			ffn_bias = config.ffn_bias,
			ffn_activate_fn = config.ffn_activate_fn,
			length_scale = config.length_scale,
			attn_scale = config.attn_scale,
			dropout_p = config.dropout_p,
			post_layer_norm = config.post_layer_norm,
		)
		# Output Layer
		if config.cls_head:
			self.cls_projection = Linear(
				dim_out = config.cls_head,
				dim_in = config.dim_model,
				length_scale = config.length_scale,
				dtype = config.dtype,
				int8 = config.int8,
				init_mean = config.proj_init_mean,
				init_std = config.proj_init_std,
				bias = config.proj_bias,
			)
		if not config.tied:
			self.output_projection = Linear(
				dim_out = config.vocab_size,
				dim_in = config.dim_model,
				length_scale = config.length_scale,
				dtype = config.dtype,
				int8 = config.int8,
				init_mean = config.proj_init_mean,
				init_std = config.proj_init_std,
				bias = config.proj_bias,
			)
	
	def forward(self, 
				input_ids: Optional[torch.Tensor] = None,
				length: Optional[torch.Tensor] = None,
				attention_mask: Optional[torch.Tensor] = None,
				parameters : Optional[torch.Tensor] = None,
				deltas : Optional[Dict] = None,
				return_plug: Optional[Dict] = None,
				inputs_embeds: Optional[torch.FloatTensor] = None,
				use_cache: Optional[bool] = False,
				past_key_values: Optional[List[torch.FloatTensor]] = None,
				output_logits: Optional[bool] = True,
				pfxatt_mask : Optional[torch.Tensor] = None,
				output_attentions: Optional[bool] = False,
				output_hidden_states: Optional[bool] = False,
				return_dict: Optional[bool] = True,
				save_score=None,
		):
		""" This model inherits from BaseModel. This model is also a PyTorch torch.nn.Module subclass.
			You can use it as a regular PyTorch Module. You can also select the data and data type that 
			you want the model to return through changing the value of `output_logits`, 
			`output_pooler_output`, `output_attentions`, `output_hidden_states` and `return_dict`.

		Args:
			input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
				Indices of input sequence tokens in the vocabulary.
			length (`torch.LongTensor` of shape `(batch_size)`, *optional*):
				Length of input sequence before padding.  
			attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
				Mask to avoid performing attention on padding token indices. The values are selected in `[0, 1]`:
				- 1 for tokens that are **not masked**,
				- 0 for tokens that are **masked**.
				At least one of `length` and `attention_mask` must be given.
			inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
				Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
				is useful if you want to convert `input_ids` indices into associated vectors rather than the model's internal 
				token vectors.
			use_cache (`bool`, *optional*):
				If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).
			past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.num_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, dim_model)`):
				Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
			output_logits (`bool`, *optional*):
				Whether or not to return the prediction score for each token in vocabulary (before softmax).
			output_attentions (`bool`, *optional*):
				Whether or not to return the attentions tensors of all attention layers. 
				Unused now.
			output_hidden_states (`bool`, *optional*):
				Whether or not to return the hidden states of all layers.
				Unused now.
			return_dict (`bool`, *optional*):
				Whether or not to return a [`BaseModelOutput`] instead of a plain tuple.

		Return:
			BaseModelOutput or tuple: The GPT-2 output. 
			Depended on the value of `output_logits`, `output_attentions`, `output_hidden_states` and `return_dict`.
		"""

		with torch.no_grad():
			assert input_ids is not None or inputs_embeds is not None
			if input_ids is not None:
				batch = input_ids.size(0)
				input_length = input_ids.size(1)
				device = input_ids.device
			else:
				batch = inputs_embeds.size(0)
				input_length = inputs_embeds.size(1)
				device = inputs_embeds.device
			# print(input_ids)
			pkv_len = 0 if past_key_values is None else past_key_values[0][0].size(-2)
			seq_length = pkv_len + input_length
			
			has_prefix = False
			if deltas is not None and deltas["type"] == "prefix":
				prefix_num = deltas["prefix_num"]
				seq_length = seq_length 
				# seq_enc = seq_enc + prefix_num
				has_prefix = True
				# seq_length = seq_length

			if attention_mask is not None and length is None:

				length = attention_mask # which is one, which is zero, can pad outside

			elif length.dim() == 1:
				length = torch.arange(seq_length, device=device)[None, :].repeat(batch, 1) < length[:, None]
			# print(input_ids)
			directional_mask_2d = torch.arange(seq_length, device=device) <= torch.arange(
					seq_length, device=device
				).view(-1, 1)
			attention_mask = directional_mask_2d.view(1, seq_length, seq_length)

			if has_prefix:
				if pfxatt_mask is None:
					prefix_mask = torch.ones(batch, prefix_num, dtype=torch.bool, device=attention_mask.device)
				else:
					prefix_mask = length.view(batch, seq_length, 1) & pfxatt_mask.view(batch, 1, prefix_num)
 
				prefix_mask = prefix_mask[:, -input_length:, :]
			else:
				prefix_mask = None
			attention_mask = length.view(batch, seq_length, 1) & length.view(batch, 1, seq_length) & attention_mask

			attention_mask = attention_mask[:, -input_length:, :]




		if inputs_embeds is None:
			hidden_states = self.input_embedding(input_ids)
		else:
			hidden_states = inputs_embeds

		current_key_values = None

		if use_cache:

			hidden_states, current_key_values = self.encoder(hidden_states, attention_mask, self.position_bias, prefix_mask, parameters, deltas, return_plug = return_plug,
															 use_cache = use_cache, past_key_values = past_key_values, save_score = save_score)
		else:
			hidden_states = self.encoder(hidden_states, attention_mask, self.position_bias, prefix_mask, parameters, deltas, return_plug = return_plug, save_score = save_score)

		
		# use the hidden states of the last layer for sequential tasks, such as sequential labeling and language modeling.
		logits = None
		if output_logits:
			# import pdb
			# pdb.set_trace()
			if self.config.cls_head:
				logits = self.cls_projection(hidden_states)
			elif self.config.tied:
				logits = self.input_embedding.projection(hidden_states)
			elif not self.config.tied:
				# import pdb
				# pdb.set_trace()
				logits = self.output_projection(hidden_states)

		# BaseModelOutput or tuple: The LLaMa output. 
		if not return_dict:
			return hidden_states, current_key_values, logits, None, None
		else:
			# return BaseModelOutput(
			#	  last_hidden_state=hidden_states,
			#	  # encoder_last_hidden_state=encoder_outputs,
			#	  # past_key_values=None,
			#	  # encoder_hidden_states=None,
			#	  # decoder_hidden_states=None,
			#	  # decoder_attentions=None,
			#	  # cross_attentions=None,
			#	  # encoder_attentions=None,
			# )
			return BaseModelOutput(
				last_hidden_state = hidden_states,
				past_key_values = current_key_values,
				logits = logits,
				hidden_states = None,
				attentions = None,
			)

	@classmethod
	def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], config=None, **kwargs):
		if config is None:
			config = cls._CONFIG_TYPE.from_pretrained(pretrained_model_name_or_path)
		path = check_web_and_convert_path(pretrained_model_name_or_path, 'model')
		model = cls(config, **kwargs)
		bmt.init_parameters(model)
		bmt.load(model, os.path.join(path, 'pytorch_model.pt'), strict=False)
		return model
