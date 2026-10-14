#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

import warnings
import random
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaAttention,
    repeat_kv,
    rotate_half,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    LLAMA_INPUTS_DOCSTRING
)

from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

from transformers.utils import (
    logging,
)

logger = logging.get_logger(__name__)

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


def apply_rotary_pos_emb(
    q, 
    k, 
    cos, 
    sin, 
    position_ids, 
    image_start_idx=35, 
    text_start_idx=None, 
    q_proj_skip_type='None',
    unsqueeze_dim=1
):
    """
    Apply RoPE with proper position handling for skipped tokens.
    Key insight: When we skip Q projection for certain tokens, we need to use
    the CORRECT position IDs for the remaining tokens, not just slice the cos/sin.
    """
    # --- K RoPE (always full sequence) ---
    k_embed = (k * cos[position_ids].unsqueeze(unsqueeze_dim)) + \
              (rotate_half(k) * sin[position_ids].unsqueeze(unsqueeze_dim))
              
    # --- Q RoPE ---
    q_proj_skip_type = q_proj_skip_type.lower()
    
    if q_proj_skip_type == 'none' or q.shape[-2] == 1:
        # No skip: use full position_ids
        q_position_ids = position_ids
    
    elif q_proj_skip_type == 'vision':
        # Skip vision tokens: Q contains [prefix, text]
        # Need position IDs for [0:image_start_idx, text_start_idx:]
        prefix_pos = position_ids[:, :image_start_idx]
        text_pos = position_ids[:, text_start_idx:]
        q_position_ids = torch.cat([prefix_pos, text_pos], dim=1)
        
    elif q_proj_skip_type == 'text':
        # Skip text tokens: Q contains [prefix, vision]
        # Need position IDs for [0:text_start_idx]
        q_position_ids = position_ids[:, :text_start_idx]
        
    elif q_proj_skip_type == 'all':
        # Skip vision and text: Q contains [prefix]
        # Need position IDs for [0:image_start_idx]
        q_position_ids = position_ids[:, :image_start_idx]
    else:
        warnings.warn(f"Unrecognized q_proj_skip_type '{q_proj_skip_type}' in RoPE, using full sequence.")
        q_position_ids = position_ids
    
    # Apply RoPE to Q with correct position IDs
    cos_q = cos[q_position_ids].unsqueeze(unsqueeze_dim)
    sin_q = sin[q_position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)

    return q_embed, k_embed


def get_skip_regions_and_compute_tensor(skip_type: str, image_start_idx: int, text_start_idx: int, hidden_states: torch.Tensor):
    
    compute_tensor = hidden_states
    # skip_regions_for_zeroing = []

    if skip_type.lower() == 'none':
        pass 
    elif skip_type.lower() == 'vision':
        # skip_regions_for_zeroing = [(image_start_idx, text_start_idx)]
        compute_tensor = torch.cat((
            hidden_states[:, :image_start_idx, :], 
            hidden_states[:, text_start_idx:, :]
        ), dim=1)
    elif skip_type.lower() == 'text':
        # skip_regions_for_zeroing = [(text_start_idx, hidden_states.shape[1])]
        compute_tensor = hidden_states[:, :text_start_idx, :]
    elif skip_type.lower() == 'all':
        # skip_regions_for_zeroing = [(image_start_idx, hidden_states.shape[1])]
        compute_tensor = hidden_states[:, :image_start_idx, :]
    
    return compute_tensor#, skip_regions_for_zeroing


class LlavaLlamaSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        image_start_idx: Optional[int] = 35,
        text_start_idx: Optional[int] = None,
        q_proj_skip_type: Optional[str] = 'None',
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        kv_seq_len = hidden_states.shape[1]
        bsz = hidden_states.shape[0]
        
        q_proj_skip_type = q_proj_skip_type.lower()

        
        # --- q_proj_skip_type implementation---
        if kv_seq_len == 1 or q_proj_skip_type == 'none' or image_start_idx is None or text_start_idx is None:
            # No Q-projection skip (no speedup from Q)
            compute_tensor_q = hidden_states
        elif q_proj_skip_type == 'vision':
            # Skip Q projection for vision tokens
            compute_tensor_q = torch.cat((
                hidden_states[:, :image_start_idx, :], 
                hidden_states[:, text_start_idx:, :]
            ), dim=1)            
        elif q_proj_skip_type == 'text':
            # Skip Q projection for text tokens
            compute_tensor_q = hidden_states[:, :text_start_idx, :]
        elif q_proj_skip_type == 'all':
            # Skip Q for both vision and text (maximum Q skip)
            compute_tensor_q = hidden_states[:, :image_start_idx, :]
        else:
            # Unrecognized skip type: default to no skip
            warnings.warn(f"Unrecognized q_proj_skip_type '{q_proj_skip_type}', defaulting to 'None'.")
            compute_tensor_q = hidden_states
            
        query_states = self.q_proj(compute_tensor_q)
        bsz, q_len, _ = query_states.size()
        

        # K/V projections always use full hidden_states (full context)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                        cos, sin, position_ids,
                                                        image_start_idx, text_start_idx,
                                                        q_proj_skip_type=q_proj_skip_type)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value



class SearchVDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int,
                 skip_config: Tuple[str, str] = ('None', 'None')):
        super().__init__(config, layer_idx)
        self.self_attn = LlavaLlamaSdpaAttention(config, layer_idx)
        
        self.mha_skip_type = skip_config[0].lower()
        self.ffn_skip_type = skip_config[1].lower()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        n_visual: Optional[int] = 0,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # print(n_visual)
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        image_start_idx = 35 
        text_start_idx = image_start_idx + n_visual

        residual = hidden_states
        norm_hidden_states = self.input_layernorm(hidden_states)
        
        is_decoding = hidden_states.shape[1] == 1
        mha_q_skip_type = 'none' if is_decoding else self.mha_skip_type
        ffn_skip_type = 'none' if is_decoding else self.ffn_skip_type

        # Self Attention
        attn_output, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=norm_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            image_start_idx=image_start_idx, 
            text_start_idx=text_start_idx, 
            q_proj_skip_type=mha_q_skip_type,
            **kwargs,
        )
        
        # MHA residual connection
        # Key insight: attn_output only contains computed tokens, need to map back correctly
        if mha_q_skip_type == 'none':
            hidden_states = residual + attn_output
        else:
            # For skipped tokens, keep residual unchanged (no attention update)
            # For computed tokens, add attention output
            hidden_states = residual.clone()
            if mha_q_skip_type == 'vision':
                # attn_output contains [prefix, text], map to correct positions
                hidden_states[:, :image_start_idx, :] += attn_output[:, :image_start_idx, :]
                hidden_states[:, text_start_idx:, :] += attn_output[:, image_start_idx:, :]
                # Vision tokens (image_start_idx:text_start_idx) keep residual only
            elif mha_q_skip_type == 'text':
                # attn_output contains [prefix, vision], map to correct positions
                hidden_states[:, :text_start_idx, :] += attn_output
                # Text tokens (text_start_idx:) keep residual only
            elif mha_q_skip_type == 'all':
                # attn_output contains [prefix] only
                hidden_states[:, :image_start_idx, :] += attn_output
                # Vision and text tokens keep residual only

        # Fully Connected (FFN)
        residual = hidden_states
        norm_hidden_states = self.post_attention_layernorm(hidden_states)
        
        if is_decoding or ffn_skip_type == 'none':
            hidden_states = residual + self.mlp(norm_hidden_states)
        else:
            # Extract tokens to compute
            compute_input = get_skip_regions_and_compute_tensor(
                ffn_skip_type, image_start_idx, text_start_idx, norm_hidden_states
            )
            
            ffn_output = self.mlp(compute_input)
            
            # Map FFN output back to correct positions
            hidden_states = residual.clone()
            if ffn_skip_type == 'vision':
                # ffn_output contains [prefix, text]
                hidden_states[:, :image_start_idx, :] += ffn_output[:, :image_start_idx, :]
                hidden_states[:, text_start_idx:, :] += ffn_output[:, image_start_idx:, :]
                # Vision tokens keep residual only
            elif ffn_skip_type == 'text':
                # ffn_output contains [prefix, vision]
                hidden_states[:, :text_start_idx, :] += ffn_output
                # Text tokens keep residual only
            elif ffn_skip_type == 'all':
                # ffn_output contains [prefix] only
                hidden_states[:, :image_start_idx, :] += ffn_output
                # Vision and text tokens keep residual only

        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs
    

class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)

        self.n_visual = 0
        # Every layer is SearchVDecoderLayer so update_skip_configs can address each layer.
        self.replaced_layers: List[int] = list(range(config.num_hidden_layers))

        self.layers = nn.ModuleList(
            [
                SearchVDecoderLayer(config, layer_idx) if layer_idx in self.replaced_layers else LlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        
        image_start_idx = 35  # prefix
        text_start_idx = 35 + self.n_visual  # text
        total_seq_len = hidden_states.shape[1]
        bsz = hidden_states.shape[0]
        device = hidden_states.device
        vision_len = self.n_visual

        # NOTE:
        # - `attention_mask` above may include padding information (esp. with left padding in batched eval).
        # - SearchV only uses a custom SDPA mask for layers where Q is shortened.
        # - For `skip_type='none'`, we keep `None` to allow SDPA's native `is_causal` path.
        attention_masks_cache = {"none": None}
        is_prefill = total_seq_len > 1  

        if is_prefill:
            # -------------------------- 2. vision (q_len = prefix + text) --------------------------
            q_len_vision = image_start_idx + (total_seq_len - text_start_idx)
            # Start fully visible, then mask disallowed positions.
            sdpa_attention_mask = torch.ones(q_len_vision, total_seq_len, dtype=torch.bool, device=device)
            
            # Prefix queries: causal within prefix only
            sdpa_attention_mask[:image_start_idx, :] = torch.cat([
                torch.tril(torch.ones(image_start_idx, image_start_idx, dtype=torch.bool, device=device)),
                torch.zeros(image_start_idx, total_seq_len - image_start_idx, dtype=torch.bool, device=device)
            ], dim=-1)
            
            # Text queries: see prefix + vision + causal text
            text_q_count = total_seq_len - text_start_idx
            sdpa_attention_mask[image_start_idx:, :] = torch.cat([
                torch.ones(text_q_count, text_start_idx, dtype=torch.bool, device=device),
                torch.tril(torch.ones(text_q_count, text_q_count, dtype=torch.bool, device=device))
            ], dim=-1)
            
            attention_masks_cache['vision'] = sdpa_attention_mask.unsqueeze(0).unsqueeze(0).expand(bsz, -1, -1, -1)
            
            # If you need 'text'/'all' masks later, add them back carefully and make sure
            # they are combined with the external/padding mask semantics in batched inference.
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            current_attention_mask = None
        
            if is_prefill:
                if isinstance(decoder_layer, SearchVDecoderLayer):
                    layer_skip_type = decoder_layer.mha_skip_type
                else:
                    layer_skip_type = 'none'

                # Keep `None` for 'none' to match SDPA causal path.
                current_attention_mask = attention_masks_cache.get(layer_skip_type, None)
            else:
                current_attention_mask = None

            if self.gradient_checkpointing and self.training:
                # Training: gradient checkpointing path
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    current_attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    current_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **({'n_visual': self.n_visual} if isinstance(decoder_layer, SearchVDecoderLayer) else {})
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
    def update_skip_configs(self, skip_config):
        
        assert len(skip_config) == len(self.layers)
        
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, SearchVDecoderLayer):
                config = skip_config[idx]
                layer.mha_skip_type = config[0].lower()
                layer.ffn_skip_type = config[1].lower()
                
    def get_skip_configs(self):
        
        skip_config = []
        
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, SearchVDecoderLayer):
                mha_skip_type = layer.mha_skip_type
                ffn_skip_type = layer.ffn_skip_type
                skip_config.append((mha_skip_type, ffn_skip_type))
            else:
                # Keep index-aligned output for all layers.
                skip_config.append(("none", "none"))
        return skip_config

    

_CONFIG_FOR_DOC = "LlamaConfig"
class SearchVLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)

        else:
            logits = self.lm_head(hidden_states)
            

        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LlavaLlamaForCausalLM(SearchVLlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        n_tokens = input_ids.shape[1] if input_ids is not None else (inputs_embeds.shape[1] if inputs_embeds is not None else 0)
        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
            
            if inputs_embeds is not None:
                self.model.n_visual = inputs_embeds.shape[1] - n_tokens + 1


        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        inputs = input_ids
        n_tokens = inputs.shape[1]
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        self.model.n_visual = inputs_embeds.shape[1] - n_tokens + 1

        

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
