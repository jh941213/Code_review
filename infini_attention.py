
'''
infini-attention 함수의 구조와 설명
repeat_kv function
키(Key)와 값(Value) 텐서들의 차원 수를 조정하는 함수입니다. 이 함수는 어텐션 헤드 수에 맞게 키와 값 벡터들을 반복 확장하여, 어텐션 메커니즘에서 더 많은 문맥 정보를 포함시킬 수 있도록 합니다.

InfiniAttention class
클래스는 초기화에서 다수의 설정을 포함하며, 포워드 패스에서 입력 텐서를 처리하는 로직을 구현합니다. 여기에는 메모리 업데이트 및 검색, 로터리 임베딩 적용, 그리고 최종 어텐션 스코어 계산이 포함됩니다.

RotaryEmbedding class
회전 위치 임베딩을 계산하여, 모델이 입력 시퀀스의 위치 정보를 보다 정확하게 처리할 수 있도록 합니다.
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import AutoConfig
# repeat_kv 함수
# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

'''
위함수의 목적은 텐서를 차원을 조작해서, 특정 차원을 반복확장 하는 것이다. Key, Value 텐선들의 차원수를 조정한다.

hidden_state.shape을 각각의 변수에 차원(batch, num_key_value_heads, seqlen, head_dim으로 구성이 되어있다)을 맵핑해서 넣어주고. n_rep(반복할 횟수)에 따라서 1이면 변환할 필요가 없어서 그냥 리턴하고, 1이 아니면 텐서에 새로운 축을 추가하고 텐서를 확장시킨다. 이 과정에서 key-value pair 가 반복되어, 확장 연산은 추가메모리를 사용하지 않고, 원본 텐서의 데이터를 재사용한다. 마지막 리턴할 때 reshape을 해서  최종 텐서의 형태를 만들어서 반환한다.

-> attention head의 key 와 value벡터들의 수를 인위적으로 늘려서 많은 문맥정보를 포함시키도록 한다.
'''

#infiniAttention Class

class InfiniAttention(nn.Module):
    def __init__(self, config: AutoConfig, layer_idx: Optional[int] = None):
        super().__init__()
        #파라미터
        self.config = config #autoconfig
        self.layer_idx = layer_idx #attention layer index
        self.hidden_size = config.hidden_size #입력 차원특성
        self.num_heads = config.num_attention_heads # 멀티헤드 갯수
        self.head_dim = self.hidden_size // self.num_heads # 헤드에서의 hidden size hiddensize / num_head
        self.num_key_value_heads = config.num_key_value_heads 
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads #key value group
        self.max_position_embeddings = config.max_position_embeddings # 포지셔널 임베딩
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
		
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        #프로젝션 레이터
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
		# 공간 정보를 기억하기 위한 로타리 임베딩
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
		#추가파라미터
        self.beta = nn.Parameter(torch.randn(1)) # memory 와 attention 결합시 사용되는 가중치 파라미터
        
        self.segment_size = 2048 #입력시퀀스 처리시 한번에 처리할 세그먼트의 크기

        '''
        파라미터를 선언해 주고,  프로젝션레이어에서는 Linear 선형변환을 정의한다. 여기서 o_projsms 최종 attention 결과를 hidden_size로 맵핑하는 역할이다. 로터리 임베딩은 하이퍼크로버에서도 쓰였는데 회전 불변의 방식을 제공해서 긴 시퀀스 간 처리에 유용해서 위치정보를 고유하게 유지한다
        '''

    #infiniAttention forward
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        M_Z: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        #초기 메모리 설정
        # Initialize memory and normalization term 
        if M_Z is None:
            M = torch.zeros(self.num_heads, self.head_dim, self.head_dim).to(hidden_states.device)
            z = torch.zeros(self.num_heads, self.head_dim).to(hidden_states.device)
        else:
            M, z = M_Z

        bsz, q_len, _ = hidden_states.size() #bsz, qlen, _ (batch_size, sequence_length, feature_dim)추출
		#입력 텐서 처리
        query_states = self.q_proj(hidden_states) #쿼리, 키, 값 벡터를 위한 텐서
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
		#view를 써서 mha 계산을 하기 위한 형태로 추출
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        #rotary positional embedding
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len) #cos, sin 값계산

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids) # 로터리 임베딩 적용해서 상대적인 위치를 보여줌
		
        #과거의 key-value 쌍 값 처리
        #이전값의 키와 밸류의 정보를 현재계산에서 재사용하고, 과거의 저장된 값과 함께 업데이트해서 cache_kwargs를 통해 cos, sin 도 같이 전달한다.
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
		
        #attention 마스크 검증(차원확인)		
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )


        # GQA
        # Memory retrieval and attention calculation per segment
        # 메모리 업데이트 및 검색
        memory_output = self._retrieve_from_memory(query_states, M, z) #메모리검색
        # Update memory with current segment's key and value states
        M, z  = self._update_memory(key_states, value_states, M, z) #새로운 키밸류 메모리 업데이트
        #위의 repeat_kv 함수 키와 벨류 상태를 attention head 에 맞게 반복확장
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
		#어텐션 계산
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
		#출력물 결합 및 최종 처리
        # attn_output 결과와 메모리로부터 검색된 출력(memory_output)을 결합합니다. 여기서 beta 파라미터를 사용하여 두 출력 사이의 결합 비율을 조절할 수 있습니다.
        combined_output = self._long_term_injection(attn_output, memory_output)
        
		
        # Prepare output for this segment
        # 차원조정 및 선형 프로젝션을 통해 차원 수정을 통해 최종 출력함
        combined_output = combined_output.transpose(1, 2).contiguous()
        combined_output = combined_output.view(bsz, q_len, self.hidden_size)
        final_output = self.o_proj(combined_output)
        return final_output, None, None, (M, z)
    
    '''
    포워드 하고 업데이트하는 과정인데 밑에 함수의 코드들이 있으니 연관 지어서 주석을 쭉 읽어보면 좋습니다. 메모리설정, key-value의 이전값을 업데이트 및 파라미터에 맞게 차원을 수정하고, 어텐션 계산해서 결과를 가져오는 코드.
    '''

    #infini Attention memory 처리,  long_term_injection 처리

#메모리검색함수
def _retrieve_from_memory(self, Q, M, z):
        # Retrieve context from compressive memory using linear attention (Eq. 3)
        M_s_1 = torch.matmul(F.elu(Q) + 1, M)
        Z_s_1 = torch.matmul(F.elu(Q) + 1, z.unsqueeze(-1)) + 1e-8
        A_mem = M_s_1 / Z_s_1
        return A_mem
        
#메모리업데이트
def _update_memory(self, K, V, M, z, use_delta=False):
    if use_delta: #논문에서 델타규칙, deltarule을 통해서 메모리를 업데이트 할지 결정
        V_retrieved = torch.matmul(F.elu(K).transpose(-2, -1) + 1, M) / (torch.matmul(F.elu(K).transpose(-2, -1) + 1, z.unsqueeze(-1)) + 1e-8)
        updated_M = M + torch.matmul(F.elu(K).transpose(-2, -1) + 1, V - V_retrieved)
    else:
        updated_M = M + torch.matmul(F.elu(K).transpose(-2, -1) + 1, V)

    updated_z = z + (F.elu(K) + 1).sum(dim=-2)
    M = updated_M.detach() #detach() 는 기존 계산 그래프와 분리해 메모리가 그래디언트 업데이트에 영향을 미치지 않도록 함
    z = updated_z.detach()
    return M, z
    
#장기기억 결합
# 어텐션결과와 이전 메모리 장기기억인 A_mem 을 결합하고 sigmoid를 통해서 0,과 1사이의 비율을 조정하는데 사용하고 A는 장기기억과 현재기억을 반영한 최종출력
def _long_term_injection(self, A_dot, A_mem):
    beta = torch.sigmoid(self.beta)
    A = beta * A_mem + (1 - beta) * A_dot
    return A

'''

'''
#RotaryEmbedding
#Transformer 모델의 회전 위치 임베딩(Rotary Position Embedding)을 계산하는 데 사용됩니다. 회전 위치 임베딩은 모델이 입력 시퀀스의 각 요소 위치 정보를 고려할 수 있도록 돕는 방법 중 하나로, 특히 Transformer 아키텍처에서 유용하게 사용됩니다.

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
 



