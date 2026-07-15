from transformers import PretrainedConfig


class ZzhMindConfig(PretrainedConfig):
    model_type = "zzhmind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

#part one
import torch
import math
import torch.nn as nn
from torch.nn import init
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


#就是写个公式
class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float=1e-5):
        super().__init__()
      
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))
    def _norm(self,x):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)        
    
    def forward(self,x):
        return self.weight*self._norm(x.float()).type_as(x)
    
    
#part two
#RoPE & Yarn
#precompute_freqs函数用于提前计算RoPE编码所需的余弦和正弦旋转参数。它接受以下参数：
def precompute_freqs(
    dim: int,#维度
    end: int = int(32*1024),#传入序列长度，缺省32*1024
    rope_base: float = 1e6,#默认1e6，为公式中提到的base
    rope_scaling: Optional[dict] =None,#RoPE 长度外推缩放配置
):
    freqs, attn_factor = (
        #feqs为标准RoPE频率，attn_factor为注意力温度补偿系数
        1.0 / (rope_base ** (torch.arange(0,dim,2)[:(dim//2)].float() / dim)),#RoPE_Core_Frequency_Formula
        1.0          
    )
    
    if rope_scaling is not None:
        orig_max,factor,beta_fast,beta_slow,attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )
        if end /orig_max > 1.0:#输入长度大于原始最大长度时，进行缩放
            
            #inv_dim: 频率逆向变换,是一个函数
            inv_dim = lambda b: (dim * math.log(orig_max/(b*math.pi*2)))/(
                2 * math.log(rope_base))#返回高低频维度分割下标
            
            low, high = (
                #floor 下取整
                max(math.floor(inv_dim(beta_fast)), 0),
                
                #ceil 上取整
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            )
            
            
            
            
              # 5. 计算混合因子 γ (Ramp)
#             # 在 low 之前，ramp 为 0；在 high 之后，ramp 为 1；在 low 和 high 之间，线性过渡。
#             # clamp 函数限制了数值只能在 [0, 1] 之间。

            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )
            
            
            # 6. 频率融合公式：f'(i) = f(i) * ((1-γ) + γ/s)
            # 当 ramp=0 时（高频）：系数为 1，保持原频率不变。
            # 当 ramp=1 时（低频）：系数为 1/factor，即对频率进行线性插值缩放。
            # ramp在0-1之间时：平滑过渡。
            
            freqs = freqs * (1-ramp + ramp / factor)


    # 7. 根据目标长度 end，生成位置索引向量 t
    #device=freqs.device 放到一张显卡
    t = torch.arange(end, device=freqs.device)
    
    #将位置 t 与处理好的频率 freqs 相乘，得到每个位置的旋转角度 θ
    freqs = torch.outer(t, freqs).float()
    
    
    # 9. 计算 Cos 和 Sin，并应用注意力补偿系数 (attn_factor)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    
    freqs_sin = torch.cat([torch.sin(freqs),torch.sin(freqs)],dim=-1) *attn_factor
    
    return freqs_cos, freqs_sin



    


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed


#part three
###GQA


#repeat_kv:扩展kv头数量，使得与q的头数量匹配，进而便于后续attenion 计算
def repeat_kv(x :torch.Tensor, n_rep: int)-> torch.Tensor:
    #bs batch size : 输入的样本数
    #slen : sequence length 输入序列的长度
    #num _key_value_heads:  kv注意力头的数
    #dim ： 头的维度
    #n_rep: 重复的次数
    
    bs, slen,num_key_value_heads,head_dim = x.shape
    
    if n_rep == 1 :
        return x
    
    return (x[:,:,:,None,:]
            .expand(bs,slen,num_key_value_heads,n_rep,head_dim)
            .reshape(bs,slen,num_key_value_heads*n_rep,head_dim)
            )
    
    
class Attention(nn.Module):
    """
    根据模型配置计算 GQA 的头数关系，
    创建 Q、K、V、输出投影层以及 Dropout。

    __init__ 只负责准备组件；
    真正的注意力计算在 forward() 中完成。
    """

    def __init__(self, args: ZzhMindConfig):
        super().__init__()

        # 确定K/V头数：
        # 如果没有指定，则令K/V头数等于Q头数，退化为普通MHA
        self.num_key_value_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )

        # Q头数必须能被K/V头数整除，才能平均分组
        assert args.num_attention_heads % self.num_key_value_heads == 0

        # hidden_size必须能平均拆分成多个Q头
        assert args.hidden_size % args.num_attention_heads == 0

        # Q的头数
        self.n_local_heads = args.num_attention_heads

        # K/V的头数
        self.n_local_kv_heads = self.num_key_value_heads

        # 每个K/V头需要对应多少个Q头
        self.n_rep = self.n_local_heads // self.n_local_kv_heads

        # 每个注意力头的维度
        self.head_dim = args.hidden_size // args.num_attention_heads

        # Q投影
        self.q_proj = nn.Linear(
            args.hidden_size,
            args.num_attention_heads * self.head_dim,
            bias=False,
        )

        # K投影
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )

        # V投影
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )

        # 输出投影
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=False,
        )

        # 注意力权重Dropout
        self.attn_dropout = nn.Dropout(args.dropout)

        # Attention输出Dropout
        self.resid_dropout = nn.Dropout(args.dropout)

        # 保存Dropout概率，供函数式注意力接口使用
        self.dropout = args.dropout

        # 判断是否可以并且允许使用PyTorch高效注意力
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attention
        )

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache = False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # bsz：一次输入的样本数
        # seq_len：每个样本的token数量
        # x：[bsz, seq_len, hidden_size]
        bsz, seq_len, _ = x.shape

        # 生成Q、K、V
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        # 拆分注意力头
        # Q：[bsz, seq_len, n_heads, head_dim]
        xq = xq.view(
            bsz,
            seq_len,
            self.n_local_heads,
            self.head_dim,
        )

        # K/V：[bsz, seq_len, n_kv_heads, head_dim]
        xk = xk.view(
            bsz,
            seq_len,
            self.n_local_kv_heads,
            self.head_dim,
        )

        xv = xv.view(
            bsz,
            seq_len,
            self.n_local_kv_heads,
            self.head_dim,
        )

        # 取出RoPE的cos和sin
        cos, sin = position_embeddings

        # 给Q、K添加旋转位置编码
        xq, xk = apply_rotary_pos_emb(
            xq,
            xk,
            cos,
            sin,
        )
        
        #凭借KV cache
        
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0],xk],dim=1)
            xv = torch.cat([past_key_value[1],xv],dim=1)
        
        past_kv = (xk,xv) if use_cache else None
        
        #扩展 K/V 头数并调整维度
        
        xq,xk,xv = (
            xq.transpose(1,2),
            repeat_kv(xk,self.n_rep).transpose(1,2),
            repeat_kv(xv,self.n_rep).transpose(1,2)
        )
        
        #判断是否使用PyTorch的高效注意力
        if (
            self.flash
            and (seq_len > 1)
            and (past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
            ):
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
                )
            
        else:
            scores = (xq @ xk.transpose(-2, -1))/(math.sqrt(self.head_dim))
            
            scores[:,:,:,-seq_len:] +=(
                torch.triu(
                    
                    torch.full(
                        (seq_len,seq_len),
                        
                        float("-inf"),
                        device=scores.device,
                        ),
                    diagonal=1
                )
            )
            #添加 Padding Mask
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0- extended_attention_mask)* -1e9 
                scores = scores + extended_attention_mask
            
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            
            scores = self.attn_dropout(scores)
            
            output = scores @ xv    
            
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
    
    
#part four 
#FNN

#总体思路，升维度，降维度，以进行详细化
class FeedForward(nn.Module):
    
    def __init__(self,config:ZzhMindConfig):
        super().__init__()
        
        
        #如果没有显式指定中间层大小，则使用公式进行计算
        if(config.intermediate_size is None):
            
            #8/3好用
            intermediate_size = int((8*config.hidden_size)//3)
            
            #保证是64倍数，上取整
            config.intermediate_size = (
                64*((intermediate_size+64 - 1)//64)
            )
            
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,

        )
        
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )
        
        #dropout是Dropout层，可以理解为一个函数
        #丢弃概率为config.dropout的dropout函数
        self.dropout = nn.Dropout(config.dropout)
        
        #根据配置里的激活函数名称，从 ACT2FN 映射表中取出对应的激活函数。
        self.act_fn = ACT2FN[config.hidden_act]
    def forward(self,x):
        
        #升维
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        
        #降维返回
        return self.dropout(self.down_proj(gated))
    
    
class ZzhMindBlock(nn.Module):
    def __init__(self,layer_id:int,config:ZzhMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attention = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = (
            FeedForward(config)
            if not config.use_moe
            else MoEFeedForward(config)  # ！修正：原MoEFeedForaward拼写错误
        )

    def forward(
        self,
        hidden_states,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        res = hidden_states

        hidden_states, present_key_value = self.self_attention(
            self.input_layernorm(hidden_states),  # pre-norm
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )

        hidden_states = res + hidden_states

        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present_key_value
    
class ZzhMindModel(nn.Module):
    def __init__(self,config:ZzhMindConfig):
        super.__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = (
            config.vocab_size,
            config.num_hidden_layers,
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [MokioMindBlock(l, config) for l in range(self.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        # input_ids: [bsz, seq_len]
        batch_size, seq_length = input_ids.shape

        if hasattr(past_key_values, "layers"):
            past_key_values = None

        past_key_values = past_key_values or [None] * len(self.layers)

        # 计算start_pos：如果存在past，则start_pos为已有past序列长度
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        # Embedding + dropout
        hidden_states = self.dropout(
            self.embed_tokens(input_ids)
        )  # [bsz, seq_len, hidden]

        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_length],
            self.freqs_sin[start_pos : start_pos + seq_length],
        )
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(
            zip(self.layers, past_key_values)
        ):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            [
                layer.mlp.aux_loss
                for layer in self.layers
                if isinstance(
                    layer.mlp, MoEFeedForward
                )  # ！修正：原MoEFeedForaward拼写错误
            ],
            hidden_states.new_zeros(1).squeeze(),
        )

        return hidden_states, presents, aux_loss        
    

        
        
        
            
                
                
                
            
        
        
            
            
    
    
    
    

    