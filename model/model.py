from transformers import PretrainedConfig


class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

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
import torch.nn as nn

class RmsNorm(nn.Module):
    def __init__(self,dim:int,eps:float=1e-5):
        super().__init__()
        self.dim=dim
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))
    def _norm(self,x):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)        
    
    def forward(self,x):
        return self.weight*self._norm(x.float()).type_as(x)
    
    
#part two

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
            orig_max = rope_scaling.get("original_max_position_embeddings", 2048),#模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
            factor = rope_scaling.get("factor", 16),#要扩展的倍数 s (比如从 2k 扩展到 32k，factor 就是 16)
            beta_fast = rope_scaling.get("beta_fast", 32.0),#高频边界，波长比例大于此值的维度不缩放
            beta_low = rope_scaling.get("beta_slow", 1.0),#低频边界，波长比例小于此值的维度全量缩放
            attn_factor = rope_scaling.get("attention_factor", 1.0)

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
                (torch.arange(dim//2, device = freqs.device()).float() - low),
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
