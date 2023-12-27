import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import opt_einsum as oe

d_model = 32
l_max = 8
batch_size = 2

TERMS = [
    0, 
    1,
    2
]
use_denom = True
CONSTANTS = True


class TaylorExp(nn.Module):
    """
    Feature map to compute 2nd-order Taylor approx. of exp(q^T k / sqrt(d))
    """

    def __init__(self, input_dim, head_dim_idx, temp=None, eps=1e-12):
        super().__init__()

        self.input_dim = input_dim
        self.head_dim_idx = head_dim_idx
        self.temp = 1.0 if temp is None else temp
        self.eps = eps

        self.r2 = math.sqrt(2) if CONSTANTS else 1
        self.rd = math.sqrt(self.input_dim) if CONSTANTS else 1
        self.rrd = math.sqrt(self.rd) if CONSTANTS else 1

    # Running these in parallel
    def forward(self, x: torch.Tensor):
        # Get 2nd-order terms (rearrange(x * x), '... m n -> ... (m n)')
        x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(start_dim=-2) / self.r2
        term1 = torch.ones(x[..., :1].shape).to(x.device)
        term2 = x / self.rrd
        term3 = x2 / self.rd
        terms = [term1, term2, term3]
        return torch.cat([terms[t] for t in TERMS], dim=self.head_dim_idx)


class LinAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.feature_dim = 16
        self.num_heads = 2
        self.num_key_value_heads = 2
        self.head_dim = self.d_model // self.num_key_value_heads
        self.tensor_core_size = 4
        self.eps = 1e-12
        self.causal = True

        feature_map_kwargs = {
            "input_dim": self.feature_dim,
            "head_dim_idx": -1,
            "temp": 1.0,
            "eps": 1e-12,
        }
        self.feature_map = TaylorExp(**feature_map_kwargs)
        self.proj_q = nn.Linear(
            self.d_model, self.feature_dim * self.num_heads, bias=False
        )
        self.proj_k = nn.Linear(
            self.d_model, self.feature_dim * self.num_heads, bias=False
        )
        self.proj_v = nn.Linear(
            self.d_model, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.proj_o = nn.Linear(
            self.num_heads * self.head_dim, self.d_model, bias=False
        )
        self.dropout = nn.Identity()

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        b, l, _ = hidden_states.size()
        q = self.proj_q(hidden_states)
        k = self.proj_k(hidden_states)
        v = self.proj_v(hidden_states)
        q = q.view(b, l, self.num_heads, self.feature_dim).transpose(1, 2)
        k = k.view(b, l, self.num_key_value_heads, self.feature_dim).transpose(1, 2)
        v = v.view(b, l, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Linear attention
        q, k = self.feature_map(q), self.feature_map(k)
        q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)

        # Compute attention causal
        num = (q * (k * v).cumsum(dim=2)).sum(dim=-1) 
        if use_denom:
            print(f"Using denom.")
            denom = (q * k.cumsum(dim=2)).sum(dim=-1) + self.eps
            y = (num / denom)
        else:
            y = num

        y = rearrange(y, "b h l d -> b l (h d)")
        y = self.proj_o(y)
        y = self.dropout(y)
        return y.to(hidden_states.dtype)


class LinAttnFunction(torch.autograd.Function):
    """
    We can implement custom autograd by subclassing torch.autograd.Function.
    """

    @staticmethod
    def forward(ctx, Q, K, V, feature_map_kwargs):
        """
        ctx is a context to save info for backward, using ctx.save_for_backward
        """
        input_dim = feature_map_kwargs["input_dim"]
        head_dim_idx = feature_map_kwargs["head_dim_idx"]
        temp = 1.0
        eps = feature_map_kwargs["eps"]

        r2 = math.sqrt(2) if CONSTANTS else 1
        rd = math.sqrt(input_dim) if CONSTANTS else 1
        rrd = math.sqrt(rd) if CONSTANTS else 1

        print(f"Causal!")
        n = Q.shape[2]

        # compute for A2 block
        A2 = torch.einsum("bhnd,bhnf,bhne->bhndef",K,V,K).cumsum(dim=2) / (rd * r2)
        Q2 = torch.einsum("bhnd,bhne->bhnde", Q, Q) / (rd * r2)
        T2    = torch.einsum("bhnde,bhndef->bhnf", Q2, A2) 

        # compute for A1 block
        # A1 = torch.einsum("nm,bhmd,bhme->bhnde",cumsum_matrix,K,V) / (rrd)
        A1 = torch.einsum("bhnd,bhne->bhnde",K,V).cumsum(dim=2) / (rrd)
        Q1 = Q / (rrd)
        T1 = torch.einsum("bhnd,bhnde->bhne", Q1, A1)  

        # compute for A0 block
        K0 = torch.ones(Q[..., :1].shape).to(Q.device)
        Q0 = torch.ones(Q[..., :1].shape).unsqueeze(-1).to(Q.device)
        T0 = V.cumsum(dim=2)

        # denom = ((Q * K.sum(dim=2, keepdim=True)).sum(dim=-1) + eps)
        K2 = torch.einsum("bhnd,bhne->bhnde", K, K) / (rd * r2)
        D2 = torch.einsum("bhnde,bhnde->bhn", Q2, K2.cumsum(dim=2))  # sum(-1) on the final dim.
        D1 = torch.einsum("bhnd,bhnd->bhn", Q, K.cumsum(dim=2))/ ((rrd) ** 2)
        D0 =  K0.cumsum(dim=2).squeeze()

        # output
        numerators = [T0, T1, T2]
        denominators = [D0, D1, D2]
        numerator = sum(numerators[t] for t in TERMS)
        denominator = sum(denominators[t] for t in TERMS)

        if use_denom:
            print(f"Using denom.")
            result = torch.einsum("bhnd,bhn->bhnd", numerator, 1 / denominator)
        else:
            print(f"Using numerator only.")
            result = numerator

        ctx.save_for_backward(Q, K, V, Q2, K2, A2, A1, Q0, K0, numerator, denominator, torch.tensor(input_dim))
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        We receive a Tensor containing the gradient of the loss w/r/t the output,
        and we need to compute the gradient of the loss with respect to the input.
        """
        Q, K, V, Q2, K2, A2, A1, Q0, K0, numerator, denominator, input_dim = ctx.saved_tensors

        r2 = math.sqrt(2) if CONSTANTS else 1
        rd = math.sqrt(input_dim.item()) if CONSTANTS else 1
        rrd = math.sqrt(rd) if CONSTANTS else 1

        running_Q_grad = torch.zeros_like(Q)
        running_K_grad = torch.zeros_like(K)
        running_V_grad = torch.zeros_like(V)

        if use_denom:
            dl_d_numerator = torch.einsum("bhn,bhnd->bhnd", 1 / denominator, grad_output)
        else:
            dl_d_numerator = grad_output
        dl_d_denominator = torch.einsum("bhnd,bhnd->bhn", numerator, grad_output) * -1  / denominator ** 2

        n = Q.shape[2]
        rev_cumsum_matrix = torch.triu(torch.ones((n, n))).to(Q.device) # reverse cumsum

        # for the A2 block
        if 2 in TERMS:
            # numerator
            print(f"Backward: Num 2")
            dl_dA2_cs = torch.einsum("bhnd,bhne,bhnf->bhndef", Q, Q, dl_d_numerator) / (rd * r2)
            dl_dA2    = torch.einsum("nm,bhmdef->bhndef", rev_cumsum_matrix, dl_dA2_cs) 
            dl_dQ2    = torch.einsum("bhndef,bhnf->bhnde" , A2, dl_d_numerator) / (rd * r2)
            dl_dK2   = 2*torch.einsum("bhndef,bhnd,bhnf->bhne", dl_dA2, K, V)  / (rd * r2)
            dl_dQ2   = 2*torch.einsum("bhnde,bhnd->bhne", dl_dQ2, Q)
            dl_dV2 = torch.einsum("bhnd,bhne,bhndef->bhnf", K, K, dl_dA2) / (rd * r2)
            running_K_grad += dl_dK2
            running_Q_grad += dl_dQ2
            running_V_grad += dl_dV2

            # denominator
            if use_denom:
                print(f"Backward: Denom 2.")
                dl_dD2_cs = torch.einsum("bhnde,bhn->bhnde",  Q2, dl_d_denominator) 
                dl_dD2  = torch.einsum("nm,bhmde->bhnde", rev_cumsum_matrix, dl_dD2_cs) 
                dl_dK2_denom  = 2 * torch.einsum("bhnd,bhnde->bhne", K, dl_dD2) / (rd * r2)
                running_K_grad += dl_dK2_denom

                dl_dQ2_denom = torch.einsum("bhnde,bhn->bhnde", K2.cumsum(dim=2), dl_d_denominator)
                dl_dQ2_denom = 2 * torch.einsum("bhnde,bhne->bhnd", dl_dQ2_denom, Q) / (rd * r2)
                running_Q_grad += dl_dQ2_denom

        # for the A1 block
        if 1 in TERMS:
            # numerator
            print(f"Backward: Num 1")
            dl_dA1_cs = torch.einsum("bhnd,bhne->bhnde", Q, dl_d_numerator) 
            dl_dA1    = torch.einsum("nm,bhmde->bhnde", rev_cumsum_matrix, dl_dA1_cs) # reverse cumsum
            dl_dQ1    = torch.einsum("bhnde,bhne->bhnd", A1, dl_d_numerator)  / (rrd) 
            dl_dK1    = torch.einsum("bhnde,bhne->bhnd", dl_dA1, V) / (rd)
            dl_dV1    = torch.einsum("bhnd,bhndf->bhnf", K, dl_dA1) / (rd)
            running_Q_grad += dl_dQ1
            running_K_grad += dl_dK1
            running_V_grad += dl_dV1

            # denominator
            if use_denom:
                print(f"Backward: Denom 1.")
                dl_dD1_cs = torch.einsum("bhnd,bhn->bhnd", Q, dl_d_denominator)  / (rrd) ** 2
                dl_dK1_denom  = torch.einsum("nm,bhmd->bhnd", rev_cumsum_matrix, dl_dD1_cs)
                running_K_grad += dl_dK1_denom

                dl_dQ1_denom = torch.einsum("bhnd,bhn->bhnd", K.cumsum(dim=2), dl_d_denominator) / (rrd) ** 2
                running_Q_grad += dl_dQ1_denom

        # for the A0 block
        if 0 in TERMS:
            print(f"Backward: Num 0")
            # numerator
            dl_dA0_cs = torch.einsum("bhnde,bhnf->bhndf", Q0, dl_d_numerator) 
            dl_dA0 = torch.einsum("nm,bhmdf->bhndf", rev_cumsum_matrix, dl_dA0_cs)
            dl_dV0    = torch.einsum("bhnd,bhndf->bhnf", K0, dl_dA0) 
            running_V_grad += dl_dV0

            # denominator
            # none since V is not in the denominator
            print(f"Backward: Denom 0.")


        return running_Q_grad, running_K_grad, running_V_grad, None


class LinAttnManual(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.feature_dim = 16
        self.num_heads = 2
        self.num_key_value_heads = 2
        self.head_dim = self.d_model // self.num_key_value_heads
        self.tensor_core_size = 4
        self.eps = 1e-12
        self.causal = False

        self.feature_map_kwargs = {
            "input_dim": self.feature_dim,
            "head_dim_idx": -1,
            "temp": 1.0,
            "eps": 1e-12,
        }
        self.feature_map = LinAttnFunction
        self.proj_q = nn.Linear(
            self.d_model, self.feature_dim * self.num_heads, bias=False
        )
        self.proj_k = nn.Linear(
            self.d_model, self.feature_dim * self.num_heads, bias=False
        )
        self.proj_v = nn.Linear(
            self.d_model, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.proj_o = nn.Linear(
            self.num_heads * self.head_dim, self.d_model, bias=False
        )
        self.dropout = nn.Identity()

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        b, l, _ = hidden_states.size()
        q = self.proj_q(hidden_states)
        k = self.proj_k(hidden_states)
        v = self.proj_v(hidden_states)
        q = q.view(b, l, self.num_heads, self.feature_dim).transpose(1, 2)
        k = k.view(b, l, self.num_key_value_heads, self.feature_dim).transpose(1, 2)
        v = v.view(b, l, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        y = self.feature_map.apply(q, k, v, self.feature_map_kwargs)

        y = rearrange(y, "b h l d -> b l (h d)")
        y = self.proj_o(y)
        y = self.dropout(y)
        return y.to(hidden_states.dtype)


if __name__ == "__main__":
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    seq_mixer = LinAttn()
    seq_mixer_manual = LinAttnManual()

    # set the weights to be the same
    seq_mixer.proj_q.weight = torch.nn.Parameter(seq_mixer_manual.proj_q.weight.clone())
    seq_mixer.proj_k.weight = torch.nn.Parameter(seq_mixer_manual.proj_k.weight.clone())
    seq_mixer.proj_v.weight = torch.nn.Parameter(seq_mixer_manual.proj_v.weight.clone())
    seq_mixer.proj_o.weight = torch.nn.Parameter(seq_mixer_manual.proj_o.weight.clone())

    # input tensor
    x = torch.randn(batch_size, l_max, d_model)
    y = seq_mixer(x)
    print()
    y_manual = seq_mixer_manual(x)
    print()
    print(f"{y.shape=}")
    print(f"{y_manual.shape=}")

    # check that the outputs are the same from forward pass
    print(f"\nForward pass:")
    print(torch.norm(y - y_manual))

    # check that the backwards pass is the same
    print(f"\nBackward pass:")
    y.retain_grad()
    y.sum().backward()
    y_manual.sum().backward()

    # compare the gradients
    print(f"\nGradient max:")
    try:
        print(
            "proj_q: ",
            torch.max(seq_mixer.proj_q.weight.grad - seq_mixer_manual.proj_q.weight.grad),
        )
    except:
        print(f"Skipping q grad check.")
    try:
        print(
            "proj_k: ",
            torch.max(seq_mixer.proj_k.weight.grad - seq_mixer_manual.proj_k.weight.grad),
        )
    except:
        print(f"Skipping k grad check.")
    try:
        print(
            "proj_v: ",
            torch.max(seq_mixer.proj_v.weight.grad - seq_mixer_manual.proj_v.weight.grad),
        )
    except:
        print(f"Skipping v grad check.")
    print(
        "proj_o: ",
        torch.max(seq_mixer.proj_o.weight.grad - seq_mixer_manual.proj_o.weight.grad),
    )