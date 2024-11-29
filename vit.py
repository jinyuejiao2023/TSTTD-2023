import torch
from torch import nn
import numpy as np
from einops import rearrange
torch.set_default_tensor_type(torch.cuda.FloatTensor)

def setup_seed(seed):
    # torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
            ]))
        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth - 1):
            self.skipcat.append(nn.Conv2d(num_channel + 1, num_channel + 1, [1, 2], 1, 0))

    def forward(self, x, mask=None):
        if self.mode == 1:
            for attn, ff in self.layers:
                x = attn(x, mask=mask)
                x = ff(x)
        elif self.mode == 2:
            last_output = []
            nl = 0
            for attn, ff in self.layers:
                last_output.append(x)
                if nl > 0:
                    x = self.skipcat[nl - 1](
                        torch.cat([x.unsqueeze(3), last_output[nl - 1].unsqueeze(3)], dim=3)).squeeze(3)
                x = attn(x, mask=mask)
                x = ff(x)
                nl += 1
        return x

def gain_neighborhood_band(x_train, band, band_patch):
    nn = band_patch // 2 #做除法并向下取整
    x_train_band = torch.Tensor(np.ones(((x_train.shape[0], band_patch,band))))

    # 中心区域
    x_train_band[:,nn:(nn+1),:] = x_train
    #左边镜像
    for i in range(nn):
            x_train_band[:,i:(i+1),:(nn-i)] = x_train[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train[:,0:1,:(band-nn+i)]
    #右边镜像
    for i in range(nn):
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train[:,0:1,(i+1):]
    return x_train_band

def data_normal(data):
    d_min = data.min()
    if d_min < 0:
        data = data + torch.abs(d_min)
        d_min = data.min()
    d_max = data.max()
    dst = d_max - d_min
    norm_data = (data - d_min).true_divide(dst + 1e-8)
    return norm_data


def PearsonCorrelation(tensor_1,tensor_2):
    x = tensor_1
    y = tensor_2
    vx = x - torch.mean(x, dim=1).unsqueeze(1)
    vy = y - torch.mean(y, dim=1).unsqueeze(1)
    cost = torch.sum(vx * vy, dim=-1) / (1e-8 + torch.norm(vx, 2, dim=-1) * torch.norm(vy, 2,dim=-1))
    return cost

class ViT(nn.Module):

    def __init__(
        self,
        num_layers: int,
        num_patches: int,
        patch_dim: int,
        embedding_dim: int,
        num_heads: int,
        dropout_rate: float,
        attention_mlp_hidden: int,
        emb_dropout: float,
    ):
        super(ViT, self).__init__()
        #-----------------------参数---------------------
        self.num_layers = num_layers
        self.num_patches = num_patches #band_num
        self.patch_dim = patch_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention_mlp_hidden = attention_mlp_hidden
        setup_seed((torch.rand(1) * 10000).int().item())

        #-----------------------网络结构------------------
        self.linear_proj = nn.Linear(self.patch_dim, self.embedding_dim) # 3->embedding_dim
        self.class_token = nn.Parameter(torch.rand((1, 1, self.embedding_dim)))  # randn or zeros? authors said zeros
        # Learnable Positional Embedding, as opposed to sinusoidal
        self.position_embeddings = nn.Parameter(torch.rand((1, num_patches + 1, embedding_dim)))
        self.dropout1 = nn.Dropout(emb_dropout)
        self.transformers = Transformer(embedding_dim, num_layers, num_heads, 16, self.attention_mlp_hidden,
                                        self.dropout_rate, self.num_patches, 2)
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        target = patches[0]
        candidate = patches[1]
        target1 = target
        candidate1 = candidate

        # ----------------emb-----------------
        # ---------batch_size = 64------------
        patch_embeddings_0 = torch.cat([target1, candidate1])#(256,193)
        patch_embeddings = patch_embeddings_0#(256,193)

        #------------pixel exbanded-----------
        x_train = patch_embeddings.unsqueeze(1)#(256,1,193)
        x_train_band = gain_neighborhood_band(x_train, self.num_patches, self.patch_dim)#(256,band_patch,193)
        x_train_band = torch.transpose(x_train_band, 2, 1) # (256,193,band_patch)
        patch_embeddings = self.linear_proj(x_train_band)# (256,193,embedding_dim)
        B = patch_embeddings.shape[0]                       #batchsize*4 = 128*4 = 512
        # --------------cls+pos-------------------------------
        class_tokens = self.class_token
        class_tokens = class_tokens.repeat(B, 1, 1)
        patch_embeddings1 = torch.cat([class_tokens, patch_embeddings], dim=1)
        position_embeddings = self.position_embeddings
        position_embeddings = position_embeddings.repeat(B, 1, 1)
        out1 = patch_embeddings1 + position_embeddings#(256,194,48)

        # --------------encoder----------------------
        out1 = self.dropout1(out1)
        out = self.transformers(out1)#(256,194,48)
        # ---------------取cls_token-----------------
        class_head = out[:, 0, :].squeeze(1)#(batchsize*4, embedding_dim)  #(256,48)

        # ------------------score--------------------
        target1_s = class_head[:target.shape[0]]#target_single(1,class_number)
        target1 = target1_s.repeat(candidate.shape[0], 1)#(batchsize*2,class_number)
        candidate1 = class_head[target.shape[0]:]#(batchsize*2,class_number)
        corr1 = nn.functional.cosine_similarity(target1, candidate1)
        score_pred = corr1.abs().clamp(None, 1).squeeze()#128(batch_size*2)
        return score_pred, class_head

    def loss(self, label, pred, class_head, corr_contain=0.4, margin = 0.8):
        assert label.ndim == pred.ndim
        loss_n = nn.BCEWithLogitsLoss()
        loss_cls = loss_n(pred, label)

        #----triplet_loss------
        candidate1 = class_head[1:]  # (batchsize*2,1,class_number)
        neg = candidate1[:64]
        pos = candidate1[64:]
        sim_pn = nn.functional.cosine_similarity(neg, pos).abs().squeeze()
        sim_an = pred[:64]
        sim_ap = pred[64:]
        sim_star = torch.max(sim_pn, sim_an)
        loss_trip = torch.mean(sim_star - sim_ap + 0.8)
        loss = loss_cls + loss_trip
        return loss
