####################################################
# Author: fvilmos
# https://github.com/fvilmos
####################################################

import torch
import torch.nn as nn
from utils.vision_encoder import VisionProjection

class DecoderLayer(nn.Module):
    """
    Decoder layer with self and cross-attention
    """
    def __init__(self, dim, num_heads,mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
           nn.Linear(dim, int(mlp_ratio * dim)),
           nn.GELU(),
           nn.Dropout(dropout),
           nn.Linear(int(mlp_ratio * dim), dim)
        )
    
    def forward(self, x, vision, causal_mask,return_att=False):
        # masked self-attention
        xl, _ = self.self_attn(self.norm1(x),self.norm1(x),self.norm1(x),attn_mask=causal_mask)
        x = x + xl
        
        # cross-attention
        att = []
        if return_att:
           # B, embeding_dim, 197 = <CLS> + 196 (if image 224, patch 16 => 14x14 = 196)
           xl, att = self.cross_attn(self.norm2(x), vision, vision, need_weights=return_att, average_attn_weights=False)
        else:
           xl, _ = self.cross_attn(self.norm2(x), vision, vision, need_weights=False, average_attn_weights=True)
        x = x + xl
        x = x + self.ffn(self.norm3(x))
        
        ret = (x, att) if return_att==True else x
        return ret

class TextEmbedding(nn.Module):
    """
    Prepare text tokens 
    """
    def __init__(self, vocab_size, dim, max_len=50):
        super().__init__()
        self.token = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_len, dim)

    def forward(self, input_ids):
        _, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device)
        return self.token(input_ids) + self.pos(pos)

class ImageCaptioner(nn.Module):
    """
    Assamble the captioner from Vision encoder - text and decoder, fusing the visual tokens with thext
    """
    def __init__(self,vocab_size,dim=768,num_heads=8,num_layers=4, vis_out_dimension=512, vis_hxw_out=49, freeze_vision=True, VisionEncoder=None, max_len=100):
        super().__init__()
        assert VisionEncoder !=None, "provide a vision encoder class"
        self.vision = VisionEncoder()
        self.vision_proj = VisionProjection(vis_out_dimension,dim, vis_hxw_out)
        self.text_embed = TextEmbedding(vocab_size, dim, max_len=max_len)
        self.decoder_layers = nn.ModuleList([
           DecoderLayer(dim, num_heads) for _ in range(num_layers)
        ])
        
        self.lm_head = nn.Linear(dim, vocab_size)
        if freeze_vision:
           for p in self.vision.parameters():
               p.requires_grad = False

    def forward(self, images, input_ids, return_att=False):
        with torch.no_grad():
           vision = self.vision(images)
        vision = self.vision_proj(vision)
        x = self.text_embed(input_ids)
        L = input_ids.size(1)
        causal_mask = torch.triu(torch.ones(L, L, device=input_ids.device),diagonal=1).bool()
        
        # hold only the last layer for attention
        att =[]
        for layer in self.decoder_layers:
           if return_att:
               x, att = layer(x, vision, causal_mask,return_att=return_att)
           else:
               x = layer(x, vision, causal_mask,return_att=return_att)
        ret = (self.lm_head(x), att[-1]) if return_att else self.lm_head(x)
        return ret


@torch.no_grad()
def generate_caption(model,image,voc,max_len=50,return_att=False):
    """
    Generate caption in autoregressive manner
    """
    model.eval()
    image = image.unsqueeze(0)
    start_token_id = voc('<start>')
    end_token_id = voc('<end>')
    device = image.device
    input_ids = torch.tensor([[start_token_id]], device=device)
    for _ in range(max_len):
       if return_att == True:
           logits, att = model(image, input_ids,return_att=return_att)
       else:
           logits = model(image, input_ids)
       next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
       input_ids = torch.cat([input_ids, next_token], dim=1)
       if next_token.item() == end_token_id:
           break
    ret = [voc.idx2word[id] for id in list(input_ids.detach().cpu().numpy()[0])]
    return ret, att if return_att == True else ret
