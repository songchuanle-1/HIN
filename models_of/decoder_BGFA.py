import torch
from torch import nn
from torch.nn import functional as F
from common.models.containers import Module, ModuleList
from common.models.transformer.attention import MultiHeadAttention, OSAttention
from common.models.transformer.utils import PositionWiseFeedForward, sinusoid_encoding_table

class DecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(DecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True)
        self.grid_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.region_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.grid_cross_region = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.region_cross_grid = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.self_cross_enc = MultiHeadAttention(d_model, d_k, d_v, h, dropout, attention_module=OSAttention)
        self.self_cross_enc_cross = MultiHeadAttention(d_model, d_k, d_v, h, dropout, attention_module=OSAttention)

        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.proj1 = nn.Linear(d_model, d_model)
        self.proj2 = nn.Linear(d_model, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(d_model)

        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, input, grid_features, region_features, mask_pad, mask_self_att, mask_enc_att=None):
        self_att = self.self_att(input, input, input, mask_self_att)
        self_att = self_att * mask_pad


        grid_att = self.grid_att(self_att, grid_features, grid_features)
        grid_att = grid_att * mask_pad
        # print(grid_features.size())
        # print(region_features.size())
        region_att = self.region_att(self_att, region_features, region_features, mask_enc_att)
        region_att = region_att * mask_pad

        grid_cross_att = self.grid_cross_region(grid_att, region_features, region_features, mask_enc_att)
        grid_cross_att = grid_cross_att * mask_pad

        region_cross_att = self.region_cross_grid(region_att, grid_features, grid_features)
        region_cross_att = region_cross_att * mask_pad

        enc_features = torch.stack([grid_att, region_att], dim=-2)
        enc_att = self.self_cross_enc(self_att, enc_features, enc_features)
        enc_att = enc_att * mask_pad

        enc_cross_features = torch.stack([grid_cross_att, region_cross_att], dim=-2)
        # print(self_att.size())
        # print(enc_cross_features.size())
        # print('\n')
        enc_att_cross = self.self_cross_enc_cross(self_att, enc_cross_features, enc_cross_features)
        enc_att_cross = enc_att_cross * mask_pad

        gate = self.gate(self_att+enc_att+enc_att_cross)
        enc_att_final = self.proj1(enc_att) * gate + self.proj2(enc_att_cross) * (1-gate)
        enc_att_final = self.lnorm2(self_att + self.dropout2(enc_att_final))
        enc_att_final = enc_att_final * mask_pad

        ff = self.pwff(enc_att_final)
        ff = ff * mask_pad
        return ff

class TransformerDecoder(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, return_logits = False):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.num_queries = 5

        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList([DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec
        self.return_logits = return_logits
        self.g_num=256
        self.r_num=50
        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, grid_features, region_features, mask_encoder=None):

        # input (b_s, seq_len)
        b_s, seq_len = input.shape[:2]

        # split_num = grid_features.shape[0]//b_s
        # grid_features
        mask_queries = (input != self.padding_idx).unsqueeze(-1)  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device), diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention
            self.running_seq.add_(1)
            seq = self.running_seq

        out = self.word_emb(input) + self.pos_emb(seq)
        for i, l in enumerate(self.layers):
            out = l(out, grid_features[:,i*self.g_num:(i+1)*self.g_num,:], region_features[:,i*self.r_num:(i+1)*self.r_num,:], mask_queries, mask_self_attention, mask_encoder)

        out = self.fc(out)
        if self.return_logits:
            return out
        else:
            return F.log_softmax(out, dim=-1)

def build_decoder(vocab_size, max_len, N_dec, padding_idx,  
                  d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, **kwargs):
    Decoder = TransformerDecoder(vocab_size, max_len, N_dec, padding_idx,  
                               d_model, d_k, d_v, h, d_ff, dropout)
                               
    return Decoder