import torch
from torch import nn
from torch.nn import functional as F
from common.models.transformer.attention import MultiHeadAttention, NormSelfAttention
from common.models.transformer.utils import PolarRPE, PositionWiseFeedForward, RelationalEmbedding





class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(EncoderLayer, self).__init__()

        self.self_grid = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.self_region = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.global_grid = MultiHeadAttention(d_model, d_k, d_v, h, dropout, shortcut=False)
        self.global_region = MultiHeadAttention(d_model, d_k, d_v, h, dropout, shortcut=False)
        self.cls_num = 1
        self.cls_grid = nn.Parameter(torch.randn(1, self.cls_num, d_model), requires_grad=True)
        self.cls_region = nn.Parameter(torch.randn(1, self.cls_num, d_model), requires_grad=True)
        
        self.pwff_grid = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.pwff_region = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.golbal_cls_grid = MultiHeadAttention(d_model, d_k, d_v, h, dropout, shortcut=False)
        self.golbal_cls_region = MultiHeadAttention(d_model, d_k, d_v, h, dropout, shortcut=False)

        self.gate_golbal_cls=nn.Linear(2*d_model,2)


        self.middle_LN = nn.LayerNorm(d_model)
        self.middle_fc = nn.Sequential(
            nn.Linear(d_model,d_model),
            nn.ReLU()
        )


    def forward(self, gird_features, region_features, attention_mask):
        b_s = region_features.shape[0]
        cls_grid = self.cls_grid.expand(b_s, self.cls_num, -1)
        cls_region = self.cls_region.expand(b_s, self.cls_num, -1)
        cls_grid = self.global_grid(cls_grid, gird_features, gird_features)

        retion_att = attention_mask.repeat(1,1,self.cls_num,1)
        cls_region = self.global_region(cls_region, region_features, region_features, attention_mask=retion_att)


        # print(cls_region.size())
        # print(cls_grid.size())


        #### middle feature
        cls_grid_cross_att = self.golbal_cls_grid(cls_region,cls_grid,cls_grid)
        cls_region_cross_att = self.golbal_cls_region(cls_grid,cls_region,cls_region)
        combine_feature = torch.cat((cls_grid_cross_att,cls_region_cross_att),dim=-1)
        gate_weight = self.gate_golbal_cls(combine_feature)
        weights = F.softmax(gate_weight,dim=-1)
        weight1 = weights[:,:,0].unsqueeze(-1)
        weight2 = weights[:,:,1].unsqueeze(-1)
        middle_cls_feature = self.middle_LN(weight1*cls_grid_cross_att+weight2*cls_region_cross_att)
        middle_cls_feature = self.middle_fc(middle_cls_feature)

        middle_cls_grid_feature = torch.cat([middle_cls_feature,cls_grid], dim=1)
        middle_cls_region_feature = torch.cat([middle_cls_feature,cls_region], dim=1)

        gird_features = torch.cat([middle_cls_grid_feature, gird_features], dim=1)
        region_features = torch.cat([middle_cls_region_feature, region_features], dim=1)

        # print(region_features.size())

        add_mask = torch.zeros(b_s, 1, 1, 2).bool().to(region_features.device)
        attention_mask = torch.cat([add_mask, attention_mask], dim=-1)
        grid_att = self.self_grid(gird_features, gird_features, gird_features)
        region_att = self.self_region(region_features, region_features, region_features, attention_mask=attention_mask)


        gird_ff = self.pwff_grid(grid_att)
        region_ff = self.pwff_region(region_att)

        gird_ff = gird_ff[:,2:]
        region_ff = region_ff[:,2:]

        return gird_ff, region_ff

class TransformerEncoder(nn.Module):
    def __init__(self, N, device='cuda', d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.device = device

        self.grid_proj = nn.Sequential(
            nn.Linear(1024, self.d_model),
            # nn.Linear(768, self.d_model),
            # nn.Linear(4096, self.d_model),
            # nn.Linear(768, self.d_model),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.LayerNorm(self.d_model)
        )

        self.region_proj = nn.Sequential(
            nn.Linear(2048, self.d_model),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.LayerNorm(self.d_model)
        )
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N)])
        self.quer2att_region = nn.ModuleList([MultiHeadAttention(d_model, d_k, d_v, h, dropout) for _ in range(N-1)])
        self.quer2att_grid = nn.ModuleList([MultiHeadAttention(d_model, d_k, d_v, h, dropout) for _ in range(N-1)])

    def forward(self, grid_features, region_features):
        # input (b_s, seq_len)
        b_s = region_features.shape[0]
        attention_mask = (torch.sum(torch.abs(region_features), -1) == 0).unsqueeze(1).unsqueeze(1)
        grid_features = self.grid_proj(grid_features)
        region_features = self.region_proj(region_features)

        ####### 每一层的结果
        grid_features_lists = []
        region_features_lists = []
        for l in self.layers:
            grid_features, region_features = l(grid_features, region_features, attention_mask)
            grid_features_lists.append(grid_features)
            region_features_lists.append(region_features)
        ###### 进行合并
        next_output_region = region_features_lists[0]
        next_output_grid = grid_features_lists[0]
        region_res = [region_features_lists[0]]
        grid_res = [grid_features_lists[0]]

        for j,(l_r,l_g) in enumerate(zip(self.quer2att_region,self.quer2att_grid)):
            next_output_grid1,next_output_region1=next_output_grid,next_output_region
            next_output_region = l_r(next_output_grid1,region_features_lists[j+1],region_features_lists[j+1], attention_mask=attention_mask)
            next_output_grid = l_g(next_output_region1,grid_features_lists[j+1],grid_features_lists[j+1])
            if j==1:
                region_res.append(next_output_region)
                grid_res.append(next_output_grid)
            else:
                region_res.append(next_output_grid)
                grid_res.append(next_output_region)

        grid_features = torch.cat(grid_res,dim=1)
        region_features = torch.cat(region_res,dim=1)
        return grid_features, region_features, attention_mask



def build_encoder(N, device='cuda', d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
    Encoder = TransformerEncoder(N, device, d_model, d_k, d_v, h, d_ff, dropout)
    
    return Encoder