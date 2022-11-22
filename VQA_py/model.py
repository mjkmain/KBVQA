import torch
import torch.nn as nn
from transformers import AutoModel
import timm
import warnings
warnings.filterwarnings("ignore")
from transformers import logging
logging.set_verbosity_error()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class VQAModel(nn.Module):
    def __init__(self, num_target, dim_i, dim_h=1024, config=None):
        super(VQAModel, self).__init__()
        self.config = config
        self.dim_i = dim_i
        self.bert = AutoModel.from_pretrained('klue/roberta-large')
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True) 

        self.vit.head = nn.Linear(768, dim_i) 
        self.out_size = self.vit.head.out_features+self.bert.pooler.dense.out_features # 768 + 1024
        self.i_drop = nn.Dropout(0.25)
        
        self.linear = nn.Linear(self.out_size, dim_h)
        self.h_layer_norm = nn.LayerNorm(dim_h)
        self.layer_norm = nn.LayerNorm(num_target)

        self.relu = nn.ReLU()
        self.out_linear = nn.Linear(dim_h, num_target)
        self.drop = nn.Dropout(0.2)
        self.tanh = nn.Tanh()
        
    def forward(self, idx, mask, image):
        q_f = self.bert(idx, mask) 
        q_f = q_f.pooler_output
        q_f = q_f
        i_f = self.i_drop(self.tanh(self.vit(image))) 
        
        uni_f = torch.cat([i_f, q_f], axis=1)

        if self.config.use_transformer_layer:
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.out_size , nhead=8, dropout=0.2).to(DEVICE)
            transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2).to(DEVICE)
            uni_f = transformer_encoder(uni_f)

        outputs = self.out_linear(self.relu(self.drop(self.h_layer_norm(self.linear(uni_f)))))

        return outputs