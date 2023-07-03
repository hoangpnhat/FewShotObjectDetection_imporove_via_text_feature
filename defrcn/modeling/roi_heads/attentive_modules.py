import numpy as np
import torch.nn.functional as F
from typing import Optional
import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor
from detectron2.layers import ShapeSpec, cat, nonzero_tuple

from detectron2.data import MetadataCatalog, DatasetCatalog
from torchnlp.word_to_vector import GloVe
from .my_module import *
from ..meta_arch.gdl import decouple_layer, AffineLayer
from defrcn.utils.class_embedding import get_class_embed, create_normalized_orthogonal_tensor
from defrcn.utils.class_name import get_class_name


class Sequential(nn.Sequential):
    def forward(self, *x):
        for module in self:
            if type(x) == tuple:
                x = module(*x)
            else:
                x = module(x)
        return x

class MLP(Sequential):
    def __init__(self, num_channels: int, widening_factor: int):
        super().__init__(
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, widening_factor * num_channels),
            nn.GELU(),
            nn.Linear(widening_factor * num_channels, num_channels),
        )

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class FFN(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        d_ffn = 1024
        self.d_model = d_model
        # ffn
        self.linear1 = nn.Linear(self.d_model, d_ffn)
        self.activation = F.relu
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, self.d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(self.d_model)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class SingleHeadSiameseAttention(nn.Module):
    """ Single-Head Attention Module. Weights for Q and K are shared in a Siamese manner. No proj weights for V."""

    def __init__(self, d_model, dropout=0):
        super().__init__()
        self.n_head = 1
        self.d_model = d_model
        self.w_qk = nn.Linear(self.d_model, self.n_head *
                              self.d_model, bias=False)

        self.attention = ScaledDotProductAttention(
            temperature=np.power(self.d_model, 0.5), dropout=dropout)
        nn.init.normal_(self.w_qk.weight, mean=0, std=np.sqrt(
            2.0 / (self.d_model + self.d_model)))

        self.dummy = nn.Parameter(torch.Tensor(1, self.d_model))
        nn.init.normal_(self.dummy)

        self.linear1 = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2), nn.ReLU(inplace=True))
        self.linear2 = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2), nn.ReLU(inplace=True))
        self.linear3 = nn.Linear(self.d_model * 2, self.d_model)
        self.ffn = FFN(self.d_model, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # print(q.size())
        # print(k.size())
        # print(v.size())
        
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qk(q).view(sz_b, len_q, self.n_head, self.d_model)
        k = self.w_qk(k).view(sz_b, len_k, self.n_head, self.d_model)
        v = v.view(sz_b, len_v, self.n_head, self.d_model)
        # tsp = tsp.view(sz_b, len_v, self.n_head, self.d_model)

        dummy = self.dummy.reshape(1, 1, 1, self.d_model).expand(
            sz_b, -1, self.n_head, -1)
        dummy_v = torch.zeros(sz_b, 1, self.n_head,
                              self.d_model, device=v.device)

        k = torch.cat([k, dummy], dim=1)
        v = torch.cat([v, dummy_v], dim=1)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q,
                                                    self.d_model)  # (n_head * b) x lq x d_model
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k + 1,
                                                    self.d_model)  # (n_head * b) x lk x d_model
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v + 1,
                                                    self.d_model)  # (n_head * b) x lv x d_model
        # tsp = tsp.permute(2, 0, 1, 3).contiguous(
        # ).view(-1, len_v + 1, self.d_model)  # (n_head * b) x lv x d_model
      
        use_cosine = False
        if use_cosine:
            def norm(x): return torch.nn.functional.normalize(x, p=2.0, dim=-1)
            def cosine_func(x, y): return torch.einsum(
                'b i c, b j c -> b i j', norm(x), norm(y))
            output = cosine_func(q, k)
            output = F.relu(output)
            output = self.dropout(output)
            output = torch.bmm(output, v)
            # print('cosine_func:', output.shape)
        else:
            output, attn, log_attn = self.attention(q, k, v)
            # print('softmax:', output.shape)

        # tsp, _, _ = self.attention(q, k, tsp)

        output = output.view(self.n_head, sz_b, len_q, self.d_model)
        output = output.permute(1, 2, 0, 3).contiguous().view(
            sz_b, len_q, -1)  # b x lq x (n_head * d_model)

        output1 = self.linear1(output * residual)
        # print('output', output.shape)
        # print('residual', residual.shape)
        # output3 = self.linear2(residual + output)
        output2 = self.linear2(residual - output)
        # output2 = self.linear2(residual - output)
        output = self.linear3(
            torch.cat([output1, output2, residual], dim=2)
        )
        output = self.ffn(output)
        # assert 0
        return output



def _init_parameters(module, init_scale):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=init_scale)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Embedding):
            m.weight.data.normal_(mean=0.0, std=init_scale)
            

class SematicProposalAttention(nn.Module):
    def __init__(self,
                    input_size,
                    cfg=None,
                    is_multi=False,
                    dropout=0):
            super().__init__()
            self.is_multi = is_multi
            self.device = 'cuda'
            self.dropout = dropout
            self.__init_language_model__(cfg)
            self.__init_attention_layer__(input_size)
           
    def __init_language_model__(self,cfg):
        self.addition_model = cfg.MODEL.ADDITION.NAME
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        
        
        if self.addition_model is not None:
            if self.addition_model == "glove":
                self.semantic_dim = 300
            elif self.addition_model == "clip":
                self.semantic_dim = 512
                
            self.fixed_bg = False
            self.class_names = get_class_name(cfg)
            self.class_embed = get_class_embed(self.class_names, self.addition_model, include_bg=self.fixed_bg).to(self.device)
            if not self.fixed_bg:
                # self.bg_feature_init = torch.randn(1, self.semantic_dim)
                # self.bg_feature = nn.parameter.Parameter(self.bg_feature_init.clone(), requires_grad=True)
                
                average_vector_foreground = torch.mean(self.class_embed, dim=0,keepdim=True)
                self.bg_feature = torch.neg(average_vector_foreground)
                # self.bg_feature = nn.parameter.Parameter(self.bg_feature_init.clone(), requires_grad=True)
                # self.bg_feature = create_normalized_orthogonal_tensor(average_vector_frontground)


        embed = torch.zeros(len(self.class_names), self.semantic_dim).to(self.device)

        for id, name in enumerate(self.class_names):
            embed[id] = embed[id] + \
                                    self.class_embed[id].to(self.device)
        self.embed = embed
    def __init_attention_layer__(self, input_size):
            init_scale = 0.02
            self.attention = SingleHeadSiameseAttention(input_size)
            self.query_projection = nn.Linear(input_size,self.semantic_dim)
            self.output_projection = nn.Linear(input_size,self.semantic_dim)
            
            
            self.key_projection = nn.Linear(self.semantic_dim, input_size)
            self.value_projection = nn.Linear(self.semantic_dim, input_size)
            
            with torch.no_grad():
                _init_parameters(self.attention, init_scale)
        
    def forward_language_model(self):
        output = {}
        loss ={}
        # print("embeding text feature",self.embed.shape)
        if not self.fixed_bg:
            text_feat = torch.cat([self.embed, self.bg_feature], dim=0)  # add bg
        output.update({
            # 'pred_weights': pred_weights,
            'text_feat': text_feat,
        })
        return loss,output
    
    def forward(self, visual_feat):
        loss, output = self.forward_language_model()

        text_feat = output['text_feat']
        residual = text_feat.detach().clone().to(self.device)
        
        value_feat = text_feat.detach().clone().to(self.device)
        # visual_feat = self.query_projection(visual_feat)
        # visual_feat =F.relu(visual_feat)
        
        text_feat = self.key_projection(text_feat)
        value_feat = self.value_projection(value_feat)
        text_feat = F.relu(text_feat)
        value_feat = F.relu(value_feat)

        sim2stext = self.attention(
            q=visual_feat[None, :], k=text_feat[None, :], v=value_feat[None, :])[0]
        

        sim2stext = F.relu(sim2stext)
        
        # output['sim2stext'] = (1-alpha)*sim2stext + \
        #     alpha*self.forward_wo_label(visual_feat)
        # sim2stext = self.value_projection(sim2stext)
        # sim2stext = F.relu(sim2stext)
        output['sim2stext'] = sim2stext
        output['text_feat'] = residual

        return loss,output

        
class LV_attention(nn.Module):
    def __init__(self,
                 input_size,
                 cfg=None,
                 is_multi=False,
                 output_size=0,
                 dropout=0):
        super().__init__()
        self.is_multi = is_multi
        self.device = 'cuda'
        self.output_size = output_size if output_size else input_size
        self.dropout = dropout
        # self.distill_mode = cfg.MODEL.ROI_HEADS.DISTILLATE
        # self.student_training = cfg.MODEL.ROI_HEADS.STUDENT_TRAINING
        # self.teacher_training = cfg.MODEL.ROI_HEADS.TEACHER_TRAINING
        self.__init_language_model__(cfg)
        # self.__init_attention_layer__(input_size, num_super_cls)
        self.__init_attention_layer__(input_size)
        
        # if self.student_training:
        #     # self.mlp_adapter = MLP(input_size, widening_factor=2)
        #     # self.mlp_adapter = Adaptor(input_size, cfg=cfg, is_multi=False)

        #     self.mlp_adapter = torch.nn.Sequential(
        #         nn.Linear(input_size, input_size, bias=True),
        #         nn.ReLU(),
        #         nn.Linear(input_size, input_size, bias=True),
        #         nn.ReLU(),
        #     )
        #     self.mlp_adapter2 = torch.nn.Sequential(
        #         nn.Linear(input_size, input_size, bias=True),
        #         nn.ReLU(),
        #         nn.Linear(input_size, input_size, bias=True),
        #         nn.ReLU(),
        #     )
    def __init_language_model__(self, cfg):
        self.text_dim = 300
        self.l_model = GloVe(name='6B', dim=self.text_dim)
        # self.l_model = GloVe(name='42B', dim=text_dim)

        dataset_name = cfg.DATASETS.TRAIN[0]

        metadata_dict = MetadataCatalog.get(dataset_name)
        is_novel = True if 'shot' in dataset_name else False

        if is_novel:
            if 'all' in dataset_name:
                self.classes = metadata_dict.thing_classes
            else:
                self.classes = metadata_dict.novel_classes
                metadata_dict.novel_dataset_id_to_contiguous_id
        else:
            self.classes = metadata_dict.base_classes

        # metadata_dict.novel_classes
        map_voc = {'aeroplane': 'aeroplane', 'bicycle': 'bicycle', 'boat': 'boat', 'bottle': 'bottle', 'car': 'car', 'cat': 'cat', 'chair': 'chair', 'diningtable': 'dining table', 'dog': 'dog', 'horse': 'horse',
                   'person': 'person', 'pottedplant': 'potted plant', 'sheep': 'sheep', 'train': 'train', 'tvmonitor': 'tv', 'bird': 'bird', 'bus': 'bus', 'cow': 'cow', 'motorbike': 'motorbike', 'sofa': 'sofa'}

        # print(base_classes)
        embed = torch.zeros(len(self.classes), self.text_dim).to(self.device)

        for id, name in enumerate(self.classes):
            text = map_voc[name]
            for i in text.split(' '):
                embed[id] = embed[id] + \
                    self.l_model[i][None, :].to(self.device)
        self.class_id = torch.arange(len(self.classes)+1)
        self.embed = embed
        self.w_bg_init = torch.randn(1, self.text_dim)
        # self.w_bg_init = torch.zeros(1, text_dim)
        self.w_bg = torch.nn.parameter.Parameter(
            self.w_bg_init.clone(), requires_grad=True)
        return
    def __init_attention_layer__(self, input_size):
        init_scale = 0.02
        self.attention = SingleHeadSiameseAttention(input_size)
        self.proj_k = nn.Linear(input_size*2, input_size)
        self.proj2 = nn.Linear(self.text_dim, input_size)
        with torch.no_grad():
            _init_parameters(self.attention, init_scale)
            
            
    
    def forward_language_model(self, label):
        loss = {}
        output = {}
        # print("embeding text feature",self.embed.shape)

        embed = torch.cat([self.embed, self.w_bg], dim=0)  # add bg
        # print("embeding text feature add background",embed.shape)
        embed = self.proj2(embed)
        # print("embeding text feature after project",embed.shape)
        
        # print('good_vector')
        good_vector = F.one_hot(label, len(
            self.classes)+1).to(torch.float).to(self.device)
        # print('good_vector',good_vector.shape)

        text_feat = torch.einsum(
            'b i, i j ->b j', good_vector, embed)
        output.update({
            # 'pred_weights': pred_weights,
            'text_feat': text_feat,
        })
        return loss, output
    
    def forward(self, visual_feat, text, num_preds_per_image=None):
        # print(visual_feat.shape)

        loss, output = self.forward_language_model(text)
        # sim_feat, gim_feat = self.forward_vision_model(
        # visual_feat, text)
        # visual_feat[None, :],
        # sim2stext = self.attention1(visual_feat[None, :], stext_feat)[0]
        
        text_feat = output['text_feat']
        # print('text_feat',text_feat.shape)

        value_feat = torch.cat([visual_feat, text_feat], dim=-1)
        value_feat = self.proj_k(value_feat)

        text_feat = F.relu(text_feat)
        value_feat = F.relu(value_feat)

        sim2stext = self.attention(
            q=visual_feat[None, :], k=text_feat[None, :], v=value_feat[None, :])[0]

        sim2stext = F.relu(sim2stext)

        
        alpha = torch.rand(1).cuda()
        alpha = 0
        # output['sim2stext'] = (1-alpha)*sim2stext + \
        #     alpha*self.forward_wo_label(visual_feat)

        output['sim2stext'] = sim2stext

        return loss, output

 
class LV_attention_VKV(LV_attention):
    def __init__(self,
                    input_size,
                    cfg=None,
                    is_multi=False,
                    output_size=0,
                    dropout=0):
        super().__init__( input_size,
                    cfg,
                    is_multi,
                    output_size,
                    dropout)
    def forward(self, visual_feat, text, num_preds_per_image=None):
        x = visual_feat

        loss, output = self.forward_language_model(
            visual_feat, text)
        # sim_feat, gim_feat = self.forward_vision_model(
        # visual_feat, text)
        # visual_feat[None, :],
        # sim2stext = self.attention1(visual_feat[None, :], stext_feat)[0]

        text_feat = output['text_feat']


        value_feat = torch.cat([visual_feat[None, :], text_feat], dim=2)
        value_feat = self.proj_k(value_feat)

        text_feat = F.relu(text_feat)
        value_feat = F.relu(value_feat)

        sim2stext = self.attention(
            q=value_feat, k=text_feat, v=value_feat)[0]

        # sim2stext = self.attention(
        #     q=visual_feat[None, :], k=text_feat, v=value_feat)[0]

        # torch.cat([sim2stext], dim=1)
        sim2stext = F.relu(sim2stext)
        # print('sim2stext.shape', sim2stext.shape)
        alpha = torch.rand(1).cuda()
        alpha = 0
        # output['sim2stext'] = (1-alpha)*sim2stext + \
        #     alpha*self.forward_wo_label(visual_feat)

        output['sim2stext'] = sim2stext

        return loss, output

          
class LV_attention_textDomination(nn.Module):
    def __init__(self,
                 input_size,
                 cfg=None,
                 is_multi=False,
                 output_size=0,
                 dropout=0):
        super().__init__()
        self.is_multi = is_multi
        self.device = 'cuda'
        self.output_size = output_size if output_size else input_size
        self.dropout = dropout
        self.distill_mode = cfg.MODEL.ROI_HEADS.DISTILLATE
        self.student_training = cfg.MODEL.ROI_HEADS.STUDENT_TRAINING
        self.teacher_training = cfg.MODEL.ROI_HEADS.TEACHER_TRAINING
        self.__init_language_model__(cfg)
        # self.__init_attention_layer__(input_size, num_super_cls)
        self.__init_attention_layer__(input_size)
        
        if self.student_training:
            # self.mlp_adapter = MLP(input_size, widening_factor=2)
            # self.mlp_adapter = Adaptor(input_size, cfg=cfg, is_multi=False)

            self.mlp_adapter = torch.nn.Sequential(
                nn.Linear(input_size, input_size, bias=True),
                nn.ReLU(),
                nn.Linear(input_size, input_size, bias=True),
                nn.ReLU(),
            )
            self.mlp_adapter2 = torch.nn.Sequential(
                nn.Linear(input_size, input_size, bias=True),
                nn.ReLU(),
                nn.Linear(input_size, input_size, bias=True),
                nn.ReLU(),
            )
    def __init_language_model__(self, cfg, num_clusters=6):
        self.text_dim = 300
        self.l_model = GloVe(name='6B', dim=self.text_dim)
        # self.l_model = GloVe(name='42B', dim=text_dim)

        dataset_name = cfg.DATASETS.TRAIN[0]

        metadata_dict = MetadataCatalog.get(dataset_name)
        is_novel = True if 'shot' in dataset_name else False

        if is_novel:
            if 'all' in dataset_name:
                self.classes = metadata_dict.thing_classes
            else:
                self.classes = metadata_dict.novel_classes
                metadata_dict.novel_dataset_id_to_contiguous_id
        else:
            self.classes = metadata_dict.base_classes

        # metadata_dict.novel_classes
        map_voc = {'aeroplane': 'aeroplane', 'bicycle': 'bicycle', 'boat': 'boat', 'bottle': 'bottle', 'car': 'car', 'cat': 'cat', 'chair': 'chair', 'diningtable': 'dining table', 'dog': 'dog', 'horse': 'horse',
                   'person': 'person', 'pottedplant': 'potted plant', 'sheep': 'sheep', 'train': 'train', 'tvmonitor': 'tv', 'bird': 'bird', 'bus': 'bus', 'cow': 'cow', 'motorbike': 'motorbike', 'sofa': 'sofa'}

        # print(base_classes)
        embed = torch.zeros(len(self.classes), self.text_dim).to(self.device)

        for id, name in enumerate(self.classes):
            text = map_voc[name]
            for i in text.split(' '):
                embed[id] = embed[id] + \
                    self.l_model[i][None, :].to(self.device)
        self.class_id = torch.arange(len(self.classes)+1)
        self.embed = embed
        self.w_bg_init = torch.randn(1, self.text_dim)
        # self.w_bg_init = torch.zeros(1, text_dim)
        self.w_bg = torch.nn.parameter.Parameter(
            self.w_bg_init.clone(), requires_grad=True)
        return
    def __init_attention_layer__(self, input_size):
        init_scale = 0.02
        self.attention = SingleHeadSiameseAttention(self.text_dim)
        self.proj_k = nn.Linear(input_size*2, input_size)
        self.proj2 = nn.Linear(self.text_dim, input_size)
        self.proj_visual = nn.Linear(input_size,self.text_dim)
        self.proj_value = nn.Linear(self.text_dim*2, self.text_dim)
        
        with torch.no_grad():
            _init_parameters(self.attention, init_scale)
            
            
    
    def forward_language_model(self, visual_feat, label):
        loss = {}
        output = {}

        embed = torch.cat([self.embed, self.w_bg], dim=0)  # add bg
        # embed = self.proj2(embed)
        # print('good_vector')
        good_vector = F.one_hot(label, len(
            self.classes)+1).to(torch.float).to(self.device)
        # print('good_vector')
        

        text_feat = torch.einsum(
            'b i, i j ->b j', good_vector, embed)

        # pred_weights = self.predict_w(visual_feat)
        output.update({
            # 'pred_weights': pred_weights,
            'text_feat': text_feat[None, :],
        })
        return loss, output
    def forward(self, visual_feat, text, num_preds_per_image=None):
        visual_feat = self.proj_visual(visual_feat)

        loss, output = self.forward_language_model(
            visual_feat, text)
        # sim_feat, gim_feat = self.forward_vision_model(
        # visual_feat, text)
        # visual_feat[None, :],
        # sim2stext = self.attention1(visual_feat[None, :], stext_feat)[0]

        text_feat = output['text_feat']

        value_feat = torch.cat([visual_feat[None, :], text_feat], dim=2)
        # print("value_feat before project: ",value_feat.shape)
        
        value_feat = self.proj_value(value_feat)
        # print("value_feat after project: ",value_feat.shape)
        
        text_feat = F.relu(text_feat)
        value_feat = F.relu(value_feat)
        

        sim2stext = self.attention(
            q=visual_feat[None, :], k=text_feat, v=value_feat)[0]

        # torch.cat([sim2stext], dim=1)
        sim2stext = F.relu(sim2stext)
        sim2stext = self.proj2(sim2stext)
        
        
        alpha = torch.rand(1).cuda()
        alpha = 0
        # output['sim2stext'] = (1-alpha)*sim2stext + \
        #     alpha*self.forward_wo_label(visual_feat)

        output['sim2stext'] = sim2stext

        return loss, output


class LV_attention_textDomination_VKV(LV_attention_textDomination):
    def __init__(self,
                    input_size,
                    cfg=None,
                    is_multi=False,
                    output_size=0,
                    dropout=0):
        super().__init__( input_size,
                    cfg,
                    is_multi,
                    output_size,
                    dropout)
    
    def forward(self, visual_feat, text, num_preds_per_image=None):
        visual_feat = self.proj_visual(visual_feat)

        loss, output = self.forward_language_model(
            visual_feat, text)
        # sim_feat, gim_feat = self.forward_vision_model(
        # visual_feat, text)
        # visual_feat[None, :],
        # sim2stext = self.attention1(visual_feat[None, :], stext_feat)[0]

        text_feat = output['text_feat']

        value_feat = torch.cat([visual_feat[None, :], text_feat], dim=2)
        # print("value_feat before project: ",value_feat.shape)
        
        value_feat = self.proj_value(value_feat)
        # print("value_feat after project: ",value_feat.shape)
        
        text_feat = F.relu(text_feat)
        value_feat = F.relu(value_feat)
        

        sim2stext = self.attention(
            q=value_feat, k=text_feat, v=value_feat)[0]

        # torch.cat([sim2stext], dim=1)
        sim2stext = F.relu(sim2stext)
        sim2stext = self.proj2(sim2stext)
        
        
        alpha = torch.rand(1).cuda()
        alpha = 0
        # output['sim2stext'] = (1-alpha)*sim2stext + \
        #     alpha*self.forward_wo_label(visual_feat)

        output['sim2stext'] = sim2stext

        return loss, output
