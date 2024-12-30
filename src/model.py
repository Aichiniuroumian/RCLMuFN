from typing import Optional

from torch import Tensor
from transformers import CLIPModel,BertConfig
from transformers.models.bert.modeling_bert import BertLayer
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy


from backbone import build_backbone

from transformers import BertTokenizer, BertModel


class MultimodalEncoder(nn.Module):
    def __init__(self, config, layer_number):
        super(MultimodalEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_number)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        all_encoder_attentions = []
        for layer_module in self.layer:

            hidden_states, attention = layer_module(hidden_states, attention_mask, output_attentions=True)
            all_encoder_attentions.append(attention)

            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_attentions


class CrossAttention(nn.Module):
    def __init__(self, feature_dim, dropout_prob=0.1):
        super(CrossAttention, self).__init__()
        self.text_linear = nn.Linear(feature_dim, feature_dim)
        self.extra_linear = nn.Linear(feature_dim, feature_dim)
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, query, key, value):
        if query.shape[-1] != 768:
            query = self.text_linear(query)
        if key.shape[-1] != 768:
            key = self.extra_linear(key)
            value = self.extra_linear(value)
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(
            torch.tensor(key.size(-1), dtype=torch.float32)
        )
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, value)
        attended_values = self.dropout(attended_values)
        return attended_values


class RCLMuFN(nn.Module):
    def __init__(self, args):
        super(RCLMuFN, self).__init__()
        self.model = CLIPModel.from_pretrained("./MMSD2.0-main/openai/clip-vit-base-patch32")
        self.config = BertConfig.from_pretrained("./MMSD2.0-main/bert-base-uncased")
        self.config.hidden_size = 768
        self.config.num_attention_heads = 8
        self.trans = MultimodalEncoder(self.config, layer_number=args.layers)
        if args.simple_linear:
            self.text_linear =  nn.Linear(args.text_size, args.image_size)
            self.image_linear =  nn.Linear(args.image_size, args.image_size)
        else:
            self.text_linear =  nn.Sequential(
                nn.Linear(args.text_size, args.image_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )
            self.image_linear =  nn.Sequential(
                nn.Linear(args.image_size, args.image_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )
        self.classifier_fuse = nn.Linear(args.image_size , args.label_number)
        self.cross_att = CrossAttention(feature_dim=768, dropout_prob=0.1)
        self.loss_fct = nn.CrossEntropyLoss()
        self.tokenizer = BertTokenizer.from_pretrained("./MMSD2.0-main/bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("./MMSD2.0-main/bert-base-uncased")
        self.backbone = build_backbone(args)
        self.d_model = 768
        self.nheads = 8
        self.dim_feedforward = 2048
        self.txt = nn.Sequential(nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU(),
                                 nn.LayerNorm(self.d_model)
                                 )
        self.txt2 = nn.Sequential(nn.Linear(self.d_model*2, self.d_model),
                                 nn.ReLU(),
                                 nn.Linear(self.d_model, self.d_model),
                                 nn.LayerNorm(self.d_model)
                                 )
        self.vis2 = nn.Sequential(nn.Linear(self.d_model * 2, self.d_model),
                                 nn.ReLU(),
                                  nn.Linear(self.d_model, self.d_model),
                                 nn.LayerNorm(self.d_model)
                                 )
        self.text_self = TransformerEncoderLayer(self.d_model, self.nheads, dim_feedforward=self.dim_feedforward)
        self.text_cross = TransformerCrossLayer(self.d_model, self.nheads, dim_feedforward=self.dim_feedforward)
        self.vis_self = TransformerEncoderLayer(self.d_model, self.nheads, dim_feedforward=self.dim_feedforward)
        self.vis_cross = TransformerCrossLayer(self.d_model, self.nheads, dim_feedforward=self.dim_feedforward)
        self.imtxt_cross = TransformerCrossLayer(768, self.nheads, dim_feedforward=self.dim_feedforward)
        self.attetion_block = nn.Sequential(nn.Linear(self.d_model * 2, self.d_model),
                                            nn.ReLU(),
                                            nn.Linear(self.d_model, self.d_model),
                                            nn.Sigmoid())
        self.mlp_layer = nn.Sequential(nn.Linear(self.d_model * 2, self.d_model),
                                       nn.ReLU(),
                                       nn.Linear(self.d_model, self.d_model),
                                       nn.LayerNorm(self.d_model))
        self.input_proj = nn.Conv2d(self.dim_feedforward, 768, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, inputs, batch, labels):
        output = self.model(**inputs,output_attentions=True)

        # 提取对应的特征
        text_features = output['text_model_output']['last_hidden_state']  # 128，77，512
        image_features = output['vision_model_output']['last_hidden_state']  # 128，50，768
        text_feature = output['text_model_output']['pooler_output'] # 64，512
        image_feature = output['vision_model_output']['pooler_output'] # 64，768
        text_feature = self.text_linear(text_feature)  # 64，768
        image_feature = self.image_linear(image_feature)  # 64,768

        text_list, image_list, label_list, id_list, samples = batch

        features, pos = self.backbone(samples.to(inputs['input_ids'].device) )
        src, mask = features[-1].decompose()  # 32,2048,7,7

        # resnet50
        src = self.input_proj(src)  # 64,768,7,7
        pooled_features = self.pool(src)  # [batch_size, 768, 1, 1]
        res_features = pooled_features.view(pooled_features.size(0), -1)  # [32, 768]
        # bert
        encoded_input = self.tokenizer(text_list, padding=True, truncation=True, return_tensors='pt')
        encoded_input = encoded_input.to(inputs['input_ids'].device)
        with torch.no_grad():
            outputs_bert = self.bert_model(**encoded_input)
            last_hidden_states = outputs_bert.last_hidden_state  # 64,56,768
            pooler_outputs = outputs_bert.pooler_output  # 64,768
        bert_text_features = self.txt(pooler_outputs)  # 64,768

        # SFIM
        image_t = self.imtxt_cross(tgt=res_features, memory=bert_text_features)  # 32,768
        text_im = self.imtxt_cross(tgt=bert_text_features, memory=res_features)  # 32,768

        # RCLM
        text_feature2 = self.txt2(torch.cat([text_feature, text_im], dim=-1))  # 32,768
        text_feature2 = text_feature2.unsqueeze(1)  # 32,1,768
        txt_cat = self.text_self(torch.stack([text_feature, text_im], dim=1))  # 32,2,768
        txt_output = self.text_cross(tgt=text_feature2, memory=txt_cat)  # 32,1,768
        txt_output = txt_output.squeeze(1)  # 32,768

        image_feature2 = self.vis2(torch.cat([image_feature, image_t],dim=-1))  # 32,768
        image_feature2 = image_feature2.unsqueeze(1)  # 32,1,768
        image_cat = self.vis_self(torch.stack([image_feature, image_t], dim=1))  # 32,2,768
        image_output = self.vis_cross(tgt=image_feature2, memory=image_cat)  # 32,1,768
        image_output = image_output.squeeze(1)  # 32,768

        txt_out = self.cross_att(txt_output, image_output, image_output)  # 32,768
        image_out = self.cross_att(image_output, txt_output, txt_output)  # 32,768
        res_bert = 0.6 * image_out + 0.4 * txt_out  # 32,768
        # res_bert = image_t + text_im

        # CLIP-View Feature Fusion
        cross_feature_text = self.cross_att(text_feature, image_feature, image_feature)  # 32,768
        cross_feature_image = self.cross_att(image_feature, text_feature, text_feature)  # 32,768
        fuse_feature = 0.7 * cross_feature_text + 0.3 * cross_feature_image

        # MuFFM
        att = self.attetion_block(torch.cat([fuse_feature, res_bert], dim=-1))  # 32,768
        output = 0.5 * fuse_feature + 0.5 * (att * self.mlp_layer(torch.cat([fuse_feature, res_bert], dim=-1)))  # 32,768

        # Predict
        logits_fuse = self.classifier_fuse(output)  # 64,2  output
        fuse_score = nn.functional.softmax(logits_fuse, dim=-1)  # 64,2

        score = fuse_score

        outputs = (score,) # (64,2)
        if labels is not None:
            loss_fuse = self.loss_fct(logits_fuse, labels)
            loss = loss_fuse
            outputs = (loss,) + outputs
        return outputs

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerCrossLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,return_attn=False,kdim=None,vdim=None):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,kdim=kdim,vdim=vdim,batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.return_attn = return_attn

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        tgt2, cross_attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
