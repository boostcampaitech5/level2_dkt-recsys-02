import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel
import pdb
import math
import numpy as np
import copy
import pickle
import torch.nn.init as init
from .layer import SASRecBlock, PositionalEncoding, Feed_Forward_block
import re
import json

class GraphEmbedding:
    def __init__(self, args, model):
        self.args = args
        curr_dir = __file__[:__file__.rfind('/')+1]
        with open(curr_dir + f'../../lightgcn/models_param/{model}_item_emb_{args.graph_dim}.pkl', 'rb') as f: 
            item_emb_dic = pickle.load(f)
            
            item_emb = []
            
            # Apply Xavier uniform initialization
            
            item_label = list(item_emb_dic.keys())
            item_label.sort()
            for i, label in enumerate(item_label):
                if i == 0:
                    self.emb_dim = len(item_emb_dic[label])
                    item_emb.append(item_emb_dic[label])
                item_emb.append(item_emb_dic[label])
            item_emb = torch.tensor(item_emb).to(self.args.device)
            self.item_emb = nn.Embedding.from_pretrained(item_emb)

    def item_emb(self, item_seq): return self.item_emb(item_seq)
    def user_emb(self, user_seq): return self.user_emb(user_seq)
    

class ModelBase(nn.Module):
    def __init__(
        self,
        args,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        **kwargs
    ):

        super().__init__()
        self.args = args
        self.use_res = self.args.use_res
        self.max_seq_len = self.args.max_seq_len
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_tests = n_tests
        self.n_questions = n_questions
        self.n_tags = n_tags
        self.resize_factor = self.args.resize_factor

        curr_dir = __file__[:__file__.rfind('/')+1]
        with open(curr_dir + f'../models_param/num_feature.json', 'r') as f: self.num_feature =  json.load(f)

########Graph Embedding
        self.use_graph = self.args.use_graph
        if self.use_graph:
            self.graph_emb = GraphEmbedding(args, self.args.graph_model)
            self.graph_emb_dim = self.graph_emb.emb_dim

######### Embeddings
        # hd: Hidden dimension, intd: Intermediate hidden dimension
        hd, intd = hidden_dim, hidden_dim // self.resize_factor

        self.embedding_interaction = nn.Embedding(3, intd) # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_test = nn.Embedding(n_tests + 1, intd)
        self.embedding_question = nn.Embedding(n_questions + 1, intd)
        self.embedding_tag = nn.Embedding(n_tags + 1, intd)
        self.embedding_question_N = nn.Embedding(self.num_feature['question_N'] + 1, intd)
        #self.embedding_New Feature = nn.Embedding(n_New Feature + 1, intd)

        self.interaction_dic = {}
        for i in range(n_questions+1):
            self.interaction_dic[i] = nn.Embedding(3, intd)
        

######## FE시 추가해야함
        self.embedding_dict = {}
        self.embedding_dict['testId'] = self.embedding_test
        self.embedding_dict['assessmentItemID'] = self.embedding_question
        self.embedding_dict['KnowledgeTag'] = self.embedding_tag
        self.embedding_dict['interaction'] = self.embedding_interaction
        self.embedding_dict['question_N'] = self.embedding_question_N
        #self.embedding_dict['New Feature'] = self.New Feature Embedding

########Concatentaed Embedding Projection, Feature 개수 바뀌면 바꿔야함 4, 5, 6
        if self.use_graph:
            self.comb_proj = nn.Sequential(
                nn.Linear(intd * len(self.embedding_dict) + self.graph_emb_dim , hd),
                nn.LayerNorm(hd, eps=1e-6)
            )     
        else:
            self.comb_proj = nn.Sequential(
                nn.Linear(intd * len(self.embedding_dict) , hd),
                nn.LayerNorm(hd, eps=1e-6)
            )

        
        #self.cont_proj = nn.Sequential(
        #    nn.Linear(n_cont , hd),
        #    nn.LayerNorm(hd, eps=1e-6)
        #)

######### Fully connected layer
        if (self.use_res == True) & (self.use_graph == True):
            self.fc = nn.Linear(hd + self.graph_emb_dim, 1)
        else:
            self.fc = nn.Linear(hd, 1)

    def get_interaction(self, item_seqs, interactions):
        emb_list = []
        for item_seq, interaction in zip(item_seqs, interactions):
            tmp = []
            for id, it in zip(item_seq, interaction):
                self.interaction_dic[int(id)](torch.tensor(it).cpu()).unsqueeze(dim = 0).unsqueeze(dim = 0) 
                tmp.append(self.interaction_dic[int(id)](torch.tensor(it).cpu()).unsqueeze(dim = 0).unsqueeze(dim = 0))
            emb_list.append(torch.cat(tmp, dim = 1))
    
        return torch.cat(emb_list, dim = 0).cuda()


    def get_graph_emb_dim(self): return self.graph_emb_dim

    def dic_embed(self, input_dic):
        
        input_dic = input_dic['category']
        embed_list = []
        for feature, feature_seq in input_dic.items():
            if feature not in ('answerCode','mask'):
                batch_size = feature_seq.size(0)
                embed_list.append(self.embedding_dict[feature](feature_seq.long()))

        if (self.use_graph== True) & ('assessmentItemID' in input_dic): 
            embed_list.append(self.graph_emb.item_emb(input_dic['category']['assessmentItemID'].long()))

        embed = torch.cat(embed_list, dim = 2)
        return embed, batch_size
    
    def get_graph_emb(self, seq):
        return self.graph_emb.item_emb(seq.long())
    
    def pad(self, seq):
        seq_len = seq.size(1)
        tmp = torch.zeros((seq.size(0), self.max_seq_len), dtype=torch.int16)
        tmp[:, self.max_seq_len-seq_len:] = seq
        tmp = tmp.to(self.args.device)
        return tmp.long()
        

    def dic_forward(self, input_dic):

#######Category
        input_cat = input_dic['category']
        embed_list = []
        for feature, feature_seq in input_cat.items():
            batch_size = feature_seq.size(0)
            
            if feature not in ('answerCode','mask', 'interaction'):
                embed_list.append(self.embedding_dict[feature](feature_seq.long()))
            if feature == 'interaction':
                embed_list.append(self.get_interaction(input_cat['assessmentItemID'].long(), input_cat['interaction'].long()))
        if self.use_graph: 
            embed_list.append(self.graph_emb.item_emb(input_cat['assessmentItemID'].long()))

        
        
        embed = torch.cat(embed_list, dim = 2)
        X = self.comb_proj(embed)
#######Continous
        #input_cont = input_dic['continous']

        return X, batch_size

    def short_forward(self, input_dic, length):

#######Category
        input_cat = input_dic['category']
        embed_list = []
        for feature, feature_seq in input_cat.items():
            batch_size = feature_seq.size(0)
            
            if feature not in ('answerCode','mask'):
                short_feature_seq = feature_seq[:, -length:].long()
                short_feature_seq = self.pad(short_feature_seq)
                embed_list.append(self.embedding_dict[feature](short_feature_seq))

        if self.use_graph: 
            short_feature_seq = input_cat['assessmentItemID'][:, -length:].long()
            short_feature_seq = self.pad(short_feature_seq)
            embed_list.append(self.graph_emb.item_emb(short_feature_seq))

        embed = torch.cat(embed_list, dim = 2)
        X = self.comb_proj(embed)

#######Continous

        return X, batch_size


class LSTM(ModelBase):
    def __init__(
        self,
        args,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        **kwargs
    ):
        super().__init__(
            args,
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags
        )
        self.args = args

        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )
    ######################## FE시 추가해야함
    def forward(self, input_dic):
        #X, batch_size = super().forward(testId=testId,
        #                                assessmentItemID=assessmentItemID,
        #                                KnowledgeTag=KnowledgeTag,
        #                                answerCode=answerCode,
        #                                mask=mask,
        #                                interaction=interaction)
        X, batch_size = super().dic_forward(input_dic)
        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out


class LSTMATTN(ModelBase):
    def __init__(
        self,
        args,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_heads: int = 2,
        drop_out: float = 0.1,
        **kwargs
    ):
        super().__init__(
            args,
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags
        )
        self.n_heads = n_heads
        self.drop_out = drop_out
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)
    ######################## FE시 추가해야함
    def forward(self, input_dic):
        #X, batch_size = super().forward(testId=testId,
        #                                assessmentItemID=assessmentItemID,
        #                               KnowledgeTag=KnowledgeTag,
        #                              answerCode=answerCode,
        #                               mask=mask,
        #                              interaction=interaction)

        X, batch_size = super().dic_forward(input_dic)
        mask = input_dic['category']['mask']

        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
   
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        if self.args.use_res:
            out = torch.cat([sequence_output, out], dim = 2)
            out = out.contiguous().view(batch_size, -1, self.hidden_dim * 2)
        else:
            out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out).view(batch_size, -1)
        return out


class BERT(ModelBase):
    def __init__(
        self,
        args,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_heads: int = 2,
        drop_out: float = 0.1,
        max_seq_len: float = 20,
        **kwargs
    ):
        super().__init__(
            args,
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags
        )
        self.n_heads = n_heads
        self.drop_out = drop_out
        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.n_layers,
            num_attention_heads=self.n_heads,
            max_position_embeddings=max_seq_len,
        )
        self.encoder = BertModel(self.config)
    ######################## FE시 추가해야함
    def forward(self, input_dic):
        #X, batch_size = super().forward(testId=testId,
        #                                assessmentItemID=assessmentItemID,
        #                                KnowledgeTag=KnowledgeTag,
        #                                answerCode=answerCode,
        #                                mask=mask,
        #                                interaction=interaction)

        X, batch_size = super().dic_forward(input_dic)
        mask = input_dic['category']['mask']

        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]

        if self.args.use_res:
            graph_out = super().get_graph_emb(input_dic['category']['assessmentItemID'])
            out = torch.cat([out, graph_out], dim = 2)
        
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out
    
    
    def forward(self, input_dic):
        """
        "now_testId"
        "now_assessmentItemID"
        "now_KnowledgeTag"
        "now_answerCode"
        "now_New Feature"

        "past_testId"
        "past_assessmentItemID"
        "past_KnowledgeTag",
        "past_answerCode"
        "past_New Feature"
        
        """
        past_answerCode = input_dic['past_answerCode']
        # masking 
        mask_pad = torch.BoolTensor(past_answerCode.cpu() > 0).unsqueeze(1).unsqueeze(1) # (batch_size, 1, 1, max_len)
        mask_time = (1 - torch.triu(torch.ones((1, 1, past_answerCode.size(1), past_answerCode.size(1))), diagonal=1)).bool() # (batch_size, 1, max_len, max_len)
        mask = (mask_pad & mask_time).to(self.device) # (batch_size, 1, max_len, max_len)

        # past
        past_cat_emb_list = []
        for idx, cat_col in enumerate(self.past_cat_cols):
            past_cat_emb_list.append(self.past_emb_dict[cat_col](input_dic[cat_col]))
        if self.use_graph: 
            past_cat_emb_list.append(self.graph_emb.item_emb(input_dic['past_assessmentItemID'].long()))

        past_cat_emb = torch.concat(past_cat_emb_list, dim = 2)
        past_cat_emb = self.past_cat_emb(past_cat_emb)
        #past_num_emb = self.past_num_emb(past_num_feature)

        #past_emb = torch.concat([past_cat_emb, past_num_emb], dim = 2)
        past_emb = past_cat_emb, 
        past_emb += self.past_answerCode_emb(past_answerCode.to(self.device))
        past_emb = self.emb_layernorm(past_emb) # LayerNorm

        for block in self.past_blocks:
            past_emb, attn_dist = block(past_emb, mask)

        past_emb, _ = self.past_lstm(past_emb)

        # now
        now_cat_emb_list = []
        for idx, cat_col in enumerate(self.cat_cols):
            now_cat_emb_list.append(self.now_emb_dict[cat_col](self.now_cat_feature[:, :, idx]))
        if self.use_graph: 
            now_cat_emb_list.append(self.graph_emb.item_emb(input_dic['now_assessmentItemID'].long()))

        now_cat_emb = torch.concat(now_cat_emb_list, dim = -1)
        now_cat_emb = self.now_cat_emb(now_cat_emb)

        #now_num_emb = self.now_num_emb(now_num_feature)

        #now_emb = torch.concat([now_cat_emb, now_num_emb], dim = -1)
        now_emb = now_cat_emb
        for block in self.now_blocks:
            now_emb, attn_dist = block(now_emb, mask)

        now_emb, _ = self.now_lstm(now_emb)
        
        # predict
        emb = torch.concat([self.dropout(past_emb), self.dropout(now_emb)], dim = -1)
        output = self.predict_layer(emb)

        return output.squeeze(2)



class Saint(ModelBase):
    def __init__(
        self,
        args,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_heads: int = 2,
        max_seq_len : int = 20,
        **kwargs
    ):
        super().__init__(
            args,
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags
        )
        self.args = args
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.device = self.args.device
        self.dropout = self.args.drop_out

        # decoder combination projection
        if self.args.use_graph:
            self.enc_comb_proj = nn.Sequential(
                nn.Linear((self.hidden_dim//3)*3 + super().get_graph_emb_dim(), self.hidden_dim),
                nn.LayerNorm(self.hidden_dim, eps=1e-6),
                )
            self.dec_comb_proj = nn.Sequential(
            nn.Linear((self.hidden_dim//3)*4 + super().get_graph_emb_dim(), self.hidden_dim),
            nn.LayerNorm(self.hidden_dim, eps=1e-6),
                )
        else:
            self.enc_comb_proj = nn.Sequential(
                nn.Linear((self.hidden_dim//3)*3, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim, eps=1e-6),
                )
            self.dec_comb_proj = nn.Sequential(
            nn.Linear((self.hidden_dim//3)*4, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim, eps=1e-6),
                )
        

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout, self.max_seq_len)
        self.pos_decoder = PositionalEncoding(self.hidden_dim, self.dropout, self.max_seq_len)

        # DECODER embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.transformer = nn.Transformer(
            d_model=self.hidden_dim, 
            nhead=self.n_heads,
            num_encoder_layers=self.n_layers, 
            num_decoder_layers=self.n_layers, 
            dim_feedforward=self.hidden_dim, 
            dropout=self.dropout, 
            activation='relu')

        self.fc = nn.Linear(self.hidden_dim, 1)
        self.activation = nn.Sigmoid()

        self.enc_mask = None
        self.dec_mask = None
        self.enc_dec_mask = None
    
    def get_mask(self, seq_len):
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1)).to(torch.float)

        return mask.masked_fill(mask==1, float('-inf'))
    
    
    def forward(self, input_dic):
   
        enc_input = {key: input_dic[key] for key in ['testId', 'assessmentItemID', 'KnowledgeTag']}
        embed_enc, batch_size = super().dic_forward(enc_input)
        embed_dec, batch_size = super().dic_forward(input_dic)
        
        seq_len = input_dic['category']['interaction'].size(1)
        
        # ATTENTION MASK 생성
        # encoder하고 decoder의 mask는 가로 세로 길이가 모두 동일하여
        # 사실 이렇게 3개로 나눌 필요가 없다
        if self.enc_mask is None or self.enc_mask.size(0) != seq_len:
            self.enc_mask = self.get_mask(seq_len).to(self.device)

        if self.dec_mask is None or self.dec_mask.size(0) != seq_len:
            self.dec_mask = self.get_mask(seq_len).to(self.device)

        if self.enc_dec_mask is None or self.enc_dec_mask.size(0) != seq_len:
            self.enc_dec_mask = self.get_mask(seq_len).to(self.device)

  
        embed_enc = embed_enc.permute(1, 0, 2)
        embed_dec = embed_dec.permute(1, 0, 2)
    
        # Positional encoding
        embed_enc = self.pos_encoder(embed_enc)
        embed_dec = self.pos_decoder(embed_dec)
        
        out = self.transformer(embed_enc, embed_dec,
                               src_mask=self.enc_mask,
                               tgt_mask=self.dec_mask,
                               memory_mask=self.enc_dec_mask)

        out = out.permute(1, 0, 2)

        if self.args.use_res:
            graph_out = super().get_graph_emb(input_dic['category']['assessmentItemID'])
            out = torch.cat([out, graph_out], dim = 2)
            out = out.contiguous().view(batch_size, -1, self.hidden_dim + self.graph_emb_dim)
        else:
            out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds
    



class LastQuery(ModelBase):
    def __init__(
        self,
        args,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_heads: int = 2,
        drop_out: float = 0.1,
        max_seq_len: float = 20,
        **kwargs
    ):
        super().__init__(
            args,
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags
        )
        self.args = args
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.drop_out = drop_out
        self.device = self.args.device
        
        self.embedding_position = nn.Embedding(self.max_seq_len, self.hidden_dim)
        
        self.query = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.key = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        
        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.n_heads)

        self.ffn = Feed_Forward_block(self.hidden_dim)
        
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True)

        self.activation = nn.Sigmoid()

    def get_pos(self, seq_len):
        # use sine positional embeddinds
        return torch.arange(seq_len).unsqueeze(0)

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self,input_dic):
        X, batch_size = super().dic_forward(input_dic)
        seq_len = input_dic['category']['interaction'].size(1)

        # position = self.get_pos(self.max_seq_len).to(self.device)
        # embed_pos = self.embedding_position(position)
        # X = X + embed_pos

        q = self.query(X).permute(1, 0, 2)
        q = self.query(X)[:, -1:, :].permute(1, 0, 2)
        k = self.key(X).permute(1, 0, 2)
        v = self.value(X).permute(1, 0, 2)

        out, _ = self.attn(q, k, v)

        out = out.permute(1, 0, 2)
        out = X + out
        out = self.ln1(out)

        out = self.ffn(out)

        out = X + out
        out = self.ln2(out)
        
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(out, hidden)
        
        if self.args.use_res:
            graph_out = super().get_graph_emb(input_dic['category']['assessmentItemID'])
            out = torch.cat([graph_out, out], dim = 2)
            out = out.contiguous().view(batch_size, -1, self.hidden_dim + self.graph_emb_dim)
        else:
            out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        
        out = self.activation(out).view(batch_size, -1)

        return out



class TransLSTM_G(ModelBase):
    def __init__(
        self,
        args,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_heads: int = 2,
        drop_out: float = 0.1,
        max_seq_len: float = 20,
        **kwargs
    ):
        super().__init__(
            args,
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags
        )
        self.n_heads = self.args.n_heads
        self.dropout = self.args.drop_out
        self.max_seq_len = self.args.max_seq_len
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        
        self.use_res = self.args.use_res
        # Bert config
        self.graph_dim = super().get_graph_emb_dim()
        self.device = self.args.device

        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.n_layers,
            num_attention_heads=self.n_heads,
            max_position_embeddings=self.max_seq_len,
        )
        self.encoder = BertModel(self.config)
        self.transformer = nn.Transformer(
            d_model=self.hidden_dim, 
            nhead=self.n_heads,
            num_encoder_layers=self.n_layers, 
            num_decoder_layers=self.n_layers, 
            dim_feedforward=self.hidden_dim, 
            dropout=self.dropout, 
            activation='relu')
        
        self.pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout, self.max_seq_len)
        self.pos_decoder = PositionalEncoding(self.hidden_dim, self.dropout, self.max_seq_len)

        self.enc_mask = None
        self.dec_mask = None
        self.enc_dec_mask = None

        if self.use_res:
            self.fc = nn.Linear(self.hidden_dim + self.hidden_dim, 1)
        else:
            self.fc = nn.Linear(self.hidden_dim, 1)

    def get_mask(self, seq_len):
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1)).to(torch.float)
        return mask
    ######################## FE시 추가해야함
    def forward(self, input_dic):
        #X, batch_size = super().forward(testId=testId,
        #                                assessmentItemID=assessmentItemID,
        #                                KnowledgeTag=KnowledgeTag,
        #                                answerCode=answerCode,
        #                                mask=mask,
        #                                interaction=interaction)
        mask = input_dic['category']['mask']

        #enc_input = {key: input_dic[key] for key in ['testId', 'assessmentItemID', 'KnowledgeTag']}
        embed_enc, batch_size = super().dic_forward(input_dic)
        embed_dec, batch_size = super().dic_forward(input_dic)
        
        seq_len = input_dic['category']['interaction'].size(1)
        

        if self.enc_mask is None or self.enc_mask.size(0) != seq_len:
            self.enc_mask = self.get_mask(seq_len).to(self.device)

        if self.dec_mask is None or self.dec_mask.size(0) != seq_len:
            self.dec_mask = self.get_mask(seq_len).to(self.device)

        if self.enc_dec_mask is None or self.enc_dec_mask.size(0) != seq_len:
            self.enc_dec_mask = self.get_mask(seq_len).to(self.device)

  
        embed_enc = embed_enc.permute(1, 0, 2)
        embed_dec = embed_dec.permute(1, 0, 2)
        
        # Positional encoding

        embed_enc = self.pos_encoder(embed_enc)
        embed_dec = self.pos_decoder(embed_dec)
        
        out = self.transformer(embed_enc, embed_dec,
                               src_mask=self.enc_mask,
                               tgt_mask=self.dec_mask,
                               memory_mask=self.enc_dec_mask)
        out = out.permute(1, 0, 2)

        out_lstm, _ = self.lstm(out)
        #out_lstm , _ = self.lstm(encoded_layers[0])
    
        if self.args.use_res:
            out = torch.cat([out_lstm, out], dim = 2).contiguous().view(batch_size, -1, self.hidden_dim*2)
        else:
            out = out.contiguous().view(batch_size, -1, self.hidden_dim)
    
        out = self.fc(out).view(batch_size, -1)
        return out





class LongShort(ModelBase):
    def __init__(
        self,
        args,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_heads: int = 2,
        drop_out: float = 0.1,
        max_seq_len: float = 20,
        **kwargs
    ):
        super().__init__(
            args,
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags
        )

        self.args = args
        self.hidden_dim = args.hidden_dim
        self.n_layers = args.n_layers
        self.n_heads = args.n_heads
        self.drop_out = args.drop_out 
        self.max_seq_len = args.max_seq_len
        self.short_len = self.args.short_seq_len
        self.device = self.args.device

        self.n_tests = n_tests
        self.n_questions = n_questions
        self.n_tags = n_tags

        self.get_long_short()
        if self.args.use_graph:
            self.graph_dim = self.short.get_graph_emb_dim()
        

        self.long_lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )
        self.short_lstm = nn.LSTM(
            self.hidden_dim//4, self.hidden_dim//4 , self.n_layers, batch_first=True
        )

        self.long_transformer = nn.Transformer(
            d_model=self.hidden_dim, 
            nhead=self.n_heads,
            num_encoder_layers=self.n_layers, 
            num_decoder_layers=self.n_layers, 
            dim_feedforward=self.hidden_dim, 
            dropout=self.drop_out, 
            activation='relu')
        
        #self.short_transformer = nn.Transformer(
        #    d_model=self.hidden_dim, 
        #    nhead=self.n_heads,
        #    num_encoder_layers=self.n_layers, 
        #    num_decoder_layers=self.n_layers, 
        #    dim_feedforward=self.hidden_dim, 
        #    dropout=self.drop_out, 
        #    activation='relu')
        
        self.long_pos_encoder = PositionalEncoding(self.hidden_dim, self.drop_out, self.max_seq_len)
        self.long_pos_decoder = PositionalEncoding(self.hidden_dim, self.drop_out, self.max_seq_len)
        #self.short_pos_encoder = PositionalEncoding(self.hidden_dim, self.drop_out, self.max_seq_len)
        #self.short_pos_decoder = PositionalEncoding(self.hidden_dim, self.drop_out, self.max_seq_len)

        self.long_enc_mask = None
        self.long_dec_mask = None
        self.long_enc_dec_mask = None
        self.short_enc_mask = None
        self.short_dec_mask = None
        self.short_enc_dec_mask = None

        if (self.args.use_res == True) & (self.args.use_graph == True):  
            self.fc_long = nn.Linear(self.hidden_dim + self.graph_dim, self.hidden_dim//4)
            self.fc_short = self.fc_long = nn.Linear(self.hidden_dim//4 + self.graph_dim, self.hidden_dim//8)
        else:
            self.fc_long = nn.Linear(self.hidden_dim, self.hidden_dim//4)
            self.fc_short = nn.Linear(self.hidden_dim//4, self.hidden_dim//8)

        self.fc_last = nn.Linear(self.hidden_dim//4 + self.hidden_dim//8, 1)

    def get_long_short(self):
        self.short = ModelBase(
            self.args,
            self.hidden_dim//4,
            self.n_layers,
            self.n_tests,
            self.n_questions,
            self.n_tags
        )

        self.long = ModelBase(
            self.args,
            self.hidden_dim,
            self.n_layers,
            self.n_tests,
            self.n_questions,
            self.n_tags
        )

        
    def get_mask(self, seq_len):
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1)).to(torch.float)
        return mask
    

        
    ######################## FE시 추가해야함
    def forward(self, input_dic):
    
##################################Long################################
        Long_mask = input_dic['category']['mask']

        #enc_input = {key: input_dic[key] for key in ['testId', 'assessmentItemID', 'KnowledgeTag']}
        long_embed_enc, batch_size = self.long.dic_forward(input_dic)
        long_embed_dec, batch_size = self.long.dic_forward(input_dic)
        
        self.max_seq_len = input_dic['category']['interaction'].size(1)
        
        if self.long_enc_mask is None or self.long_enc_mask.size(0) != self.max_seq_len:
            self.long_enc_mask = self.get_mask(self.max_seq_len).to(self.device)

        if self.long_dec_mask is None or self.long_dec_mask.size(0) != self.max_seq_len:
            self.long_dec_mask = self.get_mask(self.max_seq_len).to(self.device)

        if self.long_enc_dec_mask is None or self.long_enc_dec_mask.size(0) != self.max_seq_len:
            self.long_enc_dec_mask = self.get_mask(self.max_seq_len).to(self.device)

        long_embed_enc = long_embed_enc.permute(1, 0, 2)
        long_embed_dec = long_embed_dec.permute(1, 0, 2)
        
        # Positional encoding

        long_embed_enc = self.long_pos_encoder(long_embed_enc)
        long_embed_dec = self.long_pos_decoder(long_embed_dec)
        
        long_out = self.long_transformer(long_embed_enc, long_embed_dec,
                               src_mask=self.long_enc_mask,
                               tgt_mask=self.long_dec_mask,
                               memory_mask=self.long_enc_dec_mask)
        long_out_trans = long_out.permute(1, 0, 2)

        long_out , _ = self.long_lstm(long_out_trans)
        if self.args.use_res:
            graph_emb = super().get_graph_emb(input_dic['category']['assessmentItemID'])
            long_out = torch.cat([long_out, graph_emb], dim = 2)
            long_out = long_out.contiguous().view(batch_size, -1, self.hidden_dim + self.graph_emb_dim)
        else:
            long_out = long_out.contiguous().view(batch_size, -1, self.hidden_dim)

        long_out = self.fc_long(long_out)

##################################Short################################

        #enc_input = {key: input_dic[key] for key in ['testId', 'assessmentItemID', 'KnowledgeTag']}
        short_embed_enc, batch_size = self.short.short_forward(input_dic, self.short_len)
        #short_embed_dec, batch_size = self.short.short_forward(input_dic, self.short_len)

        #if self.short_enc_mask is None or self.short_enc_mask.size(0) != self.max_seq_len:
        #    self.short_enc_mask = self.get_mask(self.max_seq_len).to(self.device)

        #if self.short_dec_mask is None or self.short_dec_mask.size(0) != self.max_seq_len:
        #   self.short_dec_mask = self.get_mask(self.max_seq_len).to(self.device)

        #if self.short_enc_dec_mask is None or self.short_enc_dec_mask.size(0) != self.max_seq_len:
        #   self.short_enc_dec_mask = self.get_mask(self.max_seq_len).to(self.device)

        #short_embed_enc = short_embed_enc.permute(1, 0, 2)
        #short_embed_dec = short_embed_dec.permute(1, 0, 2)
        
        # Positional encoding

        #short_embed_enc = self.short_pos_encoder(short_embed_enc)
        #short_embed_dec = self.short_pos_decoder(short_embed_dec)
        
        #short_out = self.short_transformer(short_embed_enc, short_embed_dec,
        #                       src_mask=self.short_enc_mask,
        #                       tgt_mask=self.short_dec_mask,
        #                       memory_mask=self.short_enc_dec_mask)
        
        #short_out_trans = short_out.permute(1, 0, 2)

        #short_out , _ = self.short_lstm(short_out_trans)
        short_out , _ = self.short_lstm(short_embed_enc)
        #out_lstm , _ = self.lstm(encoded_layers[0])


        if self.args.use_res:
            short_seq = input_dic['category']['assessmentItemID'][:, -self.max_seq_len:]
            padded_seq = self.short.pad(short_seq)
            graph_emb = self.short.get_graph_emb(padded_seq )
            short_out = torch.cat([short_out, graph_emb], dim = 2)
            short_out = short_out.contiguous().view(batch_size, -1, self.hidden_dim//4 + self.graph_emb_dim)
        else:
            short_out = short_out.contiguous().view(batch_size, -1, self.hidden_dim//4)

        short_out = self.fc_short(short_out)

        out = torch.cat([short_out, long_out], dim = 2)
        out = self.fc_last(out).view(batch_size, -1)

        return out




class sakt(ModelBase):  
    def __init__(
        self,
        args,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_heads: int = 2,
        drop_out: float = 0.1,
        max_seq_len: float = 20,
        **kwargs
    ):
        super().__init__(
            args,
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags
        )
        super(sakt, self).__init__()
        self.max_seq_len = args.seq_len
        self.emb_dim = args.hidden_dim // args.resize_factor
        self.n_heads = args.n_heads
        self.drop_out = args.drop_out

        self.embd_in = super().embedding_dict['interaction']        # Interaction embedding
        self.embd_pos = nn.Embedding( self.max_seq_len , embedding_dim = self.emb_dim)

        self.linear = nn.ModuleList( [nn.Linear(in_features= self.emb_dim , out_features= self.emb_dim ) for x in range(3)] )   # Linear projection for each embedding 
        self.attn = nn.MultiheadAttention(embed_dim= self.emb_dim , num_heads= self.n_head, dropout= self.drop_out )                                   
        self.ffn = nn.ModuleList([nn.Linear(in_features= self.emb_dim  , out_features=self.emb_dim , bias= True) for x in range(2)])  # feed forward layers post attention

        self.linear_out = nn.Linear(in_features= self.emb_dim  , out_features= 1 , bias=True) 
        self.layer_norm1 = nn.LayerNorm( self.emb_dim )
        self.layer_norm2 = nn.LayerNorm( self.emb_dim )                           # output with correctnness prediction 
        self.drop = nn.Dropout(self.drop_out)

    def get_mask(self, seq_len):
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1)).to(torch.float)
        return mask
    
    def forward( self , input_in , input_ex, input_dic):

        ## positional embedding 
        pos_in = self.embd_pos(torch.arange(self.max_seq_len).unsqueeze(0) )        #making a tensor of 12 numbers, .unsqueeze(0) for converting to 2d, so as to get a 3d output #print('pos embd' , pos_in.shape)
        ## get the interaction embedding output

        out_in = self.embd_in(input_dic['category']['interaction'])                         # (b, n) --> (b,n,d)
        out_in = out_in + pos_in

        ## split the interaction embeding into v and k ( needs to verify if it is slpited or not)
        value_in = out_in
        key_in   = out_in                                         #print('v,k ', value_in.shape)
        
        ## get the excercise embedding output

        query_ex = self.super().dic_forward(input_dic)                       # (b,n) --> (b,n,d) #print(query_ex.shape)
        
        ## Linearly project all the embedings
        value_in = self.linear[0](value_in).permute(1,0,2)        # (b,n,d) --> (n,b,d)
        key_in = self.linear[1](key_in).permute(1,0,2)
        query_ex =  self.linear[2](query_ex).permute(1,0,2)

        ## pass through multihead attention
        att_mask = self.get_mask(self, self.max_seq_len)
        atn_out , _ = self.attn(query_ex , key_in, value_in , attn_mask= att_mask, k=1).astype('bool')     # lower triangular mask, bool, torch    (n,b,d)
        atn_out = query_ex + atn_out                                  # Residual connection ; added excercise embd as residual because previous ex may have imp info, suggested in paper.
        atn_out = self.layer_norm1( atn_out )                          # Layer norm                        #print('atn',atn_out.shape) #n,b,d = atn_out.shape

        #take batch on first axis 
        atn_out = atn_out.permute(1,0,2)                              #  (n,b,d) --> (b,n,d)
        
        ## FFN 2 layers
        ffn_out = self.drop(self.ffn[1]( nn.ReLU()( self.ffn[0]( atn_out ) )))   # (n,b,d) -->    .view([n*b ,d]) is not needed according to the kaggle implementation
        ffn_out = self.layer_norm2( ffn_out + atn_out )                # Layer norm and Residual connection

        ## sigmoid
        ffn_out = torch.sigmoid(self.linear_out( ffn_out )  )

        return ffn_out