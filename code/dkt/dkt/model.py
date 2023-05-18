import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel
import pdb
import math
import numpy as np
import copy

class ModelBase(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_tests = n_tests
        self.n_questions = n_questions
        self.n_tags = n_tags

        # Embeddings
        # hd: Hidden dimension, intd: Intermediate hidden dimension
        hd, intd = hidden_dim, hidden_dim // 3
        self.embedding_interaction =nn.Embedding(3, intd) # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_test = nn.Embedding(n_tests + 1, intd)
        self.embedding_question = nn.Embedding(n_questions + 1, intd)
        self.embedding_tag = nn.Embedding(n_tags + 1, intd)
        #self.embedding_New Feature = nn.Embedding(n_New Feature + 1, intd)

######################## FE시 추가해야함
        self.embedding_dict = {}
        self.embedding_dict['testId'] = self.embedding_test
        self.embedding_dict['assessmentItemID'] = self.embedding_question
        self.embedding_dict['KnowledgeTag'] = self.embedding_tag
        self.embedding_dict['interaction'] = self.embedding_interaction
        #self.embedding_dict['New Feature'] = self.New Feature Embedding

        # Concatentaed Embedding Projection, Feature 개수 바뀌면 바꿔야함 4, 5, 6
        self.comb_proj = nn.Linear(intd * 4, hd)

        # Fully connected layer
        self.fc = nn.Linear(hd, 1)

    def dic_embed(self, input_dic):
        embed_list = []
        for feature, feature_seq in input_dic.items():
            if feature not in ('answerCode','mask'):
                batch_size = feature_seq.size(0)
                embed_list.append(self.embedding_dict[feature](feature_seq.long()))

        embed = torch.cat(embed_list, dim = 2)
        return embed, batch_size
    
    def dic_forward(self, input_dic):
        embed_list = []
        for feature, feature_seq in input_dic.items():
            if feature not in ('answerCode','mask'):
                batch_size = feature_seq.size(0)
                embed_list.append(self.embedding_dict[feature](feature_seq.long()))
        embed = torch.cat(embed_list, dim = 2)
        X = self.comb_proj(embed)
        return X, batch_size
    
######################## FE시 추가해야함
    def forward(self, testId, assessmentItemID, KnowledgeTag, answerCode, mask, interaction):
        batch_size = interaction.size(0)
        # Embedding
        embed_interaction = self.embedding_interaction(interaction.long())
        embed_test = self.embedding_test(testId.long())
        embed_question = self.embedding_question(assessmentItemID.long())
        embed_tag = self.embedding_tag(KnowledgeTag.long())
        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
            ],
            dim=2,
        )
        X = self.comb_proj(embed)
        return X, batch_size


class LSTM(ModelBase):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        **kwargs
    ):
        super().__init__(
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags
        )
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
        mask = input_dic['mask']

        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output).view(batch_size, -1)
        return out


class BERT(ModelBase):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        #
        n_heads: int = 2,
        drop_out: float = 0.1,
        max_seq_len: float = 20,
        **kwargs
    ):
        super().__init__(
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags
        )
        self.n_heads = n_heads #BERT 내의 어텐션 헤드 개수
        self.drop_out = drop_out #드롭아웃 비율

        # Bert config(설정)
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.n_layers,
            num_attention_heads=self.n_heads,
            max_position_embeddings=max_seq_len,
        )
        # BertModel 
        # : 입력으로 들어오는  임베딩과 그에 따른 어텐션 맵을 출력
        # : Transformer Encoder 레이어의 집합으로 구성 (어텐션 + feedforward + Add&Norm)
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
        mask = input_dic['mask']

        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0] # Bert모델에서 출력하는 마지막 레이어
        
        # 메모리 상의 데이터가 contiguous하게 저장되도록 텐서의 저장 방식 조정
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        
        # fully connected 
        out = self.fc(out).view(batch_size, -1)
        return out



class TransformerAndLSTMEncoderDeocoderEachEmbedding(ModelBase):
    def __init__(self, hidden_dim, embedding_size, 
                 n_assessmentItemID, n_tests, n_questions, n_tags, 
                 n_heads, n_layers, dropout_rate):

        super().__init__(
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags
        )
        
        self.num_assessmentItemID = n_assessmentItemID
      
        self.num_large_paper_number = self.args.num_large_paper_number
        self.num_hour = self.args.num_hour
        self.num_dayofweek = self.args.num_dayofweek
        self.num_week_number = self.args.num_week_number
        self.cat_cols = self.args.cat_cols
        self.num_cols = self.args.num_cols

        self.hidden_dim = hidden_dim
        self.embedding_size = embedding_size
        self.num_heads = n_heads
        self.num_layers = n_layers
        self.dropout_rate = dropout_rate
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # past
        past_emb = {}
        for cat_col in self.cat_cols:
            if cat_col == 'assessmentItemID2idx':
                past_emb[cat_col] = nn.Embedding(self.num_assessmentItemID + 1, self.embedding_size, padding_idx = 0) # 문항에 대한 정보
            elif cat_col == 'testId2idx':
                past_emb[cat_col] = nn.Embedding(self.num_testId + 1, self.embedding_size, padding_idx = 0) # 시험지에 대한 정보
            elif cat_col == 'KnowledgeTag2idx':
                past_emb[cat_col] = nn.Embedding(self.num_KnowledgeTag + 1, self.embedding_size, padding_idx = 0) # 지식 태그에 대한 정보
            elif cat_col == 'large_paper_number2idx':
                past_emb[cat_col] = nn.Embedding(self.num_large_paper_number + 1, self.embedding_size, padding_idx = 0) # 학년에 대한 정보
            elif cat_col == 'hour':
                past_emb[cat_col] = nn.Embedding(self.num_hour + 1, self.embedding_size, padding_idx = 0) # 문제 풀이 시간에 대한 정보
            elif cat_col == 'dayofweek':
                past_emb[cat_col] = nn.Embedding(self.num_dayofweek + 1, self.embedding_size, padding_idx = 0) # 문제 풀이 요일에 대항 정보
            elif cat_col == 'week_number':
                past_emb[cat_col] = nn.Embedding(self.num_week_number + 1, self.embedding_size, padding_idx = 0) # 문제 풀이 주에 대항 정보

        self.past_emb_dict = nn.ModuleDict(past_emb)

        self.past_answerCode_emb = nn.Embedding(3, self.hidden_dim, padding_idx = 0) # 문제 정답 여부에 대한 정보

        self.past_cat_emb = nn.Sequential(
            nn.Linear(len(self.cat_cols) * self.embedding_size, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2, eps=1e-6)
        )

        self.past_num_emb = nn.Sequential(
            nn.Linear(len(self.num_cols), self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2, eps=1e-6)
        )

        self.emb_layernorm = nn.LayerNorm(self.hidden_dim, eps=1e-6)

        self.past_lstm = nn.LSTM(
            input_size = self.hidden_dim,
            hidden_size = self.hidden_dim,
            num_layers = self.num_layers,
            batch_first = True,
            bidirectional = False,
            dropout = self.dropout_rate,
            )

        self.past_blocks = nn.ModuleList([SASRecBlock(self.n_heads, self.hidden_dim, self.dropout_rate) for _ in range(self.n_layers)])

        # now

        now_emb = {}
        for cat_col in self.cat_cols:
            if cat_col == 'assessmentItemID2idx':
                now_emb[cat_col] = nn.Embedding(self.num_assessmentItemID + 1, self.embedding_size, padding_idx = 0) # 문항에 대한 정보
            elif cat_col == 'testId2idx':
                now_emb[cat_col] = nn.Embedding(self.num_testId + 1, self.embedding_size, padding_idx = 0) # 시험지에 대한 정보
            elif cat_col == 'KnowledgeTag2idx':
                now_emb[cat_col] = nn.Embedding(self.num_KnowledgeTag + 1, self.embedding_size, padding_idx = 0) # 지식 태그에 대한 정보
            elif cat_col == 'large_paper_number2idx':
                now_emb[cat_col] = nn.Embedding(self.num_large_paper_number + 1, self.embedding_size, padding_idx = 0) # 학년에 대한 정보
            elif cat_col == 'hour':
                now_emb[cat_col] = nn.Embedding(self.num_hour + 1, self.embedding_size, padding_idx = 0) # 문제 풀이 시간에 대한 정보
            elif cat_col == 'dayofweek':
                now_emb[cat_col] = nn.Embedding(self.num_dayofweek + 1, self.embedding_size, padding_idx = 0) # 문제 풀이 요일에 대항 정보
            elif cat_col == 'week_number':
                now_emb[cat_col] = nn.Embedding(self.num_week_number + 1, self.embedding_size, padding_idx = 0) # 문제 풀이 주에 대항 정보

        self.now_emb_dict = nn.ModuleDict(now_emb)

        self.now_cat_emb = nn.Sequential(
            nn.Linear(len(self.cat_cols) * self.embedding_size, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2, eps=1e-6)
        )

        self.now_num_emb = nn.Sequential(
            nn.Linear(len(self.num_cols), self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2, eps=1e-6)
        )

        self.now_lstm = nn.LSTM(
            input_size = self.hidden_dim,
            hidden_size = self.hidden_dim,
            num_layers = self.num_layers,
            batch_first = True,
            bidirectional = False,
            dropout = self.dropout_rate,
            )

        self.now_blocks = nn.ModuleList([SASRecBlock(self.num_heads, self.hidden_dim, self.dropout_rate) for _ in range(self.num_layers)])

        # predict

        self.dropout = nn.Dropout(self.dropout_rate)

        self.predict_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 1),
            nn.Sigmoid()
        )
    
    
    def forward(self, input):
        """
        past_cat_feature : (batch_size, max_len, cat_cols)
        past_num_feature : (batch_size, max_len, num_cols)
        past_answerCode : (batch_size, max_len)

        now_cat_feature : (batch_size, max_len, cat_cols)
        now_num_feature : (batch_size, max_len, num_cols)
        
        """

        past_cat_feature = input['past_cat_feature'].to(self.device)
        past_num_feature = input['past_num_feature'].to(self.device) 
        past_answerCode = input['past_answerCode']
        now_cat_feature = input['now_cat_feature'].to(self.device)
        now_num_feature = input['now_num_feature'].to(self.device)

        # masking 
        mask_pad = torch.BoolTensor(past_answerCode > 0).unsqueeze(1).unsqueeze(1) # (batch_size, 1, 1, max_len)
        mask_time = (1 - torch.triu(torch.ones((1, 1, past_answerCode.size(1), past_answerCode.size(1))), diagonal=1)).bool() # (batch_size, 1, max_len, max_len)
        mask = (mask_pad & mask_time).to(self.device) # (batch_size, 1, max_len, max_len)

        # past
        past_cat_emb_list = []
        for idx, cat_col in enumerate(self.cat_cols):
            past_cat_emb_list.append(self.past_emb_dict[cat_col](past_cat_feature[:, :, idx]))

        past_cat_emb = torch.concat(past_cat_emb_list, dim = -1)
        past_cat_emb = self.past_cat_emb(past_cat_emb)
        past_num_emb = self.past_num_emb(past_num_feature)

        past_emb = torch.concat([past_cat_emb, past_num_emb], dim = -1)
        past_emb += self.past_answerCode_emb(past_answerCode.to(self.device))
        past_emb = self.emb_layernorm(past_emb) # LayerNorm

        for block in self.past_blocks:
            past_emb, attn_dist = block(past_emb, mask)

        past_emb, _ = self.past_lstm(past_emb)

        # now
        now_cat_emb_list = []
        for idx, cat_col in enumerate(self.cat_cols):
            now_cat_emb_list.append(self.now_emb_dict[cat_col](now_cat_feature[:, :, idx]))

        now_cat_emb = torch.concat(now_cat_emb_list, dim = -1)
        now_cat_emb = self.now_cat_emb(now_cat_emb)
        now_num_emb = self.now_num_emb(now_num_feature)

        now_emb = torch.concat([now_cat_emb, now_num_emb], dim = -1)

        for block in self.now_blocks:
            now_emb, attn_dist = block(now_emb, mask)

        now_emb, _ = self.now_lstm(now_emb)
        
        # predict
        emb = torch.concat([self.dropout(past_emb), self.dropout(now_emb)], dim = -1)
        output = self.predict_layer(emb)

        return output.squeeze(2)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)

class Saint(ModelBase):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_heads: int = 2,
        max_seq_len : int = 20,
        device : str = 'gpu',
        **kwargs
    ):
        super().__init__(
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags
        )
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.dropout = 0.1
        self.device = device
        # self.dropout = self.args.dropout

        # decoder combination projection
        self.enc_comb_proj = nn.Linear((self.hidden_dim//3)*3, self.hidden_dim)
        self.dec_comb_proj = nn.Linear((self.hidden_dim//3)*4, self.hidden_dim)

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
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1))

        return mask.masked_fill(mask==1, float('-inf'))
    
    
    def forward(self, input_dic):
   
        enc_input = {key: input_dic[key] for key in ['testId', 'assessmentItemID', 'KnowledgeTag']}
        embed_enc, batch_size = super().dic_embed(enc_input)
        embed_dec, batch_size = super().dic_embed(input_dic)
        
        seq_len = input_dic['interaction'].size(1)
        
        embed_enc = self.enc_comb_proj(embed_enc)
        embed_dec = self.dec_comb_proj(embed_dec)
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
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds


#####################saint plus
"""
Reference:
https://arxiv.org/abs/2002.07033
"""

class FFN(nn.Module):
    def __init__(self, d_ffn, d_model, dropout):
        super(FFN, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ffn) #[batch, seq_len, ffn_dim]
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(d_ffn, d_model) #[batch, seq_len, d_model]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu_1(x)
        x = self.linear_2(x)
        return self.dropout(x)

class SaintPlus(nn.Module):
    def __init__(self, seq_len, num_layers, d_ffn, d_model, num_heads, max_len, n_questions, n_tasks, dropout):
        super(SaintPlus, self).__init__()
        self.d_model = d_model
        self.n_questions = n_questions
        self.num_heads = num_heads

        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.contentId_emb = nn.Embedding(n_questions+1, d_model)
        #self.task_emb = nn.Embedding(n_tasks+1, d_model)
        
        self.timelag_emb = nn.Linear(1, d_model, bias=False)
        self.elapsedT_emb = nn.Linear(1, d_model, bias=False)
        self.itemAver_emb = nn.Linear(1, d_model, bias=False)
        self.userAver_emb = nn.Linear(1, d_model, bias=False)
        self.tagAver_emb =nn.Linear(1, d_model, bias=False)
        
        self.answerCorr_emb = nn.Embedding(3, d_model)

        self.emb_dense1 = nn.Linear(2*d_model, d_model)
        self.emb_dense2 = nn.Linear(6*d_model, d_model)

        self.transformer = nn.Transformer(d_model=d_model, nhead=num_heads, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=d_ffn, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.FFN = FFN(d_ffn, d_model, dropout=dropout)
        self.final_layer = nn.Linear(d_model, 1)

    def forward(self, content_ids,time_lag, ques_elapsed_time,  item_aver, user_aver ,tag_aver,answer_correct):
        device = content_ids.device
        seq_len = content_ids.shape[1]

        content_id_emb = self.contentId_emb(content_ids)
        #task_id_emb = self.task_emb(test_id)
        time_lag = torch.log(time_lag+1)
        time_lag = time_lag.view(-1, 1) # [batch*seq_len, 1]
        time_lag = self.timelag_emb(time_lag) # [batch*seq_len, d_model]
        time_lag = time_lag.view(-1, seq_len, self.d_model) # [batch, seq_len, d_model]
        elapsed_time = torch.log(ques_elapsed_time+1)
        elapsed_time = elapsed_time.view(-1, 1) # [batch*seq_len, 1]
        elapsed_time = self.elapsedT_emb(elapsed_time) # [batch*seq_len, d_model]
        elapsed_time = elapsed_time.view(-1, seq_len, self.d_model) # [batch, seq_len, d_model]
        ####
        item_aver = torch.log(item_aver+1)
        item_aver = item_aver.view(-1, 1) # [batch*seq_len, 1]
        item_aver = self.itemAver_emb(item_aver) # [batch*seq_len, d_model]
        item_aver = item_aver.view(-1, seq_len, self.d_model) # [batch, seq_len, d_model]

        user_aver = torch.log(user_aver+1)
        user_aver = user_aver.view(-1, 1) # [batch*seq_len, 1]
        user_aver = self.userAver_emb(user_aver) # [batch*seq_len, d_model]
        user_aver = user_aver.view(-1, seq_len, self.d_model) # [batch, seq_len, d_model]
        
        tag_aver = torch.log(tag_aver+1)
        tag_aver = tag_aver.view(-1, 1) # [batch*seq_len, 1]
        tag_aver = self.tagAver_emb(tag_aver) # [batch*seq_len, d_model]
        tag_aver = tag_aver.view(-1, seq_len, self.d_model) # [batch, seq_len, d_model]

        answer_correct_emb = self.answerCorr_emb(answer_correct)

        encoder_val = torch.cat((content_id_emb ,time_lag), axis=-1)
        encoder_val = self.emb_dense1(encoder_val)
        decoder_val = torch.cat((time_lag, elapsed_time, item_aver, user_aver ,tag_aver,answer_correct_emb), axis=-1)
        decoder_val = self.emb_dense2(decoder_val)

    
        pos = torch.arange(seq_len).unsqueeze(0).to(device)
        pos_emb = self.pos_emb(pos)
        encoder_val += pos_emb
        decoder_val += pos_emb

        over_head_mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1).astype('bool'))
        over_head_mask = over_head_mask.to(device)

        encoder_val = encoder_val.permute(1, 0, 2)
        decoder_val = decoder_val.permute(1, 0, 2)
        decoder_val = self.transformer(encoder_val, decoder_val, src_mask=over_head_mask, tgt_mask=over_head_mask, memory_mask=over_head_mask)

        decoder_val = self.layer_norm(decoder_val)
        decoder_val = decoder_val.permute(1, 0, 2)

        final_out = self.FFN(decoder_val)
        final_out = self.layer_norm(final_out + decoder_val)
        final_out = self.final_layer(final_out)
        final_out = torch.sigmoid(final_out)
        return final_out.squeeze(-1)

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))