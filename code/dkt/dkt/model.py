import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel
import pdb

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
        self.embedding_interaction = nn.Embedding(3, intd) # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_test = nn.Embedding(n_tests + 1, intd)
        self.embedding_question = nn.Embedding(n_questions + 1, intd)
        self.embedding_tag = nn.Embedding(n_tags + 1, intd)

        # Concatentaed Embedding Projection
        self.comb_proj = nn.Linear(intd * 4, hd)

        # Fully connected layer
        self.fc = nn.Linear(hd, 1)
    
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
    def forward(self, testId, assessmentItemID, KnowledgeTag, answerCode, mask, interaction):
        X, batch_size = super().forward(testId=testId,
                                        assessmentItemID=assessmentItemID,
                                        KnowledgeTag=KnowledgeTag,
                                        answerCode=answerCode,
                                        mask=mask,
                                        interaction=interaction)
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
    def forward(self, testId, assessmentItemID, KnowledgeTag, answerCode, mask, interaction):
        X, batch_size = super().forward(testId=testId,
                                        assessmentItemID=assessmentItemID,
                                        KnowledgeTag=KnowledgeTag,
                                        answerCode=answerCode,
                                        mask=mask,
                                        interaction=interaction)

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
    def forward(self, testId, assessmentItemID, KnowledgeTag, answerCode, mask, interaction):
        X, batch_size = super().forward(testId=testId,
                                        assessmentItemID=assessmentItemID,
                                        KnowledgeTag=KnowledgeTag,
                                        answerCode=answerCode,
                                        mask=mask,
                                        interaction=interaction)

        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
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