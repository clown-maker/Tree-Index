from typing import OrderedDict
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_

# from recstudio.ann import sampler
# from recstudio.data import dataset
# from recstudio.model import basemodel, loss_func, module, scorer

class SeqPoolingLayer(torch.nn.Module):
    #TODO: make it a pooling function or rename it as sequence pooling
    # mean, max, last
    def __init__(self, pooling_type='mean', keepdim=False) -> None:
        super().__init__()
        if not pooling_type in ['sum', 'mean', 'max', 'last']:
            raise ValueError("pooling_type can only be one of ['sum', 'mean', 'max', 'last']"\
                f"but {self.pooling_type} is given.")
        self.pooling_type = pooling_type
        self.keepdim = keepdim


    def forward(self, batch_seq_embeddings, seq_len, weight=None):
        # batch_seq_embeddings: [B, L, D]
        # seq_len: [B], weight: [BxL]
        if weight is not None:
            batch_seq_embeddings = weight.unsqueeze(1) * batch_seq_embeddings
        if self.pooling_type in ['mean', 'sum', 'max']:
            mask = torch.arange(batch_seq_embeddings.size(1)).unsqueeze(0).unsqueeze(2).to(batch_seq_embeddings.device)
            mask = mask.expand(batch_seq_embeddings.size(0), -1,  batch_seq_embeddings.size(2))
            seq_len = seq_len.unsqueeze(1).unsqueeze(2) + torch.finfo(torch.float32).eps
            seq_len_ = seq_len.expand(-1, mask.size(1), -1)
            mask = mask >= seq_len_
            if self.pooling_type == 'max':
                batch_seq_embeddings = batch_seq_embeddings.masked_fill(mask, 0.0)
                if not self.keepdim:
                    return batch_seq_embeddings.max(dim=1)
                else:
                    return batch_seq_embeddings.max(dim=1).unsqueeze(1)
            else:
                batch_seq_embeddings = batch_seq_embeddings.masked_fill(mask, 0.0)
                batch_seq_embeddings_sum = batch_seq_embeddings.sum(dim=1, keepdim=self.keepdim)
                if self.pooling_type == 'sum':
                    return batch_seq_embeddings_sum
                else:
                    return batch_seq_embeddings_sum / (seq_len if self.keepdim else seq_len.squeeze(2))

        elif self.pooling_type == 'last':
            gather_index = (seq_len-1).view(-1, 1, 1).expand(-1, -1, batch_seq_embeddings.size(2)) # B x 1 x D
            output = batch_seq_embeddings.gather(dim=1, index=gather_index).squeeze(1)  # B x D
            return output if not self.keepdim else output.unsqueeze(1)


class SASRecQueryEncoder(torch.nn.Module):
    def __init__(self, embed_dim=96,d_model=512, max_seq_len=69, n_head=4, hidden_size=1024,\
                 dropout=0.2, activation='gelu', n_layer=2,item_embedding=None, bidirectional=False) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.item_embedding = item_embedding#itme embedding learnable

        self.position_emb = torch.nn.Embedding(max_seq_len, embed_dim)
        transformer_encoder = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=hidden_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=False
        )
        self.transformer_layer = torch.nn.TransformerEncoder(
            encoder_layer=transformer_encoder,
            num_layers=n_layer,
        )
        self.dropout = torch.nn.Dropout(p=dropout)
        self.gather_layer = SeqPoolingLayer(pooling_type='last')
        


    def forward(self, user_hist, mask=None):#
        """
        user_hist:[bs,max_seq_len]
        """
        #user_hist = batch['in_'+self.fiid]
        positions = torch.arange(user_hist.shape[1], \
            dtype=torch.long, device=user_hist.device).expand(*(user_hist.shape))
        position_embs = self.position_emb(positions)
        seq_embs = self.item_embedding(user_hist)

        mask4padding = user_hist== self.item_embedding.weight.size(0)-1  # BxL
        L = user_hist.shape[1]
        if not self.bidirectional:
            attention_mask = ~torch.tril(torch.ones((L, L), dtype=torch.bool, device=user_hist.device))# upper triangle matrix
        else:
            attention_mask = torch.zeros((L, L), dtype=torch.bool, device=user_hist.device)
        
        transformer_out = self.transformer_layer(
            src=self.dropout(seq_embs+position_embs), 
            mask=attention_mask, 
            src_key_padding_mask=mask4padding)  # BxLxD
        # return transformer_out[:,-1,:]
        if mask is not None:
            return transformer_out[mask]
        else:
            seqlen = (~mask4padding).float().sum(-1).long()
            return self.gather_layer(transformer_out, seqlen)




class SASRec(torch.nn.Module):
    r"""
    SASRec models user's sequence with a Transformer.
    Model hyper parameters:
        - ``embed_dim(int)``: The dimension of embedding layers. Default: ``64``.
        - ``hidden_size(int)``: The output size of Transformer layer. Default: ``128``.
        - ``layer_num(int)``: The number of layers for the Transformer. Default: ``2``.
        - ``dropout_rate(float)``:  The dropout probablity for dropout layers after item embedding
         | and in Transformer layer. Default: ``0.5``.
        - ``head_num(int)``: The number of heads for MultiHeadAttention in Transformer. Default: ``2``.
        - ``activation(str)``: The activation function in transformer. Default: ``"gelu"``.
        - ``layer_norm_eps``: The layer norm epsilon in transformer. Default: ``1e-12``.
    """
    def __init__(self,item_num=100,embed_dim=96,d_model=512,max_seq_len=69,n_head=2,\
        hidden_size=1024,dropout=0.5,n_layer=2,\
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3, amsgrad=True),device='cpu'):
        super().__init__()
        self.item_num=item_num
        self.item_embedding=torch.nn.Embedding(item_num+1,embed_dim,padding_idx=item_num)
        self.query_encoder=SASRecQueryEncoder(
            embed_dim=embed_dim,
            d_model=d_model,
            max_seq_len=max_seq_len,
            n_head=n_head,
            hidden_size=hidden_size,
            dropout=dropout,
            n_layer=n_layer,
            item_embedding=self.item_embedding
        )
        self.optimizer = optimizer(self.parameters())
        self.batch_num=0
        self.device=device
        self.to(device)
        self.apply(self.xavier_normal_initialization)

    def _get_score_func(self,user_hist,positive_ids,negative_ids):
        """
        user_hist:[bs,max_len]
        positive_ids,negative_ids: [bs,1]
        """
        hidden_output=self.query_encoder(user_hist)#hidden_output [bs,max_len,dim]

        positive_scores=(hidden_output*self.item_embedding(positive_ids.view(-1))).sum(-1)
        negative_socres=(hidden_output*self.item_embedding(negative_ids.view(-1))).sum(-1)
        return positive_scores,negative_socres


    def _get_loss_func(self,positive_socres,negative_scores):
        """
        positive_socres: [bs,1]
        negative_scores: [bs,1]
        """
        # print(positive_socres,negative_scores)
        loss = ((-F.logsigmoid(positive_socres)) + F.softplus(negative_scores)).mean()
        # loss= (-torch.log(torch.sigmoid(positive_socres))-torch.log(1.0-torch.sigmoid(negative_scores))).mean()
        return loss

    def _get_sampler(self,batch_label):
        #batch_label [bs,1]
        r"""Uniform sampler is used as negative sampler."""
        bs=batch_label.shape[0]
        #print(batch_label.shape)
        positive_index=torch.full((bs,),True,dtype=torch.bool,device=batch_label.device)
        negative_sample_ids=torch.full((bs,),-1,dtype=torch.int64,device=batch_label.device)
        negative_sample_ids[:]=batch_label[:]
        temp_sum=positive_index.sum().item()
        while temp_sum>0:
            negative_sample_ids[positive_index]=torch.randint(low=0,high=self.item_num,size=(temp_sum,),dtype=torch.int64, device=self.device)
            positive_index=negative_sample_ids==batch_label
            temp_sum=positive_index.sum().item()
        return negative_sample_ids

    def update_learning_rate(self, t, learning_rate_base=1e-3, warmup_steps=5000,
                             decay_rate=1./3, learning_rate_min=1e-5):
        """ Learning rate with linear warmup and exponential decay """
        lr = learning_rate_base * np.minimum(
            (t + 1.0) / warmup_steps,
            np.exp(decay_rate * ((warmup_steps - t - 1.0) / warmup_steps)),
        )
        lr = np.maximum(lr, learning_rate_min)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def update(self,user_hist,positive_labels):
        """
        user_hist:[bs,max_len]
        positive_labels: [bs,1]
        """
        # user_hist[user_hist<0]=self.item_num
        # user_hist=torch.flip(input=user_hist,dims=[1])
        user_hist = self.adjust(user_hist)
        negative_sample_ids=self._get_sampler(positive_labels)
        positive_scores,negative_socres=\
            self._get_score_func(user_hist,positive_labels,negative_sample_ids)
        loss=self._get_loss_func(positive_scores,negative_socres)
        loss.backward()#compute the gradient
        self.optimizer.step()# update the parameters
        self.optimizer.zero_grad()# clean the gradient
        self.update_learning_rate(self.batch_num)
        self.batch_num+=1
        return loss

    def predict(self,user_hist,topk=20):
        """
        user_hist: [bs,max_len]
        """
        # user_hist[user_hist<0]=self.item_num
        # user_hist=torch.flip(input=user_hist,dims=[1])
        user_hist = self.adjust(user_hist)

        with torch.no_grad():
            hidden_output=self.query_encoder(user_hist)#hidden_output:[bs,dim]
            all_item_id=torch.arange(self.item_num,dtype=torch.int64,device=user_hist.device)
            item_embeddings=self.item_embedding(all_item_id)

            #fast version
            _, indices=torch.topk(torch.mm(hidden_output,item_embeddings.T),k=topk,dim=-1)

            #slow version
            #out=(hidden_output.squeeze(1)*item_embeddings).sum(-1)#[bs,item_num]
            #indices=out.argsort(dim=-1,descending=True)[:,:topk]

        return indices.cpu().numpy()


    def adjust(self,user_hist):
        user_hist2 = torch.zeros_like(user_hist) + self.item_num
        padding_mask = user_hist>=0
        len = padding_mask.float().sum(dim=-1).unsqueeze(-1) # Bx1
        mask = torch.arange(user_hist.size(1), device=user_hist.device)\
                                        .expand(*(user_hist.shape)) < len
        user_hist2[mask] = user_hist[padding_mask]
        return user_hist2

    def xavier_normal_initialization(self,module):
        if isinstance(module, torch.nn.Embedding):
            xavier_normal_(module.weight.data)
            if module.padding_idx is not None:
                constant_(module.weight.data[module.padding_idx], 0.)
        elif isinstance(module, torch.nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)


    def normal_initialization(self,module):
        if isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                constant_(module.weight.data[module.padding_idx], 0.)
        elif isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
