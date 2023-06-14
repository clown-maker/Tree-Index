import torch
from .SASRec_Model import SASRecQueryEncoder
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_normal_, constant_

class BERT4Rec(torch.nn.Module):

    def __init__(self, item_num=100,embed_dim=96,d_model=512, max_seq_len=69, n_head=4, hidden_size=1024,\
                 dropout=0.3, activation='gelu', n_layer=2, mask_ratio=0.1,optimizer=None,device='cpu'):
        super().__init__()
        self.item_num=item_num
        self.mask_token = item_num+1
        self.mask_ratio = mask_ratio
        self.item_embedding=torch.nn.Embedding(item_num+2,embed_dim,padding_idx=item_num)
        self.query_encoder=SASRecQueryEncoder(
            embed_dim=embed_dim,
            d_model=d_model,
            max_seq_len=max_seq_len,
            n_head=n_head,
            hidden_size=hidden_size,
            activation=activation,
            dropout=dropout,
            n_layer=n_layer,
            item_embedding=self.item_embedding,
            bidirectional=True
        )
        self.optimizer = optimizer(self.parameters())
        self.batch_num=0
        self.device=device
        self.to(device)
        self.apply(self.normal_initialization)

    def _get_score_func(self, user_hist, mask, positive_ids,negative_ids):
        """
        user_hist:[bs,max_len]
        positive_ids,negative_ids: [bs,1]
        """
        hidden_output=self.query_encoder(user_hist, mask)#hidden_output [N,dim]

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
            temp_sum= positive_index.sum().item()
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

        user_hist, positive_labels, mask = self._reconstruct_train_data(user_hist)

        negative_sample_ids=self._get_sampler(positive_labels)
        positive_scores,negative_socres=\
            self._get_score_func(user_hist,mask,positive_labels,negative_sample_ids)
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
            _,index=torch.topk(torch.mm(hidden_output,item_embeddings.T),k=topk,dim=-1)
        return index.cpu().numpy()


    def _reconstruct_train_data(self, item_seq):
        # item_seq = batch['in_'+self.fiid]
        device = item_seq.device

        padding_mask = item_seq == self.item_num
        rand_prob = torch.rand_like(item_seq, dtype=torch.float, device=device)
        rand_prob.masked_fill_(padding_mask, 1.0)
        masked_mask = rand_prob < self.mask_ratio
        masked_token = item_seq[masked_mask]
        # masked_num = masked_mask.float().sum(-1)

        item_seq[masked_mask] = self.mask_token
        # batch['in_'+self.fiid] = item_seq

        # batch[self.fiid] = masked_token     # N
        # batch['seqlen'] = masked_mask
        return item_seq, masked_token, masked_mask

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

