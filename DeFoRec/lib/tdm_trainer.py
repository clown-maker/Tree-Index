from __future__ import print_function
from .Cluster_Faiss import Cluster_Faiss
import numpy as np
import torch
import math
from .Network_Model import TDMModel
from .Tree_Model import Tree
from .negative_sampling import top_down_sample,uniform_sampling_multiclassifcation,softmax_sampling,all_negative_sampling
import torch.nn.functional as F
import time
class TrainModel:
    def __init__(self,
                 ids,
                 codes,
                 embed_dim=24,
                 feature_groups=[20,20,10,10,2,2,2,1,1,1],
                 optimizer=lambda params: torch.optim.Adam(params, lr=1e-3, amsgrad=True,weight_decay=0.1),
                 parall=3,
                 N=40,
                 temperature=1.0,
                 sampling_method='uniform',
                 device='cuda'):
        self.device=device
        self.embed_dim=embed_dim
        self.feature_groups=feature_groups
        self.opti=optimizer
        self.parall=parall
        self.item_num=len(ids)
        self.feature_num=sum(feature_groups)
        self.N=N
        self.temperature=temperature
        self.sample_method=sampling_method
        self.tree_list = []

        self.tree=Tree(ids,codes)
        self.tree_list.append(self.tree)

        #model part
        self.tdm_model=TDMModel(embed_dim=self.embed_dim,feature_groups=feature_groups,item_num=self.item_num,
                                node_num=self.tree.node_num,
                                item_node_sharing_embedding=False).to(self.device)
        self.model_list=[self.tdm_model]

        #optimizer
        self.optimizer = optimizer(self.tdm_model.parameters())

        #cluster to update the tree
        self.cluster=Cluster_Faiss(parall=parall)
        self.batch_num=0


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

    def update_TDM_model(self,batch_x,batch_y):#batch_x is the rows of training instances, batch_y is the corresponding labels.
        self.batch_num+=1
        all_leaf_codes=np.array([self.tree.item_id_leaf_code[label.item()] for label in batch_y],dtype=np.int64)
        #st=time.time()
        loss=self.generate_training_instances(batch_x,all_leaf_codes)
        loss.backward()#compute the gradient
        self.optimizer.step()# update the parameters
        self.optimizer.zero_grad()# clean the gradient
        #print('update time is {}'.format(time.time()-st))
        self.update_learning_rate(self.batch_num)
        return loss

    def use_all_negative(self,batch_x, all_leaf_codes):  # all_codes is a array
        m,d=batch_x.shape
        sample_labels=all_negative_sampling(batch_x,self.tree,self.tdm_model,all_leaf_codes,self.N,self.device,
                                                   temperature=self.temperature)
        start_col,loss=0,0.0
        for p_layer in range(self.tree.max_layer_id):
            layer_node_num=self.tree.layer_node_num_list[p_layer+1]
            positive_labels=sample_labels[:,start_col:start_col+1]
            negative_labels=sample_labels[:,start_col+1:start_col+layer_node_num]
            all_labels=torch.cat((positive_labels,negative_labels),-1)
            o_pi= self.tdm_model.preference(batch_x.to(self.device).repeat(1,layer_node_num).view(-1,d),\
                                                              all_labels.view(-1,1))[:, 0].view(m,layer_node_num)
            loss += (torch.logsumexp(o_pi,dim=1)-o_pi[:,0]).mean(-1)
            start_col+=layer_node_num
        return loss
    def generate_training_instances(self,batch_x, all_leaf_codes):  # all_codes is a array
        m,d=batch_x.shape
        if self.sample_method=='softmax':
            samples,log_q_matrix=softmax_sampling(batch_x,self.tree,self.tdm_model,all_leaf_codes,self.N,self.device,
                                                   temperature=self.temperature)
        elif self.sample_method=='uniform_multiclass':
            samples,log_q_matrix=uniform_sampling_multiclassifcation(batch_x,self.tree,self.tdm_model,all_leaf_codes,
                                                                     self.N,self.device,temperature=self.temperature)
        elif self.sample_method=='top_down':
            samples,log_q_matrix=top_down_sample(batch_x,self.tree,self.tdm_model,all_leaf_codes,self.N,self.device,
                                                   temperature=self.temperature)
        elif self.sample_method=='all_negative_sampling':
            return self.use_all_negative(batch_x, all_leaf_codes)
        else:
            assert False,'{} sampling is an invalid choice'.format(self.sample_method)
        effective_index=samples>=0
        user_index=torch.arange(m).view(-1,1).expand(samples.shape)[effective_index]

        training_labels = samples[effective_index].view(-1, 1)

        o_pi=torch.full(samples.shape,-1e9,device=self.device,dtype=torch.float32)
        #print('max lable is {}, node num is {}'.format(samples.max(),self.tree.node_num))
        o_pi[effective_index]=self.tdm_model.preference(batch_x.to(self.device)[user_index],
                                                                  training_labels)[:,0]-log_q_matrix[effective_index]
        return (torch.logsumexp(o_pi.view(m*self.tree.max_layer_id,self.N+1),dim=1)-o_pi.view(-1,self.N+1)[:,0]).mean(-1)


    def create_new_tree(self,temperature=1.0,eps=0.00000001):
        cluster_data=self.tdm_model.item_embedding.weight.data.cpu().numpy()[0:self.item_num,:]
        print('cluster data shape {},{}'.format(*cluster_data.shape))
        ids,codes =self.cluster.train(ids=np.arange(self.item_num),data=cluster_data)#return ids, codes

        self.tree=Tree(ids,codes)
        self.tree_list.append(self.tree)

        del self.tdm_model
        if self.device !='cpu':
            torch.cuda.empty_cache()
        self.tdm_model=TDMModel(embed_dim=self.embed_dim,feature_groups=self.feature_groups,item_num=self.item_num,
                                node_num=self.tree.node_num,
                                item_node_sharing_embedding=False).to(self.device)
        self.model_list.append(self.tdm_model)
        #optimizer
        self.optimizer = self.opti(self.tdm_model.parameters())
        self.batch_num = 0

    def predict(self, user_x, N, topk):  # input one user
        candidate_codes = {0}  # start from root node
        result_set_A = set()  # result_set_A contains leaf node codes
        while len(candidate_codes) > 0:
            leaf_nodes = {code for code in candidate_codes if int(math.log(code + 1, 2)) == self.tree.max_layer_id}
            result_set_A |= leaf_nodes  # result_set_A contains leaf node codes
            candidate_codes -= leaf_nodes
            if len(candidate_codes) <= 0: break
            test_labels = torch.LongTensor([self.tree.node_code_node_id[code] for code in candidate_codes]).to(
                self.device).view(-1, 1)
            test_codes = torch.tensor([code for code in candidate_codes], device=self.device).view(-1)
            with torch.no_grad():  # log_probs is [len(test_labels),1]
                log_probs = self.tdm_model.preference(
                    user_x.to(self.device).expand(len(candidate_codes), user_x.shape[-1]),test_labels)[:, 0].view(-1)
            _, index = log_probs.sort(dim=-1)
            selected_codes = test_codes.gather(index=index[-N:], dim=-1)
            candidate_codes.clear()
            for code in selected_codes:
                c = int(code.item())
                if 2 * c + 1 in self.tree.node_code_node_id:
                    candidate_codes.add(2 * c + 1)
                if 2 * c + 2 in self.tree.node_code_node_id:
                    candidate_codes.add(2 * c + 2)
        result_labels = torch.LongTensor(
            [self.tree.node_code_node_id[code] for code in result_set_A])  # result_set_A contains leaf node codes
        result_codes = torch.LongTensor([code for code in result_set_A])
        with torch.no_grad():
            log_probs = self.tdm_model.preference(user_x.to(self.device).expand(len(result_set_A), user_x.shape[-1]),
                                                  result_labels.to(self.device).view(-1, 1))[:, 0].view(-1)
        sorted_values, index = log_probs.sort(dim=-1)
        return [self.tree.leaf_code_item_id[code] for code in
                result_codes.gather(index=index[-topk:].cpu(), dim=-1).numpy()]


