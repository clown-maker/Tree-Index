from __future__ import print_function
import numpy as np
import torch
from .Network_Model import TDMModel
import math
from .Tree_Model import Tree
from .tree_learning import Tree_Learner
from .negative_sampling import top_down_sample,uniform_sampling_multiclassifcation,softmax_sampling,all_negative_sampling
class TrainModel:
    def __init__(self,
                 ids,
                 codes,
                 all_training_instance=None,
                 item_user_pair_dict=None,
                 embed_dim=24,
                 feature_groups=[20,20,10,10,2,2,2,1,1,1],
                 optimizer=lambda params: torch.optim.Adam(params, lr=1e-3, amsgrad=True),
                 parall=3,
                 temperature=1.0,
                 N=40,
                 sampling_method='uniform',
                 device='cuda'):
        self.initial_codes=codes
        self.device=device
        self.embed_dim=embed_dim
        self.feature_groups=feature_groups
        self.opti=optimizer
        self.parall=parall
        self.item_num=len(ids)
        self.feature_num=sum(feature_groups)
        self.temperature=temperature
        self.sample_method=sampling_method
        self.N=N


        self.tree=Tree(ids,codes)
        self.tree_list=[self.tree]
        #model part
        self.jtm_model=TDMModel(embed_dim=embed_dim,feature_groups=feature_groups,item_num=self.item_num,
                                node_num=self.tree.node_num,item_node_sharing_embedding=True).to(self.device)
        self.model_list=[self.jtm_model]
        #optimizer
        self.optimizer = optimizer(self.jtm_model.parameters())

        self.tree_learner=Tree_Learner(all_training_instance,item_user_pair_dict)

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

    def update_JTM_model(self,batch_x,batch_y):#batch_x is the rows of training instances, batch_y is the corresponding labels.
        self.batch_num+=1
        all_leaf_codes=np.array([self.tree.item_id_leaf_code[label.item()] for label in batch_y],dtype=np.int64)
        loss=self.generate_training_instances(batch_x,all_leaf_codes)#column major
        loss.backward()#compute the gradient
        self.optimizer.step()# update the parameters
        self.optimizer.zero_grad()# clean the gradient
        self.update_learning_rate(self.batch_num)
        return loss
    def use_all_negative(self,batch_x, all_leaf_codes):  # all_codes is a array
        m,d=batch_x.shape
        sample_labels=all_negative_sampling(batch_x,self.tree,self.jtm_model,all_leaf_codes,self.N,self.device,
                                                   temperature=self.temperature)
        start_col,loss=0,0.0
        effective_item_index_s= batch_x >= 0
        effective_items=batch_x[effective_item_index_s]
        real_item_ids = torch.full((m, d), -1, dtype=torch.int64)

        for p_layer in range(self.tree.max_layer_id):
            layer_node_num=self.tree.layer_node_num_list[p_layer+1]
            all_labels=sample_labels[:,start_col:start_col+layer_node_num]

            real_item_ids[:,:]=-1
            real_item_ids[effective_item_index_s]=self.tree.item_id_node_ancestor_id[effective_items,p_layer+1]

            o_pi= self.jtm_model.preference(real_item_ids.to(self.device).repeat(1,layer_node_num).view(-1,d),\
                                                              all_labels.view(-1,1))[:, 0].view(m,layer_node_num)
            loss += (torch.logsumexp(o_pi,dim=1)-o_pi[:,0]).mean(-1)
            start_col+=layer_node_num
        return loss
    def generate_training_instances(self,batch_x, all_leaf_codes):  # all_codes is a array
        m,d=batch_x.shape
        if self.sample_method=='softmax':
            samples,log_q_matrix=softmax_sampling(batch_x,self.tree,self.jtm_model,all_leaf_codes,self.N,self.device,
                                                   temperature=self.temperature,itme_node_share_embedding=True)
        elif self.sample_method=='uniform_multiclass':
            samples,log_q_matrix=uniform_sampling_multiclassifcation(batch_x,self.tree,self.jtm_model,all_leaf_codes,
                                                                     self.N,self.device,temperature=self.temperature)
        elif self.sample_method=='top_down':
            samples,log_q_matrix=top_down_sample(batch_x,self.tree,self.jtm_model,all_leaf_codes,self.N,self.device,
                                                   temperature=self.temperature,itme_node_share_embedding=True)
        elif self.sample_method=='all_negative_sampling':
            return self.use_all_negative(batch_x, all_leaf_codes)
        else:
            assert False,'{} sampling is an invalid choice'.format(self.sample_method)

        sample_num=samples.shape[-1]
        label_id_layers=torch.arange(1,self.tree.max_layer_id+1).view(-1,1).repeat(m,self.N+1).view(m,sample_num)

        reall_item_ids = torch.full((m*sample_num,d), -1, dtype=torch.int64)
        effective_item_index = (batch_x >= 0).repeat(1,sample_num).view(-1,d)
        effective_items=batch_x.repeat(1,sample_num).view(-1,d)[effective_item_index]
        effeitive_item_layers=label_id_layers.view(-1,1).expand(samples.numel(),d)[effective_item_index]
        reall_item_ids[effective_item_index]  = self.tree.item_id_node_ancestor_id[effective_items,effeitive_item_layers]

        effective_index=samples>=0
        training_labels = samples[effective_index].view(-1, 1)
        user_index=torch.arange(samples.numel()).view(samples.shape)
        o_pi=torch.full(samples.shape,-1e9,device=self.device,dtype=torch.float32)
        o_pi[effective_index]=\
            self.jtm_model.preference(reall_item_ids[user_index[effective_index]].to(self.device),\
                                      training_labels)[:,0]-log_q_matrix[effective_index]

        return (torch.logsumexp(o_pi.view(m*self.tree.max_layer_id,self.N+1),dim=1)-o_pi.view(-1,self.N+1)[:,0]).mean(-1)


    def update_pi(self,d=7,discriminator=None):
        pi_new=self.tree_learner.tree_learning(d,self.tree,self.jtm_model,discriminator=discriminator)
        ids,leaf_codes=[],[]
        for item_id,leaf_code in pi_new.items():
            ids.append(item_id)
            leaf_codes.append(leaf_code)
        ids=np.array(ids)
        codes=np.array(leaf_codes)

        self.tree=Tree(ids,codes)
        self.tree_list.append(self.tree)

        # sort by codes
        codes.sort(-1)
        self.initial_codes.sort(-1)
        assert (codes-self.initial_codes).sum()==0


        del self.jtm_model
        if self.device !='cpu':
            torch.cuda.empty_cache()

        self.jtm_model=TDMModel(embed_dim=self.embed_dim,feature_groups=self.feature_groups,item_num=self.item_num,
                                node_num=self.tree.node_num,item_node_sharing_embedding=True).to(self.device)
        self.model_list.append(self.jtm_model)

        self.optimizer = self.opti(self.jtm_model.parameters())
        self.batch_num=0

    def predict(self,user_x,N,topk):
        candidate_codes = {0}
        effective_index = user_x >= 0
        effective_items = user_x[effective_index]
        result_set_A = set()  # result_set_A contains leaf node codes
        while len(candidate_codes) > 0:
            leaf_nodes = {code for code in candidate_codes if int(math.log(code + 1, 2)) == self.tree.max_layer_id}
            result_set_A |= leaf_nodes  # result_set_A contains leaf node codes
            candidate_codes -= leaf_nodes
            if len(candidate_codes) <= 0: break
            test_labels = torch.LongTensor([self.tree.node_code_node_id[code] for code in candidate_codes]).view(-1, 1)
            label_layer_ids = torch.LongTensor([int(math.log(code + 1, 2)) for code in candidate_codes]).view(-1, 1)
            test_codes = torch.LongTensor([code for code in candidate_codes])

            user_x_mat = torch.full((len(candidate_codes), user_x.shape[-1]), -1, dtype=torch.int64)
            user_x_mat[effective_index.expand(user_x_mat.shape)] = \
                self.tree.item_id_node_ancestor_id[effective_items.repeat(len(candidate_codes)),
                                              label_layer_ids.expand(user_x_mat.shape)[
                                                  effective_index.expand(user_x_mat.shape)]]
            with torch.no_grad():
                log_probs = self.jtm_model.preference(user_x_mat.to(self.device), test_labels.to(self.device))[:, 0]
            _, index = log_probs.sort(dim=-1)
            selected_codes = test_codes.gather(index=index[-N:].cpu(), dim=-1)
            candidate_codes.clear()
            for code in selected_codes:
                c = int(code.item())
                if 2 * c + 1 in self.tree.node_code_node_id:
                    candidate_codes.add(2 * c + 1)
                if 2 * c + 2 in self.tree.node_code_node_id:
                    candidate_codes.add(2 * c + 2)
        result_codes = torch.tensor([code for code in result_set_A]).view(-1)
        result_labels=torch.LongTensor([self.tree.node_code_node_id[code] for code in result_set_A])
        user_x_mat=torch.full(user_x.shape,-1,dtype=torch.int64)
        user_x_mat[effective_index]=self.tree.item_id_node_ancestor_id[effective_items,self.tree.max_layer_id]
        with torch.no_grad():
            log_probs = self.jtm_model.preference(user_x_mat.to(self.device).expand(len(result_set_A), user_x.shape[-1]),
                                                 result_labels.to(self.device).view(-1, 1))[:, 0]
        _, index = log_probs.sort(dim=-1)
        return [self.tree.leaf_code_item_id[code] for code in\
                result_codes.gather(index=index.cpu()[-topk:],dim=-1).numpy()]

