from __future__ import print_function
import numpy as np
import torch
from .Network_Model import Network
from .Cluster_Faiss import Cluster_Faiss
import math
from .Tree_Model import Tree
import copy
from .tree_learning import Tree_Learner
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
                 N=40,
                 negative_num=100,
                 item_node_share_embedding=True,
                 tree_learner_mode='jtm',
                 device='cuda'):

        self.device=device
        self.embed_dim=embed_dim
        self.feature_groups=feature_groups
        self.opti=optimizer
        self.parall=parall
        self.item_num=len(ids)
        self.feature_num=sum(feature_groups)
        self.item_node_share_embedding=item_node_share_embedding
        self.tree_learner_mode=tree_learner_mode
        self.initial_codes=codes
        self.tree=Tree(ids,codes)
        self.tree_list=[self.tree]
        self.N=N

        if negative_num is not None:
            self.change_negative_num=False
            self.negative_num = negative_num
        else:
            self.negative_num=0
            self.change_negative_num = True
            for layer in range(1,self.tree.max_layer_id+1):
                if self.tree.layer_node_num_list[layer]<=N+1:
                    self.negative_num+=(self.tree.layer_node_num_list[layer]-1)
                else:
                    self.negative_num+=N

        #self.coefficient=1.0*(self.tree.node_num-self.tree.max_layer_id)/self.negative_num
        #model part
        self.network_model=Network(embed_dim=embed_dim,feature_groups=feature_groups,item_num=self.item_num,
                                node_num=self.tree.node_num,
                                item_node_share_embedding=self.item_node_share_embedding).to(self.device)
        #self.model_list=[self.network_model]

        #optimizer
        self.optimizer = optimizer(self.network_model.parameters())

        if self.tree_learner_mode=='jtm':
            self.tree_learner=Tree_Learner(all_training_instance,item_user_pair_dict)
        elif self.tree_learner_mode=='tdm':
            self.cluster=Cluster_Faiss(parall=parall)
        else:
            assert False,'invalid tree learner mode : {}'.format(self.tree_learner_mode)
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

    def update_network_model(self,batch_x,batch_y):#batch_x is the rows of training instances, batch_y is the corresponding labels.
        self.batch_num+=1
        batch_size,feature_num=batch_x.shape
        batch_user=batch_x.to(self.device)
        all_leaf_codes=torch.LongTensor([self.tree.item_id_leaf_code[label.item()] for label in batch_y]).to(self.device).view(-1,1)
        training_labels, flag_matrix = self.generate_training_instances(all_leaf_codes)  # row major
        #batch_user_id=torch.arange(batch_size,device=self.device).view(-1,1).expand(training_labels.shape)
        if self.item_node_share_embedding:
            new_batch_user=batch_user.repeat(1,self.tree.max_layer_id+self.negative_num).view(-1,feature_num)
            label_layer_id=self.tree.node_id_layer_id[training_labels.view(-1,1)]#column major

            effective_item_index=new_batch_user>=0
            new_batch_user[effective_item_index]=\
                self.tree.item_id_node_ancestor_id[new_batch_user[effective_item_index],\
                                                   label_layer_id.expand(new_batch_user.shape)[effective_item_index]]
            log_probs=self.network_model.preference(new_batch_user,training_labels.view(-1,1))
            loss = (-(log_probs[:, 0:1] * flag_matrix.view(-1,1)).sum() - \
                    (log_probs[:, 1:2] * (1.0 - flag_matrix.view(-1,1))).sum()) / flag_matrix.numel()
        else:
            log_probs = self.network_model.preference(
                batch_x.to(self.device).repeat(self.tree.max_layer_id + self.negative_num,1), training_labels)
            loss = (-(log_probs[:, 0:1] * flag_matrix).sum() - (log_probs[:, 1:2] * (1.0 - flag_matrix)).sum()) / len(
                flag_matrix)

        loss.backward()#compute the gradient
        self.optimizer.step()# update the parameters
        self.optimizer.zero_grad()# clean the gradient
        self.update_learning_rate(self.batch_num)
        return loss

    def generate_training_instances(self, all_leaf_codes):  # all_codes is a tensor with 1 column
        label_size = len(all_leaf_codes)
        sample_label_matrix = torch.zeros((label_size, self.tree.max_layer_id+self.negative_num),\
                                          dtype=torch.int64,device=self.device)
        flag_matrix = torch.zeros(sample_label_matrix.shape, dtype=torch.float32,device=self.device)

        #########negative samples################
        sample_label_matrix[:, self.tree.max_layer_id:] = \
            torch.randint(low=1, high=self.tree.node_num, size=(label_size, self.negative_num),device=self.device)

        flag_matrix[:,:self.tree.max_layer_id]=1.0
        codes =all_leaf_codes
        for layer_id in range(self.tree.max_layer_id,0,-1):# list(range(self.tree.max_layer_id))[:0:-1]:
            code_labels=self.tree.node_code_node_id_array[codes]
            # positive sample
            sample_label_matrix[:, layer_id-1:layer_id] = code_labels

            flag_matrix[:,self.tree.max_layer_id:][code_labels==sample_label_matrix[:, self.tree.max_layer_id:]] =1.0
            codes = (codes-1)//2
        return sample_label_matrix,flag_matrix

    def update_tree(self,d=7,discriminator=None):
        if self.tree_learner_mode=='jtm':
            pi_new=self.tree_learner.tree_learning(d,self.tree,self.network_model)
            ids,codes=[],[]
            for item_id, leaf_code in pi_new.items():
                ids.append(item_id)
                codes.append(leaf_code)
            self.tree = Tree(ids, codes)
            self.new_network_model = Network(embed_dim=self.embed_dim, feature_groups=self.feature_groups,
                                         item_num=self.item_num,
                                         node_num=self.tree.node_num,
                                         item_node_share_embedding=self.item_node_share_embedding).to(self.device)
            self.new_network_model.load_state_dict(self.network_model.state_dict())
            self.optimizer = self.opti(self.new_network_model.parameters())
        elif self.tree_learner_mode=='tdm':
            if self.item_node_share_embedding:
                item_id_leaf_codes=[self.tree.item_id_leaf_code[item_id] for item_id in range(self.item_num)]
                leaf_node_id=[self.tree.node_code_node_id[code] for code in item_id_leaf_codes]
                cluster_data=self.network_model.node_embedding.weight.data[leaf_node_id].cpu().numpy()
            else:
                cluster_data = self.network_model.item_embedding.weight.data.cpu().numpy()[0:self.item_num, :]
            item_num,embed_dim=cluster_data.shape
            assert item_num==self.item_num and embed_dim==self.embed_dim
            print('cluster data shape {},{}'.format(*cluster_data.shape))
            ids, codes = self.cluster.train(ids=np.arange(self.item_num), data=cluster_data)  # return ids, codes
            self.tree = Tree(ids, codes)
            self.new_network_model = Network(embed_dim=self.embed_dim, feature_groups=self.feature_groups,
                                         item_num=self.item_num,
                                         node_num=self.tree.node_num,
                                         item_node_share_embedding=self.item_node_share_embedding).to(self.device)
            #self.network_model.load_state_dict(self.model_list[-1].state_dict())
            self.optimizer = self.opti(self.new_network_model.parameters())
        else:
            assert False,'invalid tree learner mode : {}'.format(self.tree_learner_mode)
        self.tree_list.append(self.tree)
        #self.model_list.append(self.network_model)
        #del self.network_model
        del self.network_model
        if self.device !='cpu':
            torch.cuda.empty_cache()
        self.network_model=self.new_network_model
        self.batch_num=0


        if self.change_negative_num:
            self.negative_num = 0
            for layer in range(1,self.tree.max_layer_id+1):
                if self.tree.layer_node_num_list[layer]<=self.N+1:
                    self.negative_num+=(self.tree.layer_node_num_list[layer]-1)
                else:
                    self.negative_num+=self.N


    def predict_share_embedding(self,user_x,N,topk,discriminator=None,tree=None):
        if discriminator is None:
            discriminator=self.network_model
        if tree is None:
            tree=self.tree
        candidate_codes = {0}
        effective_index = user_x >= 0
        effective_items = user_x[effective_index]
        result_set_A = set()  # result_set_A contains leaf node codes
        while len(candidate_codes) > 0:
            leaf_nodes = {code for code in candidate_codes if int(math.log(code + 1, 2)) == tree.max_layer_id}
            result_set_A |= leaf_nodes  # result_set_A contains leaf node codes
            candidate_codes -= leaf_nodes
            if len(candidate_codes) <= 0: break
            test_labels = torch.LongTensor([tree.node_code_node_id[code] for code in candidate_codes]).view(-1, 1)
            label_layer_ids = torch.LongTensor([int(math.log(code + 1, 2)) for code in candidate_codes]).view(-1, 1)
            test_codes = torch.LongTensor([code for code in candidate_codes])

            user_x_mat = torch.full((len(candidate_codes), user_x.shape[-1]), -1, dtype=torch.int64,device=self.device)
            user_x_mat[effective_index.expand(user_x_mat.shape)] = \
                tree.item_id_node_ancestor_id[effective_items.repeat(len(candidate_codes)),
                                              label_layer_ids.expand(user_x_mat.shape)[
                                                  effective_index.expand(user_x_mat.shape)]]
            with torch.no_grad():
                log_probs = discriminator.preference(user_x_mat, test_labels.to(self.device))[:, 0]
            _, index = log_probs.sort(dim=-1)
            selected_codes = test_codes.gather(index=index[-N:].cpu(), dim=-1)
            candidate_codes.clear()
            for code in selected_codes:
                c = int(code.item())
                if 2 * c + 1 in tree.node_code_node_id:
                    candidate_codes.add(2 * c + 1)
                if 2 * c + 2 in tree.node_code_node_id:
                    candidate_codes.add(2 * c + 2)
        result_codes = torch.tensor([code for code in result_set_A]).view(-1)
        result_labels=torch.LongTensor([tree.node_code_node_id[code] for code in result_set_A])
        user_x_mat=torch.full(user_x.shape,-1,dtype=torch.int64,device=self.device)
        user_x_mat[effective_index]=tree.item_id_node_ancestor_id[effective_items,tree.max_layer_id]
        with torch.no_grad():
            log_probs = discriminator.preference(user_x_mat.expand(len(result_set_A), user_x.shape[-1]),
                                                 result_labels.to(self.device).view(-1, 1))[:, 0]
        sorted_values, index = log_probs.sort(dim=-1)
        #return [tree.leaf_code_item_id[code] for code in\
        #        result_codes.gather(index=index.cpu()[-topk:],dim=-1).numpy()]
        return [(code, tree.leaf_code_item_id[code]) for v, code in zip(sorted_values.cpu().numpy(),
                                                                     result_codes.gather(index=index.cpu(),
                                                                                     dim=-1).numpy())]
                                                                                     
    def predict_share_embedding_parallel(self,user_ids,N,topk,discriminator=None,tree=None):
        ##user_ids is [bs,1]
        batch_size=user_ids.shape[0]
        user_x=user_ids.to(self.device)
        
        effective_user_index=user_x>=0
        effective_items=user_x[effective_user_index] 
        if discriminator is None:
            discriminator=self.network_model
        if tree is None:
            tree=self.tree
        
        selected_index=torch.arange(batch_size,device=self.device).view(-1,1).expand(batch_size,2*N)

        parent_code=torch.zeros((batch_size,N),device=self.device,dtype=torch.int64)
        candidate_codes = torch.zeros((batch_size,2*N),dtype=torch.int64,device=self.device)
        preference_mat=torch.full_like(candidate_codes,-1.0e9,device=self.device,dtype=torch.float32)
        
        layer_id=0
        assert tree.layer_node_num_list[0]==1 and len(tree.layer_node_num_list)==tree.max_layer_id+1
        while layer_id<tree.max_layer_id:
            
            p_code_num=min([tree.layer_node_num_list[layer_id],N])
            
            double_p_code=2*parent_code[:,0:p_code_num]
            candidate_codes[:,0:2*p_code_num:2]=double_p_code+1
            candidate_codes[:,1:2*p_code_num:2]=double_p_code+2

            all_labels =tree.node_code_node_id_array[candidate_codes[:,:2*p_code_num]].to(self.device)
            effective_index=all_labels>=0

            #test_codes = torch.LongTensor([code for code in candidate_codes])
            user_x[effective_user_index]=tree.item_id_node_ancestor_id[effective_items,layer_id+1]

            preference_mat[:,:]=-1.0e9
            with torch.no_grad():
                preference_mat[:,:2*p_code_num][effective_index] =\
                    torch.exp(discriminator.preference(user_x[selected_index[:,:2*p_code_num][effective_index]],\
                                                                            all_labels[effective_index].view(-1,1))[:, 0])

            index = preference_mat[:,:2*p_code_num].argsort(dim=-1)
            selected_p=min([N,tree.layer_node_num_list[layer_id+1]])
            parent_code = candidate_codes[:,:2*p_code_num].gather(index=index[:,-selected_p:], dim=-1)
            layer_id+=1
        result_code_matrix=candidate_codes[:,:2*p_code_num].gather(index=index[:,-topk:], dim=-1).cpu().numpy()
        result_items=np.zeros((batch_size,topk),dtype=np.int32)
        for i,r_code in enumerate(result_code_matrix):
            result_items[i:i+1,:]=np.array([tree.leaf_code_item_id[code] for code in r_code])
        return result_items

    def predict(self,user_x, N, topk,out_put_each_tree_result=True,forest=True):
        if not forest:
            if self.item_node_share_embedding:
                #result= self.predict_share_embedding(user_x, N, topk)
                #return [item for (_,item) in result[::-1][:topk]]
                return self.predict_share_embedding_parallel(user_x, N, topk)
            else:
                assert False,'no predict way!'
        #result = []
        each_tree_result = []
        all_result_items=set()
        for tree_id in range(len(self.tree_list)):
            discriminator=self.model_list[tree_id]

            discriminator.eval()
            re = self.predict_share_embedding(user_x, N, topk, \
                                              discriminator=discriminator,tree=self.tree_list[tree_id])
            discriminator.train()

            #result.extend(re)
            each_tree_re = [item_id for (_, item_id) in re[-topk:]]
            each_tree_result.append(each_tree_re)

            #all_result_codes.update([code for (code, _) in re])
            all_result_items.update([item_id for (_, item_id) in re])
            # network_model.train()
        discriminator=self.network_model
        discriminator.eval()
        result_codes=torch.LongTensor([self.tree.item_id_leaf_code[item_id] for item_id in all_result_items])
        with torch.no_grad():
            effective_index = user_x >= 0
            effective_items = user_x[effective_index]
            user_x_mat = torch.full(user_x.shape, -1, dtype=torch.int64,device=self.device)
            user_x_mat[effective_index] = self.tree.item_id_node_ancestor_id[effective_items, self.tree.max_layer_id]
            result_labels = torch.LongTensor([self.tree.node_code_node_id[code.item()] for code in result_codes])
            log_probs = discriminator.preference(user_x_mat.expand(len(result_codes), user_x.shape[-1]),
                                                 result_labels.to(self.device).view(-1, 1))[:, 0]
        _, index = log_probs.sort(dim=-1)
        selected_codes=result_codes.gather(index=index[-topk:].cpu(),dim=-1).tolist()
        selected_items=[self.tree.leaf_code_item_id[code] for code in selected_codes]
        discriminator.train()
        if out_put_each_tree_result:
            return selected_items, each_tree_result
        else:
            return selected_items










