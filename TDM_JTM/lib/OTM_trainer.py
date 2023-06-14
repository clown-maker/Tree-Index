from __future__ import print_function
import numpy as np
import torch
from .Network_Model import Network
from .Cluster_Faiss import Cluster_Faiss
import math
from .Tree_Model import Tree
import copy
from .tree_learning import Tree_Learner
class TrainOTM:
    def __init__(self,
                 ids,
                 codes,
                 all_training_instance=None,
                 item_user_pair_dict=None,
                 embed_dim=24,
                 feature_groups=[20,20,10,10,2,2,2,1,1,1],
                 optimizer=lambda params: torch.optim.Adam(params, lr=1e-3, amsgrad=True),
                 parall=3,
                 N=40,#also the beam size k
                 #negative_num=100,
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
        self.tree=Tree(ids,codes,device=self.device)
        self.tree_list=[self.tree]
        self.beam_size_k=N


        #self.coefficient=1.0*(self.tree.node_num-self.tree.max_layer_id)/self.negative_num
        #model part
        self.sample_num=sum([min([2*self.beam_size_k,self.tree.layer_node_num_list[layer]])\
                                                for layer in range(1,self.tree.max_layer_id+1)])


        self.network_model=Network(embed_dim=embed_dim,feature_groups=feature_groups,item_num=self.item_num,
                                node_num=self.tree.node_num,
                                item_node_share_embedding=self.item_node_share_embedding).to(self.device)
        self.model_list=[self.network_model]

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
        training_labels, flag_matrix =  self.generate_training_instances(batch_user,all_leaf_codes)  # row major
        batch_user_id=torch.arange(batch_size,device=self.device).view(-1,1).expand(training_labels.shape)
        if self.item_node_share_embedding:
            effective_label_index=training_labels>=0
            effective_labels=training_labels[effective_label_index].view(-1,1)
            effective_flag=flag_matrix[effective_label_index].view(-1,1)

            new_batch_user=batch_user[batch_user_id[effective_label_index]]
            label_layer_id=self.tree.node_id_layer_id[effective_labels]#column major

            effective_item_index=new_batch_user>=0
            item_ids=new_batch_user[effective_item_index]

            new_batch_user[effective_item_index]=\
                self.tree.item_id_node_ancestor_id[item_ids,label_layer_id.expand(new_batch_user.shape)[effective_item_index]]
            log_probs=self.network_model.preference(new_batch_user,effective_labels)

            loss = (-(log_probs[:, 0:1] * effective_flag).sum() - (log_probs[:, 1:2] * (1.0 - effective_flag)).sum()) / effective_flag.numel()
        else:
            log_probs = self.network_model.preference(
                batch_user.repeat(self.tree.max_layer_id + self.negative_num,1), training_labels)
            loss = (-(log_probs[:, 0:1] * flag_matrix).sum() - (log_probs[:, 1:2] * (1.0 - flag_matrix)).sum()) / len(
                flag_matrix)

        loss.backward()#compute the gradient
        self.optimizer.step()# update the parameters
        self.optimizer.zero_grad()# clean the gradient
        self.update_learning_rate(self.batch_num)
        return loss

    def generate_training_instances(self,batch_users,all_leaf_codes):  # all_codes is a tensor with 1 column
        batch_size,feature_num=batch_users.shape

        batch_user_id=torch.arange(batch_size,device=self.device).view(-1,1).expand(batch_size,2*self.beam_size_k)

        parent_code=torch.zeros((batch_size,1),dtype=torch.int64,device=self.device)
        candidate_matrix=torch.full((batch_size,self.sample_num),-10,dtype=torch.int64,device=self.device)
        flag=torch.full(candidate_matrix.shape,0,dtype=torch.float32,device=self.device)

        child_code=torch.full((batch_size,2*self.beam_size_k),-10,dtype=torch.int64,device=self.device)
        temp_preference=torch.full(child_code.shape,-1.0e9,device=self.device,dtype=torch.float32)
        start_col=0
        for layer_id in range(1,self.tree.max_layer_id+1):
            child_num=2*parent_code.shape[1]
            double_parent_code=2*parent_code
            child_code[:,0:child_num:2]=double_parent_code + 1
            child_code[:,1:child_num:2]=double_parent_code + 2
            candidate_matrix[:,start_col:start_col+child_num]=\
                                self.tree.node_code_node_id_array[child_code[:,0:child_num]]
            index = candidate_matrix[:, start_col:start_col + child_num] >= 0
            if child_num<=self.beam_size_k:
                parent_code=child_code[:,0:child_num][index].view(batch_size,-1)
            else:
                effective_node_ids=candidate_matrix[:, start_col:start_col+child_num][index].view(-1,1)
                temp_preference[:,:]=-1.0e9
                with torch.no_grad():
                    new_batch_users=batch_users[batch_user_id[:,0:child_num][index]]
                    effective_item_index = new_batch_users >= 0
                    item_ids = new_batch_users[effective_item_index]
                    new_batch_users[effective_item_index] = self.tree.item_id_node_ancestor_id[item_ids, layer_id]

                    temp_preference[:,0:child_num][index]=\
                        self.network_model.preference(new_batch_users,effective_node_ids)[:,0]

                sorted_index=temp_preference.argsort(dim=1)[:,-self.beam_size_k:]
                parent_code=child_code.gather(index=sorted_index,dim=1)
            start_col+=child_num

        ##assign the flag z_{n} for n\in B_{h}
        codes=all_leaf_codes#all_codes is a tensor with 1 column
        stop_po=torch.full((batch_size,1),True,dtype=torch.bool,device=self.device)
        end_col=self.sample_num
        for layer in range(self.tree.max_layer_id,0,-1):
            if stop_po.sum().item()<=0: break
            #start_col=(layer-1)*self.beam_size_k*2
            layer_sample_num=min([2*self.beam_size_k,self.tree.layer_node_num_list[layer]])
            if layer==self.tree.max_layer_id:
                flag[:,end_col-layer_sample_num:end_col]\
                    [candidate_matrix[:,end_col-layer_sample_num:end_col]==self.tree.node_code_node_id_array[codes]]=1.0
            else:
                child_node_ids=self.tree.node_code_node_id_array[child_code[:,-2:]]
                effective_index=child_node_ids>=0
                temp_preference[:,-2:]=-1.0e9
                with torch.no_grad():
                    new_batch_users=batch_users[batch_user_id[:,0:2][effective_index]]
                    effective_item_index = new_batch_users >= 0
                    item_ids = new_batch_users[effective_item_index]
                    new_batch_users[effective_item_index] = self.tree.item_id_node_ancestor_id[item_ids, layer]
                    temp_preference[:,-2:][effective_index]=\
                        self.network_model.preference(new_batch_users,child_node_ids[effective_index].view(-1,1))[:,0]
                selected_codes=torch.where(temp_preference[:,-2:-1]>temp_preference[:,-1:],\
                                                                child_code[:,-2:-1],child_code[:,-1:])
                stop_po = (codes==selected_codes)&stop_po
                codes=(codes-1)//2
                #use torch.scattar
                flag[:, end_col-layer_sample_num:end_col]\
                    [(candidate_matrix[:, end_col-layer_sample_num:end_col] ==
                     self.tree.node_code_node_id_array[codes])&stop_po]= 1.0
            even_odd_index=codes%2==0
            child_code[:,-2:-1]= torch.where(even_odd_index,codes-1,codes)
            child_code[:, -1:] = torch.where(even_odd_index,codes,codes+1)
            end_col-=layer_sample_num
        return candidate_matrix,flag



    def update_tree(self,d=7,discriminator=None):
        if self.tree_learner_mode=='jtm':
            pi_new=self.tree_learner.tree_learning(d,self.tree,self.network_model)
            '''
            ids,leaf_codes=[],[]
            for item_id,leaf_code in pi_new.items():
                ids.append(item_id)
                leaf_codes.append(leaf_code)
            ids=np.array(ids)
            codes=np.array(leaf_codes)
            # sort by codes
            codes.sort(-1)
            self.initial_codes.sort(-1)
            assert (codes-self.initial_codes).sum()==0
            '''
            new_tree=copy.deepcopy(self.tree)
            for item_id, leaf_code in pi_new.items():
                new_tree.item_id_leaf_code[item_id]=leaf_code
                new_tree.leaf_code_item_id[leaf_code]=item_id
            new_tree.generate_item_id_ancestor_node_id()
            self.tree=new_tree
            #new_network=copy.deepcopy(self.network_model)
            #self.network_model=new_network
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
        else:
            assert False,'invalid tree learner mode : {}'.format(self.tree_learner_mode)
        self.network_model = Network(embed_dim=self.embed_dim, feature_groups=self.feature_groups,
                                     item_num=self.item_num,
                                     node_num=self.tree.node_num,
                                     item_node_share_embedding=self.item_node_share_embedding).to(self.device)
        self.network_model.load_state_dict(self.model_list[-1].state_dict())
        self.optimizer = self.opti(self.network_model.parameters())
        self.tree_list.append(self.tree)
        self.model_list.append(self.network_model)
        #del self.network_model
        if self.device !='cpu':
            torch.cuda.empty_cache()
        self.batch_num=0
        self.sample_num=sum([min([2*self.beam_size_k,self.tree.layer_node_num_list[layer]])\
                                                for layer in range(1,self.tree.max_layer_id+1)])

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
    def predict(self,user_x, N, topk,out_put_each_tree_result=True,forest=True):
        if not forest:
            if self.item_node_share_embedding:
                return self.predict_share_embedding(user_x, N, topk)
            else:
                assert False,'no predict way!'
        #result = []
        each_tree_result = []
        all_result_codes=set()
        for tree_id in range(len(self.tree_list)):
            discriminator=self.model_list[tree_id]

            discriminator.eval()
            re = self.predict_share_embedding(user_x, N, topk, \
                                              discriminator=discriminator,tree=self.tree_list[tree_id])
            discriminator.train()

            #result.extend(re)
            each_tree_re = [item_id for (_, item_id) in re[-topk:]]
            each_tree_result.append(each_tree_re)

            all_result_codes.update([code for (code, _) in re])
            # network_model.train()
        discriminator=self.network_model
        discriminator.eval()
        with torch.no_grad():
            effective_index = user_x >= 0
            effective_items = user_x[effective_index]
            user_x_mat = torch.full(user_x.shape, -1, dtype=torch.int64,device=self.device)
            user_x_mat[effective_index] = self.tree.item_id_node_ancestor_id[effective_items, self.tree.max_layer_id]
            result_codes = torch.LongTensor(list(all_result_codes))
            result_labels = torch.LongTensor([self.tree.node_code_node_id[code.item()] for code in result_codes])
            log_probs = discriminator.preference(user_x_mat.expand(len(all_result_codes), user_x.shape[-1]),
                                                 result_labels.to(self.device).view(-1, 1))[:, 0]
        _, index = log_probs.sort(dim=-1)
        selected_codes=result_codes.gather(index=index[-topk:].cpu(),dim=-1).tolist()
        selected_items=[self.tree.leaf_code_item_id[code] for code in selected_codes]
        discriminator.train()
        if out_put_each_tree_result:
            return selected_items, each_tree_result
        else:
            return selected_items











