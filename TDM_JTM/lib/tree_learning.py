
import numpy as np
import torch
import math
class Tree_Learner(object):
    def __init__(self,all_training_instance,item_user_pair_dict):
        self.training_instances=all_training_instance

        self.item_user_pair_dict=item_user_pair_dict


    def tree_learning(self,d, tree,network_model,discriminator=None):
        """ The overall tree learning algorithm (Algorithm 2 in the paper)

        Returns:
            the leant new projection from item to leaf node (\pi_{new})

        Args:
            d (int, required): the tree learning level gap
            tree (tree, required): the old tree (\pi_{old})
        """
        l_max = tree.max_layer_id
        l = d

        pi_new = dict()#pi: item_id->leaf_node_code

        # \pi_{new} <- \pi_{old} pi(item:code)
        #assign all items to the root node
        for item_id,leaf_code in tree.item_id_leaf_code.items():
            pi_new[item_id] = tree.get_ancestor(leaf_code, l - d)#item_code's ancesotr code at level l-d,

        while d > 0:
            print('process level {}'.format(l))
            nodes = tree.get_nodes_given_level(l - d)#return the node codes of the give level
            for ni in nodes:
                C_ni = self.get_itemset_given_ancestor(pi_new, ni)#return item id list of the corresponding code ni
                pi_star =self.assign_parent(network_model,l_max, l, ni, C_ni, tree)#C_ni is the item list that belong to the code node ni

                # update pi_new according to the found optimal pi_star
                for item_id, node_code in pi_star.items():
                    pi_new[item_id] = node_code

            d = min(d, l_max - l)
            l = l + d
        code_min=2**l_max-1
        code_max=2**(l_max+1)-1
        print('min code {}, max code {}'.format(code_min,code_max))
        assert len(pi_new)==len(tree.item_id_leaf_code)
        #print(pi_new)
        for item_id,leaf_code in pi_new.items():
            assert leaf_code<code_max and leaf_code>=code_min,'item_id is {},leaf code is {}'.format(item_id,leaf_code)
            #tree.item_id_leaf_code[item_id]=leaf_code
            #tree.leaf_code_item_id[leaf_code]=item_id
        #tree.generate_item_id_ancestor_node_id()
        return pi_new


    def get_itemset_given_ancestor(self,pi_new, node):#node is the code
        res = []
        for ci, code in pi_new.items():#ci is item id
            if code == node:
                res.append(ci)
        return res

    def get_weights(self,network_model,C_ni, ni, children_of_ni_in_level_l, tree):
        """use the user preference prediction model to calculate the required weights

        Returns:
            all weights

        Args:
            C_ni (item, required): item set whose ancestor is the non-leaf node ni
            ni (node, required): a non-leaf node in level l-d
            children_of_ni_in_level_l (list, required): the level l-th children of ni
            tree (tree, required): the old tree (\pi_{old})
        """
        edge_weights = dict()
        network_model.eval()
        device=str(next(iter(network_model.state_dict().values())).device)
        for ck in C_ni:#ck is item_id
            if ck not in self.item_user_pair_dict:
                edge_weights[ck] = list()
                edge_weights[ck].append([node_code for node_code in children_of_ni_in_level_l])  # the first element is the list of nodes in level l
                edge_weights[ck].append([-1.0e9 for _ in children_of_ni_in_level_l])  # the second element is the list of corresponding weights
                continue
            edge_weights[ck] = list()
            edge_weights[ck].append([]) # the first element is the list of nodes in level l
            edge_weights[ck].append([]) # the second element is the list of corresponding weights

            sample_set_index = self.item_user_pair_dict[ck]
            Ak = self.training_instances[sample_set_index]#Ak contains user matrix
            instance_num=len(sample_set_index)
            effective_Ak_index =Ak>=0
            row_index=Ak[effective_Ak_index]#Ak contains item id
            start_layer=int(math.log(ni+1.0,2))
            end_layer=int(math.log(children_of_ni_in_level_l[0]+1.0,2))
            user_behaviour_history=\
                torch.full((instance_num*(end_layer-start_layer),Ak.shape[1]),-1,dtype=torch.int64,device=device)
            for layer in range(end_layer,start_layer,-1):
                ind=end_layer-layer
                #col_index=torch.LongTensor([layer]*len(row_index))
                user_behaviour_history[ind*instance_num:(ind+1)*instance_num,:][effective_Ak_index]=\
                                                                tree.item_id_node_ancestor_id[row_index,layer]

            for node in children_of_ni_in_level_l:
                path_to_ni = tree.get_parent_path(node, ni)
                assert len(path_to_ni)==end_layer-start_layer

                labels = torch.LongTensor([tree.node_code_node_id[code] for code in path_to_ni]).view(-1,1).\
                                                                                    repeat(1,instance_num).view(-1,1).to(device)

                #weight=self.calculate_weight_use_prediction_model(network_model,user_behaviour_history,labels,device)
                assert user_behaviour_history.shape[0]==labels.shape[0]
                total_size=labels.shape[0]
                batch_size=500
                if total_size%batch_size==0:
                    batch_num=total_size//batch_size
                else:
                    batch_num=total_size//batch_size+1
                log_prob=torch.full((total_size,),0,dtype=torch.float32)
                with torch.no_grad():
                    for tt in range(batch_num):
                        start_i=tt*batch_size
                        end_i=min([total_size,(tt+1)*batch_size])
                        log_prob[start_i:end_i] = network_model.preference(user_behaviour_history[start_i:end_i,:], labels[start_i:end_i,:])[:, 0].cpu()
                weight= log_prob.sum().item()#/len(labels)

                edge_weights[ck][0].append(node)
                edge_weights[ck][1].append(weight)
        network_model.train()
        return edge_weights

    def assign_parent(self,network_model,l_max, l, ni, C_ni, tree):#C_ni is the item list that belong to the code node ni
        """implementation of line 5 of Algorithm 2
        Returns:
            updated \pi_{new}

        Args:
            l_max (int, required): the max level of the tree
            l (int, required): current assign level
            d (int, required): level gap in tree_learning
            ni (node, required): a non-leaf node in level l-d
            C_ni (item, required): item set whose ancestor is the non-leaf node ni
            tree (tree, required): the old tree (\pi_{old})
        """
        # get the children of ni in level l
        children_of_ni_in_level_l = tree.get_children_given_ancestor_and_level(ni, l)#return descendant code of ni at level l

        # get all the required weights
        edge_weights = self.get_weights(network_model,C_ni, ni, children_of_ni_in_level_l, tree) #return a dict that key is c_i\in C_ni, items is (children_code,weight)

        # assign each item to the level l node with the maximum weight
        assign_dict = dict()
        for ci, info in edge_weights.items():# ci is one item w.r.t. leaf node whose ancestor is ni
            assign_candidate_nodes = info[0]#info[0] is node code list, info[1] is weight list
            assign_weights = np.array(info[1], dtype=np.float32)
            sorted_idx = np.argsort(-assign_weights)#descent sort
            # assign item ci to the node with the largest weight
            max_weight_node = assign_candidate_nodes[sorted_idx[0]]
            if max_weight_node in assign_dict:
                assign_dict[max_weight_node].append((ci, sorted_idx, assign_candidate_nodes, assign_weights))
            else:
                assign_dict[max_weight_node] = [(ci, sorted_idx, assign_candidate_nodes, assign_weights)]

        edge_weights = None

        # get each item's original assignment of level l in tree, used in rebalance process
        origin_relation = dict()
        for ci in C_ni:
            code=tree.item_id_leaf_code[ci]
            origin_relation[ci] = tree.get_ancestor(code, l)#return the ancestor code at level l

        # rebalance
        #max_assign_num = int(math.pow(2, l_max - l))
        processed_set = set()# record the node which need to reduce some item
        while True:
            max_assign_cnt=-1
            max_assign_node = None

            for node in children_of_ni_in_level_l:
                if node in processed_set:
                    continue
                if node not in assign_dict:
                    continue
                if len(assign_dict[node]) > tree.maximum_assigned_item_num[node]:
                    if len(assign_dict[node])>max_assign_cnt:
                        max_assign_cnt = len(assign_dict[node])
                        max_assign_node = node

            if max_assign_node == None or max_assign_cnt<0:
                break

            # rebalance
            max_assign_num=tree.maximum_assigned_item_num[max_assign_node]
            processed_set.add(max_assign_node)# record the node code which need to reduce some item
            elements = assign_dict[max_assign_node]
            #elements.sort(key=lambda x: (int(max_assign_node != origin_relation[x[0]]), -x[1]))#sort by ???
            elements.sort(key=lambda x: (int(max_assign_node != origin_relation[x[0]]), -x[3][x[1][0]]))

            for e in elements[max_assign_num:]:
                has_assigned = False
                for idx in e[1]:
                    other_parent_node = e[2][idx]
                    if other_parent_node in processed_set:
                        continue
                    if other_parent_node not in assign_dict:#assign_dict is the
                        assign_dict[other_parent_node] = [(e[0], e[1], e[2], e[3])]
                    else:
                        assign_dict[other_parent_node].append((e[0], e[1], e[2], e[3]))
                    has_assigned=True
                    break
                if has_assigned==False:
                    print(max_assign_num)
                    print(max_assign_node)
                    print(processed_set)
                    print(elements)
                    print(children_of_ni_in_level_l)
                    print(C_ni)
                    assert has_assigned==True
            del elements[max_assign_num:]

        pi_new = dict()
        #max_assign_num = int(math.pow(2, l_max - l))
        for parent_code, value in assign_dict.items():
            assert len(value) == tree.maximum_assigned_item_num[parent_code]
            for e in value:
                assert e[0] not in pi_new#e contains (ci, sorted_idx, assign_candidate_nodes, assign_weights)
                pi_new[e[0]] = parent_code
        return pi_new


if __name__ == '__main__':
    pass
    '''
    tree_idx = 0
    old_tree_file = './cate_tree_%d/tree.pb' % tree_idx
    new_tree_file = './cate_tree_%d/tree.pb' % (tree_idx + 1)


    # load old tree
    tree = TreeLearner('old_tree')
    tree.load_tree(old_tree_file)

    # Algorithm 2: Tree learning algorithm
    d = 7
    pi_new = tree_learning(d, tree)

    # assign leaf nodes and save new tree
    tree.assign_leaf_nodes(pi_new, new_tree_file)
    '''
