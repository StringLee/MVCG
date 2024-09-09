import argparse
import wandb

import numpy as np
import torch
import torch.nn as nn
from torch import optim as optim

import dgl
from sklearn import metrics
from libs.my_munkres import Munkres
from sklearn.cluster import KMeans

from scipy.optimize import linear_sum_assignment


def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--missing_rate", type=float, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--max_epoch", type=int, required=True,
                        help="number of training epochs")

    parser.add_argument("--D_para", type=float, required=True)
    parser.add_argument("--E_para", type=float, required=True)
    parser.add_argument("--decoder_AS_type", type=str, required=True, choices=["cat", "mean"])
    parser.add_argument("--loss_E_S_para", type=float, required=True)
    parser.add_argument("--loss_E_A_para", type=float, required=True)
    parser.add_argument("--loss_E_Z_para", type=float, required=True)
    parser.add_argument("--loss_D_S_para", type=float, required=True)
    parser.add_argument("--loss_D_A_para", type=float, required=True)

    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="weight decay")
    parser.add_argument("--optimizer", type=str, default="adam")

    parser.add_argument("--num_heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num_hidden", type=int, default=512,
                        help="number of hidden units")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--attn_drop", type=float, default=.1,
                        help="attention dropout")
    parser.add_argument("--in_drop", type=float, default=.2,
                        help="input feature dropout")
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument("--encoder", type=str, default="gat")
    parser.add_argument("--decoder", type=str, default="mlp")
    parser.add_argument("--activation", type=str, default="prelu")
    parser.add_argument("--num_dec_layers", type=int, default=1)

    args = parser.parse_args()
    return args


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "selu":
        return nn.SELU()
    elif name == "elu":
        return nn.ELU()
    elif name == "silu":
        return nn.SiLU()
    elif name is None:
        return nn.Identity()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def identity_norm(x):
    def func(x):
        return x

    return func


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "identity":
        return identity_norm
    else:
        return None


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()
    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]

    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        raise NotImplementedError("Invalid optimizer")

    return optimizer


# -------------------
def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.ones(E) * mask_prob
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def load_missing_graph_dataset(dataset_name, missing_rate=0.6, show_details=True):
    load_path = "data/" + dataset_name + "/" + dataset_name
    feat = np.load(f'{load_path}_feat_missing_{missing_rate}.npy', allow_pickle=True)
    random_values = np.random.rand(*feat.shape) * 1e-6 + 1e-6
    feat[feat == -1] = random_values[feat == -1]
    zero_rows = np.all(feat == 0, axis=1)
    feat[zero_rows] = random_values[zero_rows]
    # feat[feat == -1] = 1e-6
    label = np.load(load_path + "_label.npy", allow_pickle=True)
    adj = np.load(load_path + "_adj.npy", allow_pickle=True)
    for i in range(adj.shape[0]):
        if np.all(adj[i, :] == 0):
            adj[i, i] = 1
    missing_index = np.load(f'{load_path}_index_missing_{missing_rate}.npy')
    cluster_num = len(np.unique(label))
    node_num = feat.shape[0]
    if show_details:
        print("++++++++++++++++++++++++++++++")
        print("---details of graph dataset---")
        print("++++++++++++++++++++++++++++++")
        print("dataset name:   ", dataset_name)
        print("missing rate:   ", missing_rate)
        print("feature shape:  ", feat.shape)
        print("label shape:    ", label.shape)
        print("adj shape:      ", adj.shape)
        print("undirected edge num:   ", int(np.nonzero(adj)[0].shape[0] / 2))
        print("category num:          ", max(label) - min(label) + 1)
        print("category distribution: ")
        for i in range(max(label) + 1):
            print("label", i, end=":")
            print(len(label[np.where(label == i)]))
        print("++++++++++++++++++++++++++++++")

    similarity = get_similarity_mix_adj(feat, missing_index, adj, K=5)

    feat_tensor = torch.tensor(feat, dtype=torch.float32)
    similarity_tensor = torch.tensor(similarity, dtype=torch.int64)
    adj_tensor = torch.tensor(adj, dtype=torch.int64)

    label_tensor = torch.tensor(label, dtype=torch.int64)

    src_adj, dst_adj = adj_tensor.nonzero(as_tuple=True)
    src_similarity, dst_similarity = similarity_tensor.nonzero(as_tuple=True)

    graph_adj = dgl.graph((src_adj, dst_adj))
    graph_similarity = dgl.graph((src_similarity, dst_similarity))

    graph_adj.ndata['feat'] = feat_tensor
    graph_adj.ndata['label'] = label_tensor
    graph_similarity.ndata['feat'] = feat_tensor
    graph_similarity.ndata['label'] = label_tensor


    graph_adj = dgl.add_self_loop(graph_adj)
    return graph_adj, graph_similarity, missing_index, (feat.shape[1], cluster_num)


def get_similarity_mix_adj(feat, missing_index, adj, K):
    feat_norm = np.linalg.norm(feat, axis=1, keepdims=True)
    feat_normalized = feat / feat_norm

    similarity_matrix = np.dot(feat_normalized, feat_normalized.T)

    noise = np.random.normal(0, 1e-6, similarity_matrix.shape)
    similarity_matrix += noise

    topk_indices = np.argsort(-similarity_matrix, axis=1)[:, :K + 1]

    knn_matrix = np.zeros_like(similarity_matrix)

    row_indices = np.arange(similarity_matrix.shape[0])[:, np.newaxis]
    knn_matrix[row_indices, topk_indices] = 1

    result_matrix = adj.copy()

    for idx in missing_index:
        result_matrix[idx] = knn_matrix[idx]

    for i in range(result_matrix.shape[0]):
        if np.sum(result_matrix[i, :]) == 0:
            max_sim_idx = np.argmax(similarity_matrix[i, :])
            result_matrix[i, max_sim_idx] = 1

    for j in range(result_matrix.shape[1]):
        if np.sum(result_matrix[:, j]) == 0:
            max_sim_idx = np.argmax(similarity_matrix[:, j])
            result_matrix[max_sim_idx, j] = 1

    return result_matrix


def cluster_probing_full_batch(model, graph, x, device):
    model.eval()
    with torch.no_grad():
        x = model.embed(graph.to(device), x.to(device))

    labels = graph.ndata["label"]

    results_str = clustering(x.detach().cpu().numpy(), labels)

    return results_str


def clustering(embeds, labels):
    # labels = torch.from_numpy(labels).type(torch.LongTensor)
    num_classes = torch.max(labels).item() + 1

    # print('===================================  KMeans Clustering.  ========================================')
    # u, s, v = sp.linalg.svds(embeds, k=num_classes, which='LM')
    # predY = KMeans(n_clusters=num_classes).fit(u).labels_

    accs = []
    nmis = []
    aris = []
    f1s = []
    for i in range(10):
        # best_loss = 1e9
        best_acc = 0
        best_f1 = 0
        best_nmi = 0
        best_ari = 0
        for j in range(10):
            # set_trace()
            predY = KMeans(n_clusters=num_classes, n_init='auto').fit(embeds).labels_
            gnd_Y = bestMap(predY, labels.cpu().detach().numpy())
            # predY_temp = torch.tensor(predY, dtype=torch.float)
            # gnd_Y_temp = torch.tensor(gnd_Y)
            # loss = nn.MSELoss()(predY_temp, gnd_Y_temp)
            # if loss <= best_loss:
            #     best_loss = loss
            #     acc_temp, f1_temp, nmi_temp, ari_temp, _ = clustering_metrics(gnd_Y, predY)

            acc_temp, f1_temp, nmi_temp, ari_temp, _ = clustering_metrics(gnd_Y, predY)
            if acc_temp > best_acc:
                best_acc = acc_temp
                best_f1 = f1_temp
                best_nmi = nmi_temp
                best_ari = ari_temp

        accs.append(best_acc)
        nmis.append(best_nmi)
        aris.append(best_ari)
        f1s.append(best_f1)

        # accs.append(acc_temp)
        # nmis.append(nmi_temp)
        # aris.append(ari_temp)
        # f1s.append(f1_temp)

    accs = np.stack(accs)
    nmis = np.stack(nmis)
    aris = np.stack(aris)
    f1s = np.stack(f1s)

    results = {"acc": accs.mean().item(), "nmi": nmis.mean().item(), "ari": aris.mean().item(), "f1": f1s.mean().item()}
    print(results)
    wandb.log(results)

    return results


def bestMap(L1, L2):
    '''
    bestmap: permute labels of L2 to match L1 as good as possible
        INPUT:
            L1: labels of L1, shape of (N,) vector
            L2: labels of L2, shape of (N,) vector
        OUTPUT:
            new_L2: best matched permuted L2, shape of (N,) vector
    version 1.0 --December/2018
    Modified from bestMap.m (written by Deng Cai)
    '''
    if L1.shape[0] != L2.shape[0] or len(L1.shape) > 1 or len(L2.shape) > 1:
        raise Exception('L1 shape must equal L2 shape')
        return
    Label1 = np.unique(L1)
    nClass1 = Label1.shape[0]
    Label2 = np.unique(L2)
    nClass2 = Label2.shape[0]
    nClass = max(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        for j in range(nClass2):
            G[j, i] = np.sum((np.logical_and(L1 == Label1[i], L2 == Label2[j])).astype(np.int64))
    c, t = linear_sum_assignment(-G)
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[t[i]]
    return newL2


def clustering_metrics(true_label, pred_label):
    l1 = list(set(true_label))
    numclass1 = len(l1)

    l2 = list(set(pred_label))
    numclass2 = len(l2)
    if numclass1 != numclass2:
        print('Class Not equal, Error!!!!')
        return 0, 0, 0, 0, 0

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(true_label) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if pred_label[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()

    indexes = m.compute(cost)
    idx = indexes[2][1]
    # get the match results
    new_predict = np.zeros(len(pred_label))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]
        # ai is the index with label==c2 in the predict list
        ai = [ind for ind, elm in enumerate(pred_label) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(true_label, new_predict)
    f1_macro = metrics.f1_score(true_label, new_predict, average='macro')
    nmi = metrics.normalized_mutual_info_score(true_label, pred_label)
    ari = metrics.adjusted_rand_score(true_label, pred_label)

    return acc * 100, f1_macro * 100, nmi * 100, ari * 100, idx
