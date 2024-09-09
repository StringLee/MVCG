from typing import Optional
import torch
import torch.nn as nn
from .gat import GAT
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import torch_clustering


def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead,
                 nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type in ("gat", "tsgat"):
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "mlp":
        # * just for decoder 
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden * 2),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden * 2, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError

    return mod


class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            num_dec_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            encoder_type: str,
            decoder_type: str,
            decoder_AS_type: str,
            loss_E_S_para: float,
            loss_E_A_para: float,
            loss_E_Z_para: float,
            loss_D_A_para: float,
            loss_D_S_para: float,
    ):
        super(PreModel, self).__init__()

        self.decoder_AS_type = decoder_AS_type
        self.loss_E_S_para = loss_E_S_para
        self.loss_E_A_para = loss_E_A_para
        self.loss_E_Z_para = loss_E_Z_para
        self.loss_D_A_para = loss_D_A_para
        self.loss_D_S_para = loss_D_S_para

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat",):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead if decoder_type in ("gat",) else num_hidden

        # build encoder
        self.encoder_S = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        self.encoder_A = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        self.decoder_A = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim * 2,
            num_hidden=dec_num_hidden,
            out_dim=1,
            nhead_out=nhead_out,
            num_layers=num_dec_layers,
            nhead=nhead,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )

        self.decoder_S = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim * 2,
            num_hidden=dec_num_hidden,
            out_dim=1,
            nhead_out=nhead_out,
            num_layers=num_dec_layers,
            nhead=nhead,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )

        assert decoder_AS_type == "cat" or decoder_AS_type == "mean"
        decoder_AS_in_dim = None
        if decoder_AS_type == "cat":
            decoder_AS_in_dim = dec_in_dim * 2
        elif decoder_AS_type == "mean":
            decoder_AS_in_dim = dec_in_dim

        self.decoder_AS = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=decoder_AS_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            nhead_out=nhead_out,
            num_layers=num_dec_layers,
            nhead=nhead,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )

    def forward(self, graph_adj, graph_similarity, x, missing_index, num_clusters):
        loss = self.exchange_adj_prediction(graph_adj, graph_similarity, x, missing_index, num_clusters)
        return loss

    def exchange_adj_prediction(self, graph_adj, graph_similarity, x, missing_index, num_clusters):

        Z_A = self.encoder_A(graph_adj, x, )
        Z_S = self.encoder_S(graph_similarity, x, )

        loss_D, filled_A, filled_S = self.loss_D_part(Z_S, Z_A, missing_index, num_clusters)
        loss_E = self.loss_E_part(filled_S, filled_A, missing_index, x, graph_adj, graph_similarity)
        return loss_D, loss_E

    def loss_E_part(self, Z_S, Z_A, missing_index, x, graph_adj, graph_similarity):
        loss_A = self.loss_E_structure_part(Z_A, graph_adj, "A")
        loss_S = self.loss_E_structure_part(Z_S, graph_similarity, "S")

        restruct_Z = None
        if self.decoder_AS_type == "cat":
            restruct_Z = self.decoder_AS(torch.cat([Z_S, Z_A], dim=1))
        elif self.decoder_AS_type == "mean":
            restruct_Z = self.decoder_AS((Z_A + Z_S) / 2)
        loss_Z = self._get_loss_diff(x, restruct_Z, missing_index)

        loss = self.loss_E_S_para * loss_S + self.loss_E_A_para * loss_A + self.loss_E_Z_para * loss_Z
        return loss

    def loss_E_structure_part(self, Z, G, decoder):
        assert decoder == "A" or decoder == "S"

        u, v = G.edges()
        positive_samples = torch.stack([u, v], dim=1, )

        num_nodes = G.number_of_nodes()
        neg_u = torch.randint(0, num_nodes, (len(u),)).to(Z.device)
        neg_v = torch.randint(0, num_nodes, (len(v),)).to(Z.device)
        negative_samples = torch.stack([neg_u, neg_v], dim=1)

        samples = torch.cat([positive_samples, negative_samples], dim=0)
        labels = torch.cat([torch.ones(len(u)), torch.zeros(len(neg_u))]).to(Z.device)

        def decode_pairs(mlp, Z, samples):
            z_u = Z[samples[:, 0]]
            z_v = Z[samples[:, 1]]
            z_uv = torch.cat([z_u, z_v], dim=1)
            output = mlp(z_uv)
            return output

        out = None
        if decoder == "A":
            out = decode_pairs(self.decoder_A, Z, samples)
        elif decoder == "S":
            out = decode_pairs(self.decoder_S, Z, samples)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(out.squeeze(), labels)
        return loss

    def loss_D_part(self, Z_S, Z_A, missing_index, num_clusters):
        labels_S_T, centers_S_T = self.get_cluster_info(Z_S, num_clusters)
        labels_A_T, centers_A_T = self.get_cluster_info(Z_A, num_clusters)

        grad_centers_S = self.compute_centers(Z_S, labels_S_T, num_clusters)
        grad_centers_A = self.compute_centers(Z_A, labels_A_T, num_clusters)

        filled_S = self.knn_fill(Z_S, labels_A_T, missing_index, K=3)
        filled_A = self.knn_fill(Z_A, labels_S_T, missing_index, K=3)

        grad_centers_filled_S = self.compute_centers(filled_S, labels_S_T, num_clusters)
        grad_centers_filled_A = self.compute_centers(filled_A, labels_A_T, num_clusters)

        loss_D_S = self.compute_cluster_loss(grad_centers_filled_S, grad_centers_S, 0.5, labels_S_T, num_clusters)
        loss_D_A = self.compute_cluster_loss(grad_centers_filled_A, grad_centers_A, 0.5, labels_A_T, num_clusters)

        loss_D = self.loss_D_A_para * loss_D_A + self.loss_D_S_para * loss_D_S

        return loss_D, filled_A, filled_S

    def get_cluster_info(self, Z, num_clusters):
        kwargs = {
            'metric': 'cosine',
            'distributed': True,
            'random_state': 0,
            'n_clusters': num_clusters,
            'verbose': False
        }
        clustering_model = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
        psedo_labels = clustering_model.fit_predict(Z.data)
        cluster_centers = clustering_model.cluster_centers_

        return psedo_labels, cluster_centers

    def knn_fill(self, Z, labels, missing_index, K):
        n, k = Z.shape
        device = Z.device
        unique_labels = torch.unique(labels)

        result = Z.clone()
        # result = Z.detach()

        for label in unique_labels:
            label_indices = torch.where(labels == label)[0]

            label_missing = torch.tensor(list(set(label_indices.tolist()) & set(missing_index)))

            if len(label_missing) == 0:
                continue

            label_known = torch.tensor(list(set(label_indices.tolist()) - set(missing_index)))

            if len(label_known) == 0:
                continue

            knn = NearestNeighbors(n_neighbors=min(K, len(label_known)), metric='euclidean')
            knn.fit(Z[label_known].detach().cpu().numpy())

            distances, indices = knn.kneighbors(Z[label_missing].detach().cpu().numpy())

            neighbors = Z[label_known][torch.tensor(indices).to(device)]
            filled_values = torch.mean(neighbors, dim=1)

            result[label_missing] = filled_values

        mask = torch.zeros(Z.shape, dtype=torch.bool, device=device)
        mask[missing_index] = True
        return torch.where(mask, result, Z)

    def compute_cluster_loss(self, q_centers, k_centers, temperature, psedo_labels, num_cluster):
        d_q = q_centers.mm(q_centers.T) / temperature
        d_k = (q_centers * k_centers).sum(dim=1) / temperature
        d_q = d_q.float()
        d_q[torch.arange(num_cluster), torch.arange(num_cluster)] = d_k

        # q -> k
        # d_q = q_centers.mm(k_centers.T) / temperature

        zero_classes = torch.arange(num_cluster).cuda()[torch.sum(F.one_hot(torch.unique(psedo_labels),
                                                                            num_cluster), dim=0) == 0]
        mask = torch.zeros((num_cluster, num_cluster), dtype=torch.bool, device=d_q.device)
        mask[:, zero_classes] = 1
        d_q.masked_fill_(mask, -10)
        pos = d_q.diag(0)
        mask = torch.ones((num_cluster, num_cluster))
        mask = mask.fill_diagonal_(0).bool()
        neg = d_q[mask].reshape(-1, num_cluster - 1)
        loss = - pos + torch.logsumexp(torch.cat([pos.reshape(num_cluster, 1), neg], dim=1), dim=1)
        loss[zero_classes] = 0.
        loss = loss.sum() / (num_cluster - len(zero_classes))
        return loss

    def compute_centers(self, x, psedo_labels, num_cluster):
        n_samples = x.size(0)
        if len(psedo_labels.size()) > 1:
            weight = psedo_labels.T
        else:
            weight = torch.zeros(num_cluster, n_samples).to(x)  # L, N
            weight[psedo_labels, torch.arange(n_samples)] = 1
        weight = F.normalize(weight, p=1, dim=1)  # l1 normalization
        centers = torch.mm(weight, x)
        # centers = F.normalize(centers, dim=1)
        return centers

    @staticmethod
    def _get_loss_diff(input, target, missing_index):
        mask = torch.ones_like(input, dtype=torch.float32)
        mask[missing_index, :] = 0
        loss = F.kl_div(F.log_softmax(input * mask, dim=1), F.softmax(target * mask, dim=1), reduction='batchmean')
        return loss

    def embed(self, g, x):
        rep = self.encoder_A(g, x)
        return rep
