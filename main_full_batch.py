import numpy as np
from tqdm import tqdm
import torch
import random
import wandb
import os

from utils import (
    build_args,
    create_optimizer,
    load_missing_graph_dataset,
)

from utils import cluster_probing_full_batch
from models import build_model

os.environ['WANDB_API_KEY'] = ""
os.environ['WANDB_HOST'] = ""
# os.environ['WANDB_MODE'] = "offline"

def train(model, graph_adj, graph_similarity, feat, missing_index, optimizer, max_epoch, device, num_classes, args):
    graph_adj = graph_adj.to(device)
    graph_similarity = graph_similarity.to(device)
    x = feat.to(device)
    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        model.train()

        loss_D, loss_E = model(graph_adj, graph_similarity, x, missing_index, num_classes)
        loss = args.D_para * loss_D + args.E_para * loss_E

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")

        if epoch % 10 == 0:
            result_str = cluster_probing_full_batch(model, graph_adj, x, device)
            loss_log = {"loss_D": args.D_para * loss_D, "loss_E": args.E_para * loss_E, "loss": loss}
            wandb.log(loss_log)

    print("Final Results:")
    print(result_str)
    return model


def main(args):
    device = f"{args.device}"
    dataset_name = args.dataset
    max_epoch = args.max_epoch

    optim_type = args.optimizer

    lr = args.lr
    weight_decay = args.weight_decay
    missing_rate = args.missing_rate

    graph_adj, graph_similarity, missing_index, (num_features, num_classes) = load_missing_graph_dataset(dataset_name, missing_rate)

    args.num_features = num_features
    args.num_nodes = graph_adj.ndata["feat"].size(0)

    model = build_model(args)
    model.to(device)
    optimizer = create_optimizer(optim_type, model, lr, weight_decay)

    x = graph_adj.ndata["feat"]
    labels = graph_adj.ndata["label"]
    

    train(model, graph_adj, graph_similarity, x, missing_index, optimizer, max_epoch, device, num_classes, args)


if __name__ == "__main__":
    args = build_args()


    print(f'{args.missing_rate}  {args.dataset}')

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    project_name = f""
    group_name = f""

    run = wandb.init(project=project_name, entity='', reinit=True,
                                 group=group_name,
                                 tags=[f"dataset{args.dataset}", f"encoder{args.encoder}", f"decoder{args.decoder}", f"seed{args.seed}", f"device{args.device}", f"lr{args.lr}", f"max_epoch{args.max_epoch}", f"E_para{args.E_para}", f"D_para{args.D_para}", f"loss_E_S_para{args.loss_E_S_para}", f"loss_E_A_para{args.loss_E_A_para}", f"loss_E_Z_para{args.loss_E_Z_para}", f"loss_D_S_para{args.loss_D_S_para}", f"loss_D_A_para{args.loss_D_A_para}", f"decoder_AS_type{args.decoder_AS_type}", f"missing_rate{args.missing_rate}"])

    print(args)

    main(args)

    run.finish()
