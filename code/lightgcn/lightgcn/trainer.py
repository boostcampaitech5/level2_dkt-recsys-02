import os
import pdb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from torch import nn
from torch_geometric.nn.models import LightGCN
import wandb
import yaml
from .utils import get_logger, logging_conf


logger = get_logger(logger_conf=logging_conf)


def build(n_node: int, weight: str = None, **kwargs):
    model = LightGCN(num_nodes=n_node, **kwargs)
    if weight:
        if not os.path.isfile(path=weight):
            logger.fatal("Model Weight File Not Exist")
        logger.info("Load model")
        state = torch.load(f=weight)["model"]
        model.load_state_dict(state)
        return model
    else:
        logger.info("No load model")
        return model


def run(
    model: nn.Module,
    train_data: dict,
    valid_data: dict = None,
    n_epochs: int = 100,
    learning_rate: float = 0.01,
    model_dir: str = None,
):
    model.train()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    os.makedirs(name=model_dir, exist_ok=True)

    if valid_data is None:
        eids = np.arange(len(train_data["label"]))
        eids = np.random.permutation(eids)[:1000]
        edge, label = train_data["edge"], train_data["label"]
        label = label.to("cpu").detach().numpy()
        pdb.set_trace()
        valid_data = dict(edge=edge[:, eids], label=label[eids])

    logger.info(f"Training Started : n_epochs={n_epochs}")
    best_auc, best_epoch = 0, -1

    for e in range(n_epochs):
        logger.info("Epoch: %s", e)
        # TRAIN
        train_auc, train_acc, train_loss = train(train_data=train_data, model=model, optimizer=optimizer)
    
        # VALID
        auc, acc = validate(valid_data=valid_data, model=model)
        wandb.log(dict(train_loss_epoch=train_loss,
                       train_acc_epoch=train_acc,
                       train_auc_epoch=train_auc,
                       valid_acc_epoch=acc,
                       valid_auc_epoch=auc))

        if auc > best_auc:
            logger.info("Best model updated AUC from %.4f to %.4f", best_auc, auc)
            best_auc, best_epoch = auc, e
            best_acc = acc
            torch.save(obj= {"model": model.state_dict(), "epoch": e + 1},
                       f=os.path.join(model_dir, f"lgcn_best_model.pt"))
    torch.save(obj={"model": model.state_dict(), "epoch": e + 1},
               f=os.path.join(model_dir, f"last_model.pt"))
    logger.info(f"Best Weight Confirmed : {best_epoch+1}'th epoch")

    if args.sweep_run:
        wandb.log({
            'val_auc': best_auc,
            'val_acc': best_acc,
        })
        
        curr_dir = __file__[:__file__.rfind('/')+1]
        with open(curr_dir + '../sweep_best_auc.yaml') as file:
            output = yaml.load(file, Loader=yaml.FullLoader)
        file.close()
            
        if output[args.model.lower()]['best_auc'] < best_auc:
            output[args.model.lower()]['best_auc'] = float(best_auc)
            output[args.model.lower()]['parameter'] = dict(zip(dict(wandb.config).keys(),map(lambda x: x if type(x) == str else float(x) , dict(wandb.config).values())))
            
        with open(curr_dir + '../sweep_best_auc.yaml', 'w') as file:
            yaml.dump(output, file, default_flow_style=False)
        file.close()
        
    return model.get_embedding(train_data['edge']).detach().cpu().numpy()




def run_kfold(
    args,
    folds: list,
    n_node: int,
    device: str,
    n_epochs: int = 100,
    learning_rate: float = 0.01,
    model_dir: str = None,
):
    
    folds_weights = []

    for dic in folds:
        model = build(
            n_node=n_node,
            embedding_dim=args.hidden_dim,
            num_layers=args.n_layers,
            alpha=args.alpha
        )

        model = model.to(device)
        train_data = dic['train']
        valid_data = dic['valid']
        pdb.set_trace()
        edge, label = valid_data["edge"], valid_data["label"].to("cpu").detach().numpy()
        valid_data = dict(edge, label)

        model.train()

        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

        os.makedirs(name=model_dir, exist_ok=True)

        logger.info(f"Training Started : n_epochs={n_epochs}")
        best_auc, best_epoch = 0, -1

        for e in range(n_epochs):
            logger.info("Epoch: %s", e)
            # TRAIN
            train_auc, train_acc, train_loss = train(train_data=train_data, model=model, optimizer=optimizer)
        
            # VALID
            auc, acc = validate(valid_data=valid_data, model=model)
            wandb.log(dict(train_loss_epoch=train_loss,
                        train_acc_epoch=train_acc,
                        train_auc_epoch=train_auc,
                        valid_acc_epoch=acc,
                        valid_auc_epoch=auc))

            if auc > best_auc:
                logger.info("Best model updated AUC from %.4f to %.4f", best_auc, auc)
                best_auc, best_epoch = auc, e
                best_acc = acc
                best_model = model.state_dict()
        
        folds_weights.append(best_model)
        logger.info(f"Best Weight Confirmed : {best_epoch+1}'th epoch")

    average_weights = {}
    for key in folds_weights[0].keys():
        average_weights[key] = torch.stack([fold[key].float() for fold in folds_weights], dim=0).mean(dim=0)
    for key in average_weights: average_weights[key].to(dtype = torch.int64)

    model = LightGCN(num_nodes=n_node)
    model.load_state_dict(average_weights)

    return model.get_embedding(train_data['edge']).detach().cpu().numpy()




def train(model: nn.Module, train_data: dict, optimizer: torch.optim.Optimizer):
    pred = model(train_data["edge"])
    loss = model.link_pred_loss(pred=pred, edge_label=train_data["label"])
    
    prob = model.predict_link(edge_index=train_data["edge"], prob=True)
    prob = prob.detach().cpu().numpy()

    label = train_data["label"].cpu().numpy()
    acc = accuracy_score(y_true=label, y_pred=prob > 0.5)
    auc = roc_auc_score(y_true=label, y_score=prob)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    logger.info("TRAIN AUC : %.4f ACC : %.4f LOSS : %.4f", auc, acc, loss.item())
    return auc, acc, loss


def validate(valid_data: dict, model: nn.Module):
    with torch.no_grad():
        prob = model.predict_link(edge_index=valid_data["edge"], prob=True)
        prob = prob.detach().cpu().numpy()
        
        label = valid_data["label"]
        acc = accuracy_score(y_true=label, y_pred=prob > 0.5)
        auc = roc_auc_score(y_true=label, y_score=prob)
    logger.info("VALID AUC : %.4f ACC : %.4f", auc, acc)
    return auc, acc


def inference(model: nn.Module, data: dict, output_dir: str):
    model.eval()
    with torch.no_grad():
        pred = model.predict_link(edge_index=data["edge"], prob=True)
        
    logger.info("Saving Result ...")
    pred = pred.detach().cpu().numpy()
    os.makedirs(name=output_dir, exist_ok=True)
    write_path = os.path.join(output_dir, "submission.csv")
    pd.DataFrame({"prediction": pred}).to_csv(path_or_buf=write_path, index_label="id")
    logger.info("Successfully saved submission as %s", write_path)

