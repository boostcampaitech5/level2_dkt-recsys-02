import math
import os

import numpy as np
import torch
from torch import nn
from torch.nn.functional import sigmoid
import wandb
from dkt import trainer
from .criterion import get_criterion
from .dataloader import get_loaders
from .metric import get_metric
from .model import LSTM, LSTMATTN, BERT, Saint, LastQuery, TransLSTM_G,LongShort
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .utils import get_logger, logging_conf
import pdb
import yaml
import gc

logger = get_logger(logger_conf=logging_conf)


def run(args,
        train_data: np.ndarray,
        valid_data: np.ndarray,
        model: nn.Module):
    train_loader, valid_loader = get_loaders(args=args, train=train_data, valid=valid_data)

    # For warmup scheduler which uses step interval
    args.total_steps = int(math.ceil(len(train_loader.dataset) / args.batch_size)) * (
        args.n_epochs
    )
    args.warmup_steps = args.total_steps // 10

    optimizer = get_optimizer(model=model, args=args)
    scheduler = get_scheduler(optimizer=optimizer, args=args)

    best_auc = -1
    early_stopping_counter = 0
    losses = []
    aucs = []
    accs = []
    for epoch in range(args.n_epochs):
        logger.info("Start Training: Epoch %s", epoch + 1)

        # TRAIN
 
        train_auc, train_acc, train_loss = train(train_loader=train_loader,
                                                    model=model, optimizer=optimizer,
                                                    scheduler=scheduler, args=args)

        # VALID
        auc, acc, loss = validate(valid_loader=valid_loader, model=model, args=args)

        wandb.log(dict(epoch=epoch,
                       train_loss_epoch=train_loss,
                       train_auc_epoch=train_auc,
                       train_acc_epoch=train_acc,
                       valid_loss_epoch=loss,
                       valid_auc_epoch=auc,
                       valid_acc_epoch=acc))
        
        losses.append(loss)
        aucs.append(auc)
        accs.append(acc)
        
        if auc > best_auc:
            best_auc = auc
            # nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, "module") else model
            save_checkpoint(state={"epoch": epoch + 1,
                                   "state_dict": model_to_save.state_dict()},
                            model_dir=args.model_dir,
                            #########모델 이름_best_model.pt로 저장하기
                            model_filename=f"{args.model.lower()}_best_model.pt")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                logger.info(
                    "EarlyStopping counter: %s out of %s",
                    early_stopping_counter, args.patience
                )
                break

        # scheduler
        if args.scheduler == "plateau":
            scheduler.step(best_auc)
            
    if args.sweep_run:
        val_loss_avg = sum(losses)/len(losses)
        val_auc_avg = sum(aucs)/len(aucs)
        val_acc_avg = sum(accs)/len(accs)
        
        wandb.log({
            'val_loss': val_loss_avg,
            'val_auc': val_auc_avg,
            'val_acc': val_acc_avg,
        })
        
        try:
            with open('./sweep_best_auc.yaml') as file:
                output = yaml.load(file, Loader=yaml.FullLoader)
            file.close()
        except:
            with open('/opt/level2_dkt-recsys-02/code/dkt/sweep_best_auc.yaml') as file:
                output = yaml.load(file, Loader=yaml.FullLoader)
            file.close()
        if output[args.model.lower()]['best_auc'] < val_auc_avg:
            output[args.model.lower()]['best_auc'] = float(val_auc_avg)
            output[args.model.lower()]['parameter'] = dict(zip(dict(wandb.config).keys(),map(float, dict(wandb.config).values())))
            
        with open('./sweep_best_auc.yaml', 'w') as file:
            yaml.dump(output, file, default_flow_style=False)
        file.close()


def run_kfold(args,
        kfolds: list):
    

    fold_weights = []

    for k, fold_dict in enumerate(kfolds):
        model: torch.nn.Module = trainer.get_model(args=args).to(args.device)
        logger.info("Start Training: Fold %s", k + 1)
        train_data = fold_dict['train']
        valid_data = fold_dict['val']
        train_loader, valid_loader = get_loaders(args=args, train=train_data, valid=valid_data)

        # For warmup scheduler which uses step interval
        args.total_steps = int(math.ceil(len(train_loader.dataset) / args.batch_size)) * (
            args.n_epochs
        )
        args.warmup_steps = args.total_steps // 10

        optimizer = get_optimizer(model=model, args=args)
        scheduler = get_scheduler(optimizer=optimizer, args=args)

        best_auc = -1
        early_stopping_counter = 0
        for epoch in range(args.n_epochs):
            logger.info("Start Training: Epoch %s", epoch + 1)

            # TRAIN
            train_auc, train_acc, train_loss = train(train_loader=train_loader,
                                                    model=model, optimizer=optimizer,
                                                    scheduler=scheduler, args=args)

            # VALID
            auc, acc, loss = validate(valid_loader=valid_loader, model=model, args=args)

            wandb.log(dict(epoch=epoch,
                        train_loss_epoch=train_loss,
                        train_auc_epoch=train_auc,
                        train_acc_epoch=train_acc,
                        valid_auc_epoch=auc,
                        valid_acc_epoch=acc))
            
            if auc > best_auc:
                best_auc = auc
                # nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
                model_to_save = model.module if hasattr(model, "module") else model
                #save_checkpoint(state={"epoch": epoch + 1,
                #                    "state_dict": model_to_save.state_dict()},
                #                model_dir=args.model_dir,
                #                #########모델 이름_best_model.pt로 저장하기
                #                model_filename=f"{args.model.lower()}_best_model_fold{k+1}.pt")
                early_stopping_counter = 0
                fold_weights.append(model.state_dict())
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= args.patience:
                    logger.info(
                        "EarlyStopping counter: %s out of %s",
                        early_stopping_counter, args.patience
                    )
                    break

            # scheduler
            if args.scheduler == "plateau":
                scheduler.step(best_auc)

    average_weights = {}
    for key in fold_weights[0].keys():
        average_weights[key] = torch.stack([fold[key].float() for fold in fold_weights], dim=0).mean(dim=0)
    for key in average_weights: average_weights[key].to(dtype = torch.int64)

    model: torch.nn.Module = trainer.get_model(args=args).to(args.device)
    model.load_state_dict(average_weights)

    save_checkpoint(state={"state_dict": model_to_save.state_dict()},
                            model_dir=args.model_dir,
                            model_filename=f"{args.model.lower()}_best_model_kfold.pt")


def train(train_loader: torch.utils.data.DataLoader,
          model: nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          args):
    model.train()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(train_loader):
        for key in batch:
            tmp = {k: v.to(args.device) for k, v in batch[key].items()}
            batch[key] = tmp

        preds = model(batch)
        targets = batch['category']["answerCode"]
        
        loss = compute_loss(preds=preds, targets=targets)
        update_params(loss=loss, model=model, optimizer=optimizer,
                      scheduler=scheduler, args=args)

        if step % args.log_steps == 0:
            logger.info("Training steps: %s Loss: %.4f", step, loss.item())

        # predictions
        preds = sigmoid(preds[:, -1])
        targets = targets[:, -1]

        total_preds.append(preds.detach())
        total_targets.append(targets.detach())
        losses.append(loss)

    total_preds = torch.cat(total_preds).cpu().numpy()
    total_targets = torch.cat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc, acc = get_metric(targets=total_targets, preds=total_preds)
    loss_avg = sum(losses) / len(losses)
    logger.info("TRAIN AUC : %.4f ACC : %.4f", auc, acc)

    del batch, total_preds, total_targets, preds, losses
    gc.collect()
    torch.cuda.empty_cache()

    return auc, acc, loss_avg

def validate(valid_loader: nn.Module, model: nn.Module, args):
    model.eval()

    total_preds = []
    total_targets = []
    losses = []
    with torch.no_grad():
        for step, batch in enumerate(valid_loader):

            for key in batch:
                tmp = {k: v.to(args.device) for k, v in batch[key].items()}
                batch[key] = tmp
        
            #preds = model(**batch)
            preds = model(batch)
            targets = batch['category']["answerCode"]

            loss = compute_loss(preds=preds, targets=targets)
            # predictions
            preds = sigmoid(preds[:, -1])
            targets = targets[:, -1]

            total_preds.append(preds.detach())
            total_targets.append(targets.detach())
            losses.append(loss)

    total_preds = torch.cat(total_preds).cpu().numpy()
    total_targets = torch.cat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc, acc = get_metric(targets=total_targets, preds=total_preds)
    loss_avg = sum(losses) / len(losses)
    logger.info("VALID AUC : %.4f ACC : %.4f", auc, acc)

    del batch, total_preds, total_targets, preds, losses
    gc.collect()
    torch.cuda.empty_cache()

    return auc, acc, loss_avg



def inference(args, test_data: np.ndarray, model: nn.Module) -> None:
    model.eval()
    _, test_loader = get_loaders(args=args, train=None, valid=test_data)

    total_preds = []
    for step, batch in enumerate(test_loader):
         
        for key in batch:
            tmp = {k: v.to(args.device) for k, v in batch[key].items()}
            batch[key] = tmp
        
        #preds = model(**batch)
        preds = model(batch)
        # predictions
        preds = sigmoid(preds[:, -1])
        preds = preds.cpu().detach().numpy()
        total_preds += list(preds)

    write_path = os.path.join(args.output_dir, "submission.csv")
    os.makedirs(name=args.output_dir, exist_ok=True)
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            ### thresholding
            #if p > 0.5: p = 1
            #else: p = 0
            w.write("{},{}\n".format(id, p))
    logger.info("Successfully saved submission as %s", write_path)


def get_model(args) -> nn.Module:
    model_args = dict(
        args = args,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_tests=args.n_tests,
        n_questions=args.n_questions,
        n_tags=args.n_tags,
    )
    try:
        model_name = args.model.lower()
        model = {
            "lstm": LSTM,
            "lstmattn": LSTMATTN,
            "bert": BERT,
            'saint':Saint,
            'lastquery':LastQuery,
            'translstm_g' :TransLSTM_G,
            'longshort':LongShort
        }.get(model_name)(**model_args)
  
    except KeyError:
        logger.warn("No model name %s found", model_name)
    except Exception as e:
        logger.warn("Error while loading %s with args: %s", model_name, model_args)
        raise e
    return model


def compute_loss(preds: torch.Tensor, targets: torch.Tensor):
    """
    loss계산하고 parameter update
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = get_criterion(pred=preds, target=targets.float())

    # 마지막 시퀀드에 대한 값만 loss 계산
    loss = loss[:, -1]
    loss = torch.mean(loss)
    return loss


def update_params(loss: torch.Tensor,
                  model: nn.Module,
                  optimizer: torch.optim.Optimizer,
                  scheduler: torch.optim.lr_scheduler._LRScheduler,
                  args):
    if args.model.lower() == 'longshort': loss.backward(retain_graph=True)
    else: loss.backward()
    
    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    if args.scheduler == "linear_warmup":
        scheduler.step()
    optimizer.step()
    optimizer.zero_grad()


def save_checkpoint(state: dict, model_dir: str, model_filename: str) -> None:
    """ Saves checkpoint to a given directory. """
    save_path = os.path.join(model_dir, model_filename)
    logger.info("saving model as %s...", save_path)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(state, save_path)


def load_model(args):
    ##########모델 이름_best_model.pt 불러오기
    model_path = os.path.join(args.model_dir, args.model.lower() + '_' +  args.model_name)
    logger.info("Loading Model from: %s", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # load model state
    model.load_state_dict(load_state["state_dict"], strict=True)
    logger.info("Successfully loaded model state from: %s", model_path)
    return model
