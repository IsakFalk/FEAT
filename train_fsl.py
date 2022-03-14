import numpy as np
import os
import torch
import torch.nn.functional as F

import os
from tqdm import tqdm

from omegaconf import open_dict
import hydra

# NOTE: This from FEAT
from model.trainer.fsl_trainer import FSLOurTrainer
from model.trainer.helpers import prepare_optimizer, prepare_model
from model.utils import (
    pprint, set_gpu,
    get_command_line_parser,
    postprocess_args,
)

from torch.backends import cudnn as cudnn

# NOTE: Below from MeLa
import util
from dataset.base_dataset import MetaDataset
from dataset.data_util import get_dataset
from meta_learner import MetaLS
from models.resfc import ResFC
from models.util import create_model
from routines import parse_option
from train_routine import full_train, get_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_option_feat(opt):
    # Hydra by default makes the config frozen so that
    # we can't add additional key-value pairs to it. This is
    # good since it means we won't fail silently.
    #
    # the open_dict context manager disables this in the scope of
    # the with statement.
    with open_dict(opt):
        opt.model_name = f'_{opt.dataset}'

        if "train_db_size" in opt:
            opt.model_name += f"_f{opt.train_db_size}"
        if "epochs" in opt:
            opt.model_name += f"_e{opt.epochs}"
        if "no_replacement" in opt:
            if opt.no_replacement:
                opt.model_name += "_no-replace"

        opt.model_name += f"nways{opt.n_ways}_nshots{opt.n_shots}_{opt.trial}"

        opt.model_path = os.path.expanduser(opt.model_path)

        opt.n_gpu = torch.cuda.device_count()
        opt.data_root = os.path.expanduser(opt.data_root)

        # Finally, since we can't reuse variables in yaml, we'll
        # just add them to the omegaconf here
        opt.way = opt["n_way"]
        opt.shot = opt["n_shots"]
        opt.query = opt["n_queries"]
        opt.eval_way = opt["n_way"]
        opt.eval_shot = opt["n_shots"]
        opt.eval_query = opt["n_queries"]

    return opt

def train_feat(opt, model, train_loader, optimizer, label, label_aux, progress=False):
    avg_metric = util.AverageMeter()

    if progress:
        train_loader = tqdm(train_loader)

    for i, batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        data, _ = # TODO
        # TODO: Make sure there's an extra batch dimension and that it's the expected input format
        batch_data = util.to_cuda_list(batch_data[:2])
        # TODO: Make batch_data into instances and labels and merge support and query
        logits, reg_logits = model(data)
        if reg_logits is not None:
            loss = F.cross_entropy(logits, label)
            total_loss = loss + opt.balance * F.cross_entropy(reg_logits, label_aux)
        else:
            loss = F.cross_entropy(logits, label)
            total_loss = F.cross_entropy(logits, label)

        total_loss.backward()
        optimizer.step()
        avg_metric.update([total_loss.item()])

    return avg_metric.avg

def test_fn_feat(opt, model, test_loader):
    model.eval()

    label = torch.arange(opt.eval_way, dtype=torch.int16).repeat(opt.eval_query)
    label = label.type(torch.LongTensor)
    if torch.cuda.is_available():
        label = label.cuda()
    with torch.no_grad():
        for batch_data in test_loader:
            data = # TODO: get from batch_data
            logits = model(data)
            loss = F.cross_entropy(logits, label)
            # TODO: Port to our setting
            acc = count_acc(logits, label)
    model.train()
    return acc

def prepare_label(opt):
    # prepare one-hot label
    label = torch.arange(opt.way, dtype=torch.int16).repeat(opt.query)
    label_aux = torch.arange(opt.way, dtype=torch.int8).repeat(opt.shot + opt.query)

    label = label.type(torch.LongTensor)
    label_aux = label_aux.type(torch.LongTensor)

    if torch.cuda.is_available():
        label = label.cuda()
        label_aux = label_aux.cuda()

    return label, label_aux


def full_train_feat(opt, model, train_loader, test_loader, optimizer, lr_sch, logger, eval_cond, test_fn=test_fn):
    if torch.cuda.is_available():
        cudnn.benchmark = True

    model.train()
    if opt.fix_BN:
        model.encoder.eval()

    label, label_aux = prepare_label(opt)
    best = 0
    for epoch in range(opt.epochs):
        model.train()
        if opt.fix_BN:
            model.encoder.eval()

        train_loss = train_feat(opt, model, train_loader, optimizer, label, label_aux, progress=opt.progress)
        if lr_sch:
            lr_sch.step()

        logger.info(f"epoch {epoch}")
        info = util.print_metrics(["train_loss"], train_loss)
        logger.info(info)

        if eval_cond(epoch):
            test_acc = test_fn_feat(opt, model, test_loader)
            logger.info(f"val acc: {test_acc}")
            if test_acc > best:
                best = test_acc
                # TODO: Check that this works with feat
                util.save_routine(epoch, model, optimizer, f"{opt.model_path}/{opt.model_name}_best")


@hydra.main(config_path="config", config_name="feat.yaml")
def feat(opt):
    opt = parse_option_feat(opt)
    with open_dict(opt):
        opt.model_name = f"feat"

    # We never set rotate aug for this dataset, but always turn on data augmentation
    dataset, n_cls = get_dataset(opt, partition="train", rotate_aug=False)
    meta_train = MetaDataset(dataset, opt, no_replacement=opt.no_replacement, db_size=opt.train_db_size,
                             fixed_db=opt.fixed_db)
    trainloader = get_dataloader(meta_train, 1, opt.num_workers)

    val_dataset = get_dataset(opt, partition="val", rotate_aug=False)[0]
    valloader = get_dataloader(val_dataset, 256, opt.num_workers, shuffle=False)

    test_dataset = get_dataset(opt, partition="test", rotate_aug=False)[0]
    testloader = get_dataloader(val_dataset, 256, opt.num_workers, shuffle=False)

    logger = util.get_logger(opt.logger_name, file_name=f"{opt.logger_name}_{opt.model_name}")
    logger.info(opt)

    # Initialize backbone
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # NOTE: para_model (2nd arg) is for multi-GPU parallelization, we don't use it
    # so we can disregard this
    model, _  = prepare_model(opt)
    model = model.to(device)
    optimizer, lr_scheduler = prepare_optimizer(model, opt)

    full_train_feat(opt, model, trainloader, valloader, optimizer, lr_scheduler, logger, lambda x: x >= 0)

    (mean, confidence), feats = meta_test_new_feat(model, testloader, opt)
    logger.info(f"{opt.n_shot}-shot Acc: {mean}, Std: {confidence}")
