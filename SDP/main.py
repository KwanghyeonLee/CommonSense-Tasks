import logging.config
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from configuration import config
from utils.augment import select_transform
from utils.data_loader import get_test_datalist
from utils.data_loader import get_train_datalist
from utils.method_manager import select_method


def main():
    args = config.base_parser()

    logging.config.fileConfig("./configuration/logging.conf")
    logger = logging.getLogger()

    os.makedirs(f"results/{args.dataset}/{args.note}", exist_ok=True)
    os.makedirs(f"tensorboard/{args.dataset}/{args.note}", exist_ok=True)
    fileHandler = logging.FileHandler(f'results/{args.dataset}/{args.note}/seed_{args.rnd_seed}.log', mode="w")

    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    writer = SummaryWriter(f'tensorboard/{args.dataset}/{args.note}/seed_{args.rnd_seed}')

    logger.info(args)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        if args.gpu_transform:
            args.gpu_transform = False
            logger.warning("Augmentation on GPU not available!")
    logger.info(f"Set the device ({device})")

    # Fix the random seeds
    torch.manual_seed(args.rnd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)

    train_transform, test_transform, n_classes = select_transform(args)

    logger.info(f"Select a CIL method ({args.mode})")
    criterion = nn.CrossEntropyLoss(reduction="mean")
    method = select_method(
        args, criterion, device, train_transform, test_transform, n_classes
    )

    eval_results = defaultdict(list)

    samples_cnt = 0
    test_datalist = get_test_datalist(args.dataset)

    # get datalist
    train_datalist, cls_dict, cls_addition = get_train_datalist(args.dataset, args.sigma, args.repeat, args.init_cls,
                                                                args.rnd_seed)

    method.n_samples(len(train_datalist))

    # Reduce datalist in Debug mode
    if args.debug:
        random.shuffle(train_datalist)
        train_datalist = train_datalist[:5000]
        random.shuffle(test_datalist)
        test_datalist = test_datalist[:2000]

    for i, data in enumerate(train_datalist):
        samples_cnt += 1
        method.online_step(data, samples_cnt, args.n_worker)
        if samples_cnt % args.eval_period == 0:
            eval_dict = method.online_evaluate(test_datalist, samples_cnt, 512, args.n_worker, cls_dict, cls_addition,
                                               data["time"])
            eval_results["test_acc"].append(eval_dict['avg_acc'])
            eval_results["percls_acc"].append(eval_dict['cls_acc'])
            eval_results["data_cnt"].append(samples_cnt)
    if eval_results["data_cnt"][-1] != samples_cnt:
        eval_dict = method.online_evaluate(test_datalist, samples_cnt, 512, args.n_worker, cls_dict, cls_addition,
                                           data["time"])

    A_last = eval_dict['avg_acc']

    if args.mode == 'gdumb':
        eval_results = method.evaluate_all(test_datalist, args.memory_epoch, args.batchsize, args.n_worker, cls_dict,
                                           cls_addition)

    np.save(f'results/{args.dataset}/{args.note}/seed_{args.rnd_seed}_eval.npy', eval_results['test_acc'])
    np.save(f'results/{args.dataset}/{args.note}/seed_{args.rnd_seed}_eval_time.npy', eval_results['data_cnt'])

    # Accuracy (A)
    A_auc = np.mean(eval_results["test_acc"])

    KLR_avg = np.mean(method.knowledge_loss_rate[1:])
    KGR_avg = np.mean(method.knowledge_gain_rate)

    logger.info(f"======== Summary =======")
    logger.info(f"A_auc {A_auc} | A_last {A_last} | KLR_avg {KLR_avg} | KGR_avg {KGR_avg}")


if __name__ == "__main__":
    main()
