import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import clip
from torch import optim
from util import set_seed_logger, get_logger
from params import parse_args
from scheduler import cosine_lr
from eval import evaluate, evaluate_1K
from data import prepare_dataloader

global logger


def main():
    global logger
    args = parse_args()

    seed = set_seed_logger(args)
    dir_path = os.path.join(args.checkpoint_path, args.experiments)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    logger = get_logger(os.path.join(dir_path, "log.txt"))

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_clip, preprocess = clip.clip.load(args.vision_model, device=device,
                                            jit=False)  # Must set jit=False for training

    if args.resume:
        checkpoint = torch.load(args.resume)
        model = model_clip
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("Loaded model from {}".format(args.resume))
    else:
        model = model_clip
        logger.info("Model Initialized!")

    model = model.cuda()

    dataloader = prepare_dataloader(args, args.dataset_root, preprocess, logger, 'test')
    test_dataloader, test_length = dataloader['test']

    if args.dataset == 'coco':
        eval_Rank = evaluate(model, test_dataloader, 'test', logger)
        eval_Rank_1K = evaluate_1K(args, model, test_dataloader, logger)  # Only for MSCOCO 1K testing
    else:
        eval_Rank = evaluate(model, test_dataloader, 'test', logger)


if __name__ == '__main__':
    main()
