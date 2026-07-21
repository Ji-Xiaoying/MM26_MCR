"""
添加队列特征参与MCR构建, 并且使用ema输出特征的构建
以及Confidence历史平滑更新
"""

import os
from params import parse_args

# 解析命令行参数
args = parse_args()

if not hasattr(args, 'device'):
    raise ValueError("args must contain a 'device' parameter")

os.environ["CUDA_VISIBLE_DEVICES"] = args.device

import torch
import clip
from clip.criterions import ReWeightLoss
from torch import optim
from util import set_seed_logger, get_logger
from scheduler import cosine_lr
from eval import evaluate
from data import prepare_dataloader
from CR_construct import multi_phase_CR
from clip.EMA import SampleQueue, EMA

global logger

def main():
    global logger
    seed = set_seed_logger(args)
    dir_path = os.path.join(args.checkpoint_path, args.experiments)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    logger = get_logger(os.path.join(dir_path, "log.txt"))

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model_clip, preprocess = clip.clip.load(args.vision_model, device=device, jit=False)

    if args.resume:
        checkpoint = torch.load(args.resume)
        model = model_clip
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("Loaded model from {}".format(args.resume))
    else:
        model = model_clip
        logger.info("Model Initialized!")

    model = model.cuda()
    ema = EMA(model, decay=args.ema_decay)

    # 读取数据集
    dataloader = prepare_dataloader(args, args.dataset_root, preprocess, logger, 'train_base')
    train_dataloader, train_length = dataloader['train_base']
    dataloader_dev = prepare_dataloader(args, args.dataset_root, preprocess, logger, 'dev')
    dev_dataloader, dev_length = dataloader_dev['dev']

    global_confidences = torch.zeros(train_length, dtype=torch.float32).cuda()
    confidence_ema_decay = args.gamma_decay

    loss = ReWeightLoss().cuda()

    dataloader_len = len(train_dataloader)
    total_steps = dataloader_len * args.epochs

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_length)
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num steps = %d", total_steps)

    best_score = 0
    no_improvement_count = 0
    stop_flag = False

    # 创建队列
    image_queue = SampleQueue(max_size=args.queue_len, strategy='FIFO')
    text_queue = SampleQueue(max_size=args.queue_len, strategy='FIFO')

    for epoch in range(args.epochs):
        model.train()
        _ce_loss = 0

        for idx, batch in enumerate(train_dataloader):
            step = dataloader_len * epoch + idx
            scheduler(step)

            images, texts, is_clean, batch_indices = batch
            images = images.cuda()
            texts = texts.cuda()

            # 使用EMA参数进行前向推理
            ema.apply_shadow()
            with torch.no_grad():
                image_features, text_features, _, _ = model(images, texts)

            current_batch_size = image_features.shape[0]

            # ------------添加队列特征参与MCR构建--------------
            if image_queue.get_samples() is None:
                image_features_lp = image_features
                text_features_lp = text_features
            else:
                image_features_queue = image_queue.get_samples()
                text_features_queue = text_queue.get_samples()
                image_features_lp = torch.cat((image_features, image_features_queue))
                text_features_lp = torch.cat((text_features, text_features_queue))

            # 计算 I -> T 的权重
            weights_i2t, stages_i2t = multi_phase_CR(image_features_lp, text_features_lp, lambda_reg=args.lambda_reg)
            weights_i2t = weights_i2t[:current_batch_size]
            weights_i2t = weights_i2t / (weights_i2t.max() + 1e-8)

            # 计算 T -> I 的权重
            weights_t2i, stages_t2i = multi_phase_CR(text_features_lp, image_features_lp, lambda_reg=args.lambda_reg)
            weights_t2i = weights_t2i[:current_batch_size]
            weights_t2i = weights_t2i / (weights_t2i.max() + 1e-8)

            # 融合两个方向的权重 (取平均)
            confidence = (weights_i2t + weights_t2i) / 2

            # Confidence 平滑更新
            hist_confidence = global_confidences[batch_indices].cuda()

            if epoch == 0:
                smoothed_confidence = confidence
            else:
                smoothed_confidence = confidence_ema_decay * hist_confidence + (1 - confidence_ema_decay) * confidence

            global_confidences[batch_indices] = smoothed_confidence.detach()
            confidence = smoothed_confidence

            # 将判断为干净样本对的样本对加入queue队列中 (使用平滑后的 confidence)
            c_images = image_features[confidence > 0.9]
            c_texts = text_features[confidence > 0.9]
            image_queue.update(c_images)
            text_queue.update(c_texts)

            # 恢复原始训练参数
            ema.restore()
            _, _, logits_per_image, logits_per_text = model(images, texts)

            # 计算 loss (使用平滑后的 confidence 计算重加权)
            i2t_wei_loss, t2i_wei_loss = loss(logits_per_image, logits_per_text, confidence)
            final_loss = (i2t_wei_loss + t2i_wei_loss) / 2

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()
            ema.update()  # 更新ema
            _ce_loss += float(final_loss)

            if (idx % args.display == 0) and (idx != 0):
                curr_q_samples = image_queue.get_samples()
                curr_q_len = curr_q_samples.shape[0] if curr_q_samples is not None else 0

                logger.info("Epoch: %d/%d, step:%d/%d, lr: %.8f, ce_loss: %f, Queue_Len: %d",
                            epoch + 1, args.epochs, idx, dataloader_len,
                            optimizer.param_groups[0]['lr'], _ce_loss / args.display, curr_q_len)
                _ce_loss = 0

            if (step + 1) % args.test_frequency == 0:
                logger.info("Eval on val dataset")
                ema.apply_shadow()
                model.eval()
                eval_Rank = evaluate(model, dev_dataloader, 'dev', logger)

                if best_score <= eval_Rank:
                    best_score = eval_Rank
                    save_path = os.path.join(dir_path, f'step{step + 1}.pt')
                    torch.save(
                        {
                            "step": step + 1,
                            "name": args.name,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        save_path,
                    )
                    logger.info("The best model. Saved checkpoint {} (step {})".format(save_path, step + 1))
                    no_improvement_count = 0
                else:
                    logger.info("Not the best model. Not save.")
                    no_improvement_count += 1

                logger.info("The best model is: {}, the R1 is: {:.4f}".format(save_path, best_score))

                ema.restore()
                model.train()

            # if no_improvement_count >= 4:
            #     logger.info("Early stopping after {} consecutive validations without improvement.".format(no_improvement_count))
            #     stop_flag = True
            #     break

        if stop_flag:
            break

if __name__ == '__main__':
    main()
