import os
import numpy as np
import torch

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Config

from data import QAData

def run(args, logger):
    config = T5Config.from_pretrained(args.model_name, pruning_method=args.pruning_method, mask_init=args.mask_init, mask_scale=args.mask_scale)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_data = QAData(logger, args, args.train_file, True)
    dev_data = QAData(logger, args, args.predict_file, False)

    train_data.load_dataset(tokenizer)
    train_data.load_dataloader()

    dev_data.load_dataset(tokenizer)
    dev_data.load_dataloader()

    if args.do_train:
        if args.checkpoint is not None:
            def convert_to_single_gpu(state_dict):
                def _convert(key):
                    if key.startswith('module.'):
                        return key[7:]
                    return key
                return {_convert(key):value for key, value in state_dict.items()}
            model = T5ForConditionalGeneration.from_pretrained(args.model_name, state_dict=convert_to_single_gpu(torch.load(args.checkpoint)), config=config)
        else:
            model = T5ForConditionalGeneration.from_pretrained(args.model_name, config=config)
        if args.n_gpu>1:
            model = torch.nn.DataParallel(model)

        if torch.cuda.is_available():
            model.to(torch.device("cuda"))

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if "mask_score" in n and p.requires_grad], "lr": args.mask_scores_learning_rate,},
            {"params": [p for n, p in model.named_parameters() if "mask_score" not in n and p.requires_grad and not any(nd in n for nd in no_decay)], "lr": args.learning_rate, "weight_decay": args.weight_decay,},
            {"params": [p for n, p in model.named_parameters() if "mask_score" not in n and p.requires_grad and any(nd in n for nd in no_decay)], "lr": args.learning_rate, "weight_decay": 0.0,},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)




        scheduler =  get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=args.warmup_steps,
                                        num_training_steps=100000)
        train(args, logger, model, train_data, dev_data, optimizer, scheduler)

    if args.do_predict:
        checkpoint = os.path.join(args.output_dir, 'best-model.pt')
        def convert_to_single_gpu(state_dict):
            def _convert(key):
                if key.startswith('module.'):
                    return key[7:]
                return key
            return {_convert(key):value for key, value in state_dict.items()}
        model = T5Config.from_pretrained(args.model_name, state_dict=convert_to_single_gpu(torch.load(args.checkpoint)), config=config)
        logger.info("Loading checkpoint from {}".format(checkpoint))
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
        model.eval()
        ems = inference(args, model, dev_data, save_predictions=True)
        logger.info("%s on %s data: %.2f" % (dev_data.metric, dev_data.data_type, np.mean(ems)*100))

def train(args, logger, model, train_data, dev_data, optimizer, scheduler):
    model.train()
    global_step = 0
    if args.global_topk:
        threshold_mem = None    
    train_losses = []
    best_accuracy = -1
    stop_training=False

    logger.info("Starting training!")
    t_total = int(args.num_train_epochs * len(train_data.dataloader))
    for epoch in range(int(args.num_train_epochs)):
        for batch in train_data.dataloader:
            global_step += 1

            threshold, regu_lambda = schedule_threshold(
                step=global_step,
                total_step=t_total,
                warmup_steps=args.warmup_steps,
                final_threshold=args.final_threshold,
                initial_threshold=args.initial_threshold,
                final_warmup=args.final_warmup,
                initial_warmup=args.initial_warmup,
                final_lambda=args.final_lambda,
            )
            # Global TopK
            if args.global_topk:
                if threshold == 1.0:
                    threshold = -1e2  # Or an indefinitely low quantity
                else:
                    if (threshold_mem is None) or (global_step % args.global_topk_frequency_compute == 0):
                        # Sort all the values to get the global topK
                        concat = torch.cat(
                            [param.view(-1) for name, param in model.named_parameters() if "mask_scores" in name]
                        )
                        n = concat.numel()
                        kth = max(n - (int(n * threshold) + 1), 1)
                        threshold_mem = concat.kthvalue(kth).values.item()
                        threshold = threshold_mem
                    else:
                        threshold = threshold_mem
                    
            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch]
            batch[2][batch[2] == 0] = -100 # -100 is the canceled loss one
            loss = model(input_ids=batch[0], attention_mask=batch[1],
                         labels=batch[2], decoder_attention_mask=batch[3], return_dict=True, threshold=threshold).loss

            if args.regularization is not None:
                regu_ = regularization(model=model, mode=args.regularization)
                loss = loss + regu_lambda * regu_

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training=True
                break
            train_losses.append(loss.detach().cpu())
            loss.backward()

            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()    # We have accumulated enought gradients
                scheduler.step()
                model.zero_grad()

            if global_step % args.eval_period == 0:
                model.eval()
                curr_em = inference(args, model if args.n_gpu==1 else model.module, dev_data)
                logger.info("Step %d Train loss %.2f %s %.2f%% on epoch=%d" % (
                        global_step,
                        np.mean(train_losses),
                        dev_data.metric,
                        curr_em*100,
                        epoch))
                train_losses = []
                if best_accuracy < curr_em:
                    model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                    torch.save(model_state_dict, os.path.join(args.output_dir, "best-model.pt"))
                    logger.info("Saving model with best %s: %.2f%% -> %.2f%% on epoch=%d, global_step=%d" % \
                            (dev_data.metric, best_accuracy*100.0, curr_em*100.0, epoch, global_step))
                    best_accuracy = curr_em
                    wait_step = 0
                    stop_training = False
                else:
                    wait_step += 1
                    if wait_step >= args.wait_step:
                        stop_training = True
                        break
                model.train()
        if stop_training:
            break

def inference(args, model, dev_data, save_predictions=False):
    predictions = []
    bos_token_id = dev_data.tokenizer.bos_token_id
    if args.global_topk:
        threshold_mem = None
    with torch.no_grad():
        for i, batch in enumerate(dev_data.dataloader):    
            threshold = args.final_threshold
            if args.global_topk:
                if threshold_mem is None:
                    concat = torch.cat([param.view(-1) for name, param in model.named_parameters() if "mask_scores" in name])
                    n = concat.numel()
                    kth = max(n - (int(n * args.final_threshold) + 1), 1)
                    threshold_mem = concat.kthvalue(kth).values.item()
                threshold = threshold_mem

            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch]
            outputs = model.generate(input_ids=batch[0],
                                     attention_mask=batch[1],
                                     num_beams=dev_data.args.num_beams,
                                     max_length=dev_data.args.max_output_length,
                                     early_stopping=True, threshold=threshold)
            for input_, output in zip(batch[0], outputs):
                pred = dev_data.decode(output)
                predictions.append(pred)
        if save_predictions:
            dev_data.save_predictions(predictions)
        print(predictions[0:10])
        return np.mean(dev_data.evaluate(predictions))

def schedule_threshold(
    step: int,
    total_step: int,
    warmup_steps: int,
    initial_threshold: float,
    final_threshold: float,
    initial_warmup: int,
    final_warmup: int,
    final_lambda: float,
):
    if step <= initial_warmup * warmup_steps:
        threshold = initial_threshold
    elif step > (total_step - final_warmup * warmup_steps):
        threshold = final_threshold
    else:
        spars_warmup_steps = initial_warmup * warmup_steps
        spars_schedu_steps = (final_warmup + initial_warmup) * warmup_steps
        mul_coeff = 1 - (step - spars_warmup_steps) / (total_step - spars_schedu_steps)
        threshold = final_threshold + (initial_threshold - final_threshold) * (mul_coeff ** 3)
    regu_lambda = final_lambda * threshold / final_threshold
    return threshold, regu_lambda


def regularization(model: torch.nn.Module, mode: str):
    regu, counter = 0, 0
    for name, param in model.named_parameters():
        if "mask_scores" in name:
            if mode == "l1":
                regu += torch.norm(torch.sigmoid(param), p=1) / param.numel()
            elif mode == "l0":
                regu += torch.sigmoid(param - 2 / 3 * np.log(0.1 / 1.1)).sum() / param.numel()
            else:
                ValueError("Don't know this mode.")
            counter += 1
    return regu / counter
