from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import os
import random
import sys
import io

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_transformers import AdamW, WarmupLinearSchedule
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from seqeval.metrics import classification_report
from model.xlmr_for_token_classification import XLMRForTokenClassification
from utils.train_utils import add_xlmr_args, evaluate_model, predict_model
from utils.data_utils import NerProcessor, create_dataset, convert_examples_to_features, create_features_from_conllup

from tqdm import tqdm as tqdm
from tqdm import trange

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def main():
    parser = argparse.ArgumentParser()
    parser = add_xlmr_args(parser)

    args = parser.parse_args()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval and not args.do_predict and not args.server:
        raise ValueError(
            "At least one of `do_train`, `do_eval`, `do_predict` or `server` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processor = NerProcessor(labels=args.labels)
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1  # add one for IGNORE label

    train_examples = None
    num_train_optimization_steps = 0

    print(args.train_batch_size)
    print(args.gradient_accumulation_steps)

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    
    # preparing model configs
    hidden_size = 768 if 'base' in args.pretrained_path else 1024 # TODO: move this inside model.__init__

    device = 'cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu'

    # creating model
    model = XLMRForTokenClassification(pretrained_path=args.pretrained_path,
                                       n_labels=num_labels, hidden_size=hidden_size, seq_len=args.max_seq_length,
                                       dropout_p=args.dropout, device=device, use_norm=args.use_norm, use_li=args.use_li, 
                                       li_dropout_p=args.li_dropout, li_sigma=args.li_sigma)

    model.to(device)
    no_decay = ['bias', 'final_layer_norm.weight']

    
    params = list(model.named_parameters())

    optimizer_grouped_parameters = [
        {'params': [p for n, p in params if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in params if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)

    # freeze model if necessary
    if args.freeze_model:
        logger.info("Freezing XLM-R model...")
        for n, p in model.named_parameters():
            if 'xlmr' in n and p.requires_grad:
                p.requires_grad = False


    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    if args.do_train:

        output_p_file = os.path.join(args.output_dir, "parameters.txt")
        with open(output_p_file, "w") as writer:
            for k,v in sorted(vars(args).items()): 
                writer.write("{0}: {1}\n".format(k,v))

        if len(args.train_existing_model)>0:
            load_model_path=os.path.join(args.train_existing_model, 'model.pt')
            logger.info("Loading model {}".format(load_model_path))
            state_dict = torch.load(open(load_model_path, 'rb'),  map_location=torch.device(device))
            model.load_state_dict(state_dict, False) # do not enforce keys to be present
            logger.info("Loaded saved model")


        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, model.encode_word)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        train_data = create_dataset(train_features)

        train_sampler = RandomSampler(train_data)
    
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        # getting validation samples
        val_examples = processor.get_dev_examples(args.data_dir)
        val_features = convert_examples_to_features(
            val_examples, label_list, args.max_seq_length, model.encode_word)

        val_data = create_dataset(val_features)
        
        best_val_f1 = 0.0

        bestEpoch=0

        for currentEpoch in tqdm(range(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            tbar = tqdm(train_dataloader, desc="Iteration")
            
            model.train()
            for step, batch in enumerate(tbar):
                batch = tuple(t.to(device) for t in batch)
                input_ids, label_ids, l_mask, valid_ids, not_used = batch
                loss = model(input_ids, label_ids, l_mask, valid_ids)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                tbar.set_description('Loss = %.4f' %(tr_loss / (step+1)))

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
            
            logger.info("\nTesting on validation set...")
            f1, report = evaluate_model(model, val_data, label_list, args.eval_batch_size, device)
            if f1 > best_val_f1:
                best_val_f1 = f1
                bestEpoch=currentEpoch
                logger.info("\nFound better f1=%.4f on validation set. Saving model\n" %(f1))
                logger.info("%s\n" %(report))
                
                torch.save(model.state_dict(), open(os.path.join(args.output_dir, 'model.pt'), 'wb'))

                with open(os.path.join(args.output_dir, 'model.epoch'),'w') as outf:
                    outf.write('{}'.format(currentEpoch))
            
            else :
                logger.info("\nNo better F1 score: {}. Best F1 was in epoch {} = {}\n".format(f1,bestEpoch,best_val_f1))
    else: # load a saved model
        load_model_path=os.path.join(args.output_dir, 'model.pt')
        logger.info("Loading model {}".format(load_model_path))
        state_dict = torch.load(open(load_model_path, 'rb'),  map_location=torch.device(device))
        model.load_state_dict(state_dict)
        logger.info("Loaded saved model")

    model.to(device)

    if args.do_eval:
        if args.eval_on == "dev":
            eval_examples = processor.get_dev_examples(args.data_dir)
        elif args.eval_on == "test":
            eval_examples = processor.get_test_examples(args.data_dir)
        else:
            raise ValueError("eval on dev or test set only")
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, model.encode_word)
        
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        
        eval_data = create_dataset(eval_features)
        f1_score, report = evaluate_model(model, eval_data, label_list, args.eval_batch_size, device)

       
        logger.info("\n%s", report)
        output_eval_file = os.path.join(args.output_dir, "eval_{}_results.txt".format(args.eval_on))
        with open(output_eval_file, "w") as writer:
            logger.info("***** Writing results to file *****")
            writer.write(report)
            logger.info("Done.")

    if args.do_predict:
        if args.predict_on == "dev":
            eval_examples = processor.get_dev_examples(args.data_dir)
            eval_sentences = processor.get_dev_sentences(args.data_dir)
        elif args.predict_on == "test":
            eval_examples = processor.get_test_examples(args.data_dir)
            eval_sentences = processor.get_test_sentences(args.data_dir)
        elif args.predict_on == "deploy":
            eval_examples = processor.get_deploy_examples(args.data_dir)
            eval_sentences = processor.get_deploy_sentences(args.data_dir)
        else:
            raise ValueError("Predict on dev, test or deploy set only")
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, model.encode_word)
        
        logger.info("***** Running prediction *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Num sentences = %d", len(eval_sentences))
        logger.info("  Batch size = %d", args.predict_batch_size)
        logger.info("  Writing predictions to file [{}]".format(args.predict_filename));
        
        eval_data = create_dataset(eval_features)
        prediction = predict_model(model, eval_data, label_list, args.predict_batch_size, device)

        output_eval_file = os.path.join(args.output_dir, args.predict_filename)
        with open(output_eval_file, "w") as writer:
            logger.info("***** Writing results to file *****")
            for (i,sent) in enumerate(eval_sentences):
                for (j,tok) in enumerate(sent[0]):
                    pred="O"
                    if j>=len(prediction[i]):
                        logger.info("WARNING: Not enough tokens predicted in sentence {} ({}/{})".format(i,len(prediction[i]),len(sent[0])))
                    else:
                        pred=prediction[i][j]

                    if args.predict_format == "ann_only":
                        writer.write("{}\n".format(pred))
                    else:
                        writer.write("{} {}\n".format(tok,pred))
                writer.write("\n")
            logger.info("Done.")

    if args.server:
        from flask import Flask,request,jsonify

        app = Flask(__name__)

        @app.route("/api/v1.0/ner", methods=["GET","POST"])
        def ner():
            if request.method=="POST":
                textf=request.files['text']
                text=textf.read().decode('utf-8',errors='ignore')
            else:
                text=request.args.get("text")
                text=text.encode("latin1",errors='ignore').decode("utf8",errors='ignore')

            data = processor.get_deploy_examples_from_text(text,args.lang)
            doc=data["doc"]
            eval_examples = data["examples"]
            eval_features = convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, model.encode_word)
            eval_data = create_dataset(eval_features)
            prediction = predict_model(model, eval_data, label_list, args.predict_batch_size, device)
            
            #print(prediction)

            s=-1
            t=-1
            currentTok=None
            lastTok=None
            currentType=None
            r=[]
            eid=1
            for token in doc:
                if token.is_sent_start:
                    s+=1
                    t=-1

                t+=1

                if s<len(prediction) and t<len(prediction[s]):
                    pred=prediction[s][t]
                else:
                    pred="O"
                
                if pred.startswith("B-") or pred=="O" or t==0:
                    if currentTok!=None:
                        #r+="T{0}\t{1} {2} {3}\t{4}\n".format(eid,currentType,currentTok.idx,lastTok.idx+len(lastTok.text),text[currentTok.idx:lastTok.idx+len(lastTok.text)])
                        r.append({"id":"T%d"%(eid),"type":currentType,"start":currentTok.idx,"end":lastTok.idx+len(lastTok.text),"text":text[currentTok.idx:lastTok.idx+len(lastTok.text)]})
                        currentTok=None
                        currentType=None
                        eid+=1

                if pred.startswith("B-"):
                    currentTok=token
                    currentType=pred[2:]

                lastTok=token

            return jsonify({'status':'OK','result':r})

        def get_input_data(expected_values):
            if "input" not in request.values:
                return False, None, jsonify({"status": "ERROR", "message": "Missing input parameter"})

            try:
                data = json.loads(request.values["input"])
            except json.JSONDecodeError:
                return False, None, jsonify({"status": "ERROR", "message": "Invalid JSON provided in the input parameter"})

            if data is None:
                return False, None, jsonify(
                    {"status": "ERROR", "message": "Invalid input JSON provided in the input parameter"})

            for v in expected_values:
                if v not in data:
                    return False, None, jsonify({"status": "ERROR",
                                                "message": "Invalid input JSON provided in the input parameter. Missing field {value}".format(
                                                    value=v)})

            return True, data, None


        @app.route("/process", methods=["GET","POST"])
        def ner_process():
            status, data, error = get_input_data(["input","output"])
            if not status: return error

            input_file = data["input"]
            output_file = data["output"]

            r = create_features_from_conllup(input_file, label_list, args.max_seq_length, model.encode_word)
            eval_features=r['features']
            conllup=r['conllup']
            eval_data = create_dataset(eval_features)
            prediction = predict_model(model, eval_data, label_list, args.predict_batch_size, device, True)
            
            #print(prediction)

            with open(output_file,"w") as fout:
                lastTokenId=0
                forceStop=False
                for p in prediction:
                    if forceStop: break
                    for y in p:
                        tokId=y['token_id']
                        label=y['label']

                        if tokId<lastTokenId: continue

                        while lastTokenId<tokId and lastTokenId<len(conllup):
                            fout.write(conllup[lastTokenId])
                            fout.write("\n")
                            lastTokenId+=1
                        
                        if lastTokenId>=len(conllup):
                            forceStop=True
                            break

                        fout.write("{}\t{}\n".format(conllup[lastTokenId],label))
                        lastTokenId+=1

            return jsonify({'status':'OK','message':''})


        @app.route("/process_conllup", methods=["GET","POST"])
        def ner_process_conllup():
            input_file=request.values["input"];

            separator="\t"
            if "separator" in request.values: separator=request.values["separator"]

            column=1
            if "column" in request.values: column=int(request.values["column"])

            comment="#"
            if "comment" in request.values: comment=request.values["comment"]

            r = create_features_from_conllup(input_file, label_list, args.max_seq_length, model.encode_word,conllup_column_separator=separator, conllup_column_word=column,conllup_comment=comment, input_is_file=False)
            eval_features=r['features']
            conllup=r['conllup']
            eval_data = create_dataset(eval_features)
            prediction = predict_model(model, eval_data, label_list, args.predict_batch_size, device, True)
            
            #print(prediction)

            output_file=""
            with io.StringIO() as fout:
                lastTokenId=0
                forceStop=False
                for p in prediction:
                    if forceStop: break
                    for y in p:
                        tokId=y['token_id']
                        label=y['label']

                        if tokId<lastTokenId: continue

                        while lastTokenId<tokId and lastTokenId<len(conllup):
                            fout.write(conllup[lastTokenId])
                            fout.write("\n")
                            lastTokenId+=1
                        
                        if lastTokenId>=len(conllup):
                            forceStop=True
                            break

                        fout.write("{}{}{}\n".format(conllup[lastTokenId],separator,label))
                        lastTokenId+=1

                fout.seek(0)
                output_file=fout.read();

            return jsonify({'status':'OK','message':'','output':output_file})


        app.run(threaded=False, debug=False, host="127.0.0.1", port=args.server_port)

if __name__ == "__main__":
    main()
