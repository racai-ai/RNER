import argparse

from torch.utils.data import SequentialSampler, DataLoader
from seqeval.metrics import f1_score, classification_report
import torch

def parseLabels(string):
    return string.split(",")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def add_xlmr_args(parser):
     """
     Adds training and validation arguments to the passed parser
     """

     parser.add_argument("--data_dir",
                         default=None,
                         type=str,
                         required=True,
                         help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
     parser.add_argument("--pretrained_path", default=None, type=str, required=True,
                         help="pretrained XLM-Roberta model path")
     parser.add_argument("--task_name",
                         default=None,
                         type=str,
                         required=True,
                         help="The name of the task to train.")
     parser.add_argument("--output_dir",
                         default=None,
                         type=str,
                         required=True,
                         help="The output directory where the model predictions and checkpoints will be written.")
     parser.add_argument("--train_existing_model",
                         default="",
                         type=str,
                         help="Directory for existing model used to start training.")
     # Other parameters
     parser.add_argument("--cache_dir",
                         default="",
                         type=str,
                         help="Where do you want to store the pre-trained models downloaded from s3")
     parser.add_argument("--max_seq_length",
                         default=128,
                         type=int,
                         help="The maximum total input sequence length after WordPiece tokenization. \n"
                              "Sequences longer than this will be truncated, and sequences shorter \n"
                              "than this will be padded.")
     parser.add_argument("--num_seq",
                         default=1,
                         type=int,
                         help="Number of sequences (to allow for longer documents).")
     parser.add_argument("--do_train",
                         action='store_true',
                         help="Whether to run training.")

     parser.add_argument("--debug",
                         action='store_true',
                         help="Turn on debug mode.")

     parser.add_argument("--do_eval",
                         action='store_true',
                         help="Whether to run eval or not.")
     parser.add_argument("--eval_on",
                         default="dev",
                         help="Whether to run eval on the dev set or test set.")
     parser.add_argument("--eval_batch_size",
                         default=32,
                         type=int,
                         help="Total batch size for eval.")

     parser.add_argument("--do_predict",
                         action='store_true',
                         help="Whether to run prediction or not.")
     parser.add_argument("--predict_on",
                         default="test",
                         help="Dataset to run prediction on.")
     parser.add_argument("--predict_batch_size",
                         default=32,
                         type=int,
                         help="Total batch size for prediction.")
     parser.add_argument("--predict_format",
                         default="complete",
                         help="Prediction format ann_only/complete.")
     parser.add_argument("--predict_filename",
                         default="predict_results.txt",
                         help="Filename for prediction results.")

     parser.add_argument("--do_lower_case",
                         action='store_true',
                         help="Set this flag if you are using an uncased model.")
     parser.add_argument("--train_batch_size",
                         default=32,
                         type=int,
                         help="Total batch size for training.")
     parser.add_argument("--learning_rate",
                         default=5e-5,
                         type=float,
                         help="The initial learning rate for Adam.")
     parser.add_argument("--num_train_epochs",
                         default=3,
                         type=int,
                         help="Total number of training epochs to perform.")
     parser.add_argument("--warmup_proportion",
                         default=0.1,
                         type=float,
                         help="Proportion of training to perform linear learning rate warmup for. "
                              "E.g., 0.1 = 10%% of training.")
     parser.add_argument("--weight_decay", default=0.01, type=float,
                         help="Weight deay if we apply some.")
     parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                         help="Epsilon for Adam optimizer.")
     parser.add_argument("--max_grad_norm", default=1.0, type=float,
                         help="Max gradient norm.")
     parser.add_argument("--no_cuda",
                         action='store_true',
                         help="Whether not to use CUDA when available")
     parser.add_argument('--seed',
                         type=int,
                         default=42,
                         help="random seed for initialization")
     parser.add_argument('--gradient_accumulation_steps',
                         type=int,
                         default=1,
                         help="Number of updates steps to accumulate before performing a backward/update pass.")
     parser.add_argument('--fp16',
                         action='store_true',
                         help="Whether to use 16-bit float precision instead of 32-bit")
     parser.add_argument('--fp16_opt_level', type=str, default='O1',
                         help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                              "See details at https://nvidia.github.io/apex/amp.html")
     parser.add_argument('--loss_scale',
                         type=float, default=0,
                         help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                              "0 (default value): dynamic loss scaling.\n"
                              "Positive power of 2: static loss scaling value.\n")
     parser.add_argument('--dropout', 
                         type=float, default=0.3,
                         help = "training dropout probability")
     
     parser.add_argument('--freeze_model', 
                         action='store_true', default=False,
                         help = "whether to freeze the XLM-R base model and train only the classification heads")

     parser.add_argument('--labels', 
                        type=parseLabels,
                         default=["en","ro"],
                         help = "Labels to use for training")
     parser.add_argument('--ner_labels',
                        type=parseLabels,
                         default=["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"],
                         help = "Labels to use for training")
     parser.add_argument('--pretrained_ner', 
                        type=parseLabels,
                         default=[],
                         help = "Pretrained NER models")
     parser.add_argument("--pretrained_lang_detect",
                         default="",
                         help="Pretrained lang_detect model.")

     parser.add_argument('--use_norm', 
                         action="store_true",
                         help = "Use batchnorm1d")

     parser.add_argument('--use_li',
                         type=str2bool,default=True,
                         help="Use LateralInhibition layer")

     parser.add_argument('--li_dropout',
                         type=float, default=0.1,
                         help="Dropout probability for the LateralInhibition layer")

     parser.add_argument('--li_sigma',
                         type=float, default=10,
                         help="Parameter for approximating the Heaviside function in the backward pass")

     parser.add_argument('--server',
                         action='store_true', default=False,
                         help="Run server")

     parser.add_argument('--server_port',
                         type=int, default=5111,
                         help="Port for running the server")

     parser.add_argument('--lang',
                         default="ro",
                         help="Language used for parsing raw text (only if tokenized input is not available)")
                                   
     return parser


def evaluate_model(model, eval_dataset, label_list, batch_size, device, task):
     """
     Evaluates an NER model on the eval_dataset provided.
     Returns:
          F1_score: Macro-average f1_score on the evaluation dataset.
          Report: detailed classification report 
     """

     # Run prediction for full data
     eval_sampler = SequentialSampler(eval_dataset)
     eval_dataloader = DataLoader(
          eval_dataset, sampler=eval_sampler, batch_size=batch_size)

     model.eval() # turn of dropout

     y_true = []
     y_pred = []

     if task=="lang_detect":
        label_map = {i: label for i, label in enumerate(label_list, 0)}
     else:
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        label_map[0]="IGNORE"
     print(label_list)

     for input_ids, label_ids, l_mask, valid_ids in eval_dataloader:

          input_ids = input_ids.to(device)
          label_ids = label_ids.to(device)

          valid_ids = valid_ids.to(device)
          l_mask = l_mask.to(device)

          with torch.no_grad():
               logits = model(input_ids, labels=None, labels_mask=None,
                              valid_mask=valid_ids)


          if task=="lang_detect":
            logits = torch.argmax(logits, dim=1) # era 2 pt ner
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.cpu().numpy()

            #print(label_ids)

            for i, cur_label in enumerate(label_ids):
               #print(cur_label)
               temp_1 = []
               temp_2 = []

               temp_1.append(label_map[cur_label])
               temp_2.append(label_map[logits[i]])

               assert len(temp_1) == len(temp_2)
               y_true.append(temp_1)
               y_pred.append(temp_2)
          else:
            logits = torch.argmax(logits, dim=2) # era 2 pt ner
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.cpu().numpy()

            for i, cur_label in enumerate(label_ids):
               temp_1 = []
               temp_2 = []

               for j, m in enumerate(cur_label):
                    if valid_ids[i][j]:  # if it's a valid label
                         temp_1.append(label_map[m])
                         temp_2.append(label_map[logits[i][j]])

               assert len(temp_1) == len(temp_2)
               y_true.append(temp_1)
               y_pred.append(temp_2)

     report = classification_report(y_true, y_pred, digits=4)
     f1 = f1_score(y_true, y_pred, average='Macro')

     return f1, report

def predict_model(model, predict_dataset, label_list, batch_size, device, task):
     """
     Executes a NER model on the predict_dataset provided and returns predictions.
     Returns:
          predictions: Predictions on the predict_dataset
     """

     # Run prediction for full data
     predict_sampler = SequentialSampler(predict_dataset)
     predict_dataloader = DataLoader(
          predict_dataset, sampler=predict_sampler, batch_size=batch_size)

     model.eval() # turn of dropout

     y_pred = []

     if task=="lang_detect":
        label_map = {i: label for i, label in enumerate(label_list, 0)}
     else:
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        #label_map[0]="IGNORE" # This should not occur during predict
     print(label_list)

     for input_ids, label_ids, l_mask, valid_ids in predict_dataloader:

          input_ids = input_ids.to(device)
          valid_ids = valid_ids.to(device)

          with torch.no_grad():
               logits = model(input_ids, labels=None, labels_mask=None,
                              valid_mask=valid_ids)

          if task=="lang_detect":
            logits = torch.argmax(logits, dim=1) # era 2 pt ner
            logits = logits.detach().cpu().numpy()

            for i, cur_label in enumerate(logits):
               temp_2 = []
               temp_2.append(label_map[logits[i]])
               y_pred.append(temp_2)
          else:
            logits = torch.argmax(logits, dim=2)
            logits = logits.detach().cpu().numpy()

            for i, cur_label in enumerate(logits):
               temp_2 = []

               for j, m in enumerate(cur_label):
                    if valid_ids[i][j]:  # if it's a valid label
                         if logits[i][j]==0: temp_2.append('O')
                         else: temp_2.append(label_map[logits[i][j]]) # pt lang detect nu are 2 dimensiuni

               y_pred.append(temp_2)

     return y_pred
