import os, json
import logging
import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

TRAIN_FILE = "/home/mchen/smp2020ewect/SMP/train/all_train.json"
DEV_FILE = "/home/mchen/smp2020ewect/SMP/eval/all_eval.json"

label_list = ['neural', 'sad', 'happy', 'fear', 'angry', 'surprise']


class InputExample(object):

    def __init__(self, guid, text_a, label):
        self.guid = guid
        self.text_a = text_a
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class SMP2020_ewect(DataProcessor):
    """Base class for data converters for sequence classification data sets."""
    train_file = TRAIN_FILE
    dev_file = DEV_FILE
    test_file = None

    def get_train_examples(self, data_dir=None, filename=None):
        """Gets a collection of `InputExample`s for the train set."""
        if not data_dir:
            data_dir = ""

        input_datas = []
        with open(self.train_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                input_datas = json.loads(line)

        return self._create_examples(input_datas, "train")

    def get_dev_examples(self, data_dir=None, filename=None):
        """Gets a collection of `InputExample`s for the dev set."""
        input_datas = []
        with open(self.dev_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                input_datas = json.loads(line)
                #input_datas.append(data)
        return self._create_examples(input_datas, "dev")

    def get_test_examples(self, data_dir=None, filename=None):
        """Gets a collection of `InputExample`s for the test set."""
        input_datas = []
        with open(self.test_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                input_datas = json.loads(line)
                #input_datas.append(data)
        return self._create_examples(input_datas, "test")

    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples = []
        max_len = 0
        for eid, entry in enumerate((input_data)):
            id = entry['id']
            context = entry['content']
            if len(context)>300:
                max_len+=1
            if is_training:
                label = entry["label"]
            else:
                label = label_list[0]
            example = InputExample(guid=id, text_a=context, label=label)
            examples.append(example)

            if eid < 5:
                print("eid:", eid)
                print("context:", context)
                print("label:", label)
        print(max_len)
        return examples

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return label_list


def glue_convert_examples_to_features(examples, tokenizer, max_length, task='smp2020ewect',
                                      label_list=None, output_mode=None, ):
    if max_length is None:
        max_length = tokenizer.max_len

    if task is not None:
        processor = SMP2020_ewect()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = "classification"
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example):
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer.batch_encode_plus(
        [example.text_a for example in examples],
        max_length=max_length, pad_to_max_length=True)

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        # ['input_ids', 'token_type_ids', 'attention_mask']

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = SMP2020_ewect()
    output_mode = "classification"
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(
            args.data_dir)
        features = glue_convert_examples_to_features(examples,
                                                     tokenizer,
                                                     label_list=label_list,
                                                     max_length=args.max_seq_length,
                                                     output_mode=output_mode,
                                                     # pad_on_left=bool(args.model_type in ['xlnet']),
                                                     # pad on the left for xlnet
                                                     # pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                     # pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                     )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    # if output_mode == "classification":
    if not evaluate:
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    # elif output_mode == "regression":
    #     all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    else:
        all_labels = torch.tensor([0 for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


if __name__ == '__main__':
    SMP2020_ewect().get_train_examples()
    SMP2020_ewect().get_dev_examples()
