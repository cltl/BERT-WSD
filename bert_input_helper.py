from torch.utils.data import Dataset
from wsd_datasets_classes import Token
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import torch
import copy


def truncate_sequence(tokens_a, max_num_tokens):
    """Truncates a sequences to a maximum sequence length.
    Lifted from Google's BERT repo and modified to truncate only one sequence."""
    while True:
        total_length = len(tokens_a)
        if total_length <= max_num_tokens:
            break

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random.random() < 0.5:
            del tokens_a[0]
        else:
            tokens_a.pop()
    return tokens_a


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, tokens_a):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.tokens_a = tokens_a


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_example_to_feature(example, max_seq_length, tokenizer, logger):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :param vocab_list: list, list containing vocabulary keys (words and sub-words)
    :param logger: Logging object used to log from the calling script
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    tokens_a = example.tokens_a
    tokens_a = truncate_sequence(tokens_a=tokens_a,
                                 max_num_tokens=max_seq_length - 2)
    tokens_a.insert(0, '[CLS]')
    tokens_a.append('[SEP]')

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if example.guid < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("tokens: %s" % " ".join(
            [str(x) for x in tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info(
            "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids)
    return features


class DocumentDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, logger):
        self.vocab = tokenizer.vocab
        self.vocab_list = list(self.vocab.keys())
        self.tokenizer = tokenizer
        self.corpus_path = corpus_path
        self.logger = logger
        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file

        # load samples into memory
        self.doc = []
        corpus_df = pd.read_pickle(corpus_path)
        self.corpus_lines = len(corpus_df)  # number of non-empty lines in input corpus

        for index, instance in tqdm(corpus_df.iterrows(), desc="Loading Dataset", total=len(corpus_df)):
            self.doc.append(instance.sentence)

        self.input_ids = []
        self.input_masks = []
        self.segment_ids = []

        for line_index, line in enumerate(self.doc):
            tokens = self.tokenizer.tokenize(line)
            input_example = InputExample(guid=line_index, tokens_a=tokens)
            input_feature = convert_example_to_feature(example=input_example,
                                                       max_seq_length=73,
                                                       tokenizer=self.tokenizer,
                                                       logger=self.logger)

            self.input_ids.append(input_feature.input_ids)
            self.input_masks.append(input_feature.input_mask)
            self.segment_ids.append(input_feature.segment_ids)

        self.input_ids = np.asarray(self.input_ids)
        self.input_masks = np.asarray(self.input_masks)
        self.segment_ids = np.asarray(self.segment_ids)

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)))


class ContextDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, logger):
        self.input_ids = []
        self.input_masks = []
        self.segment_ids = []
        self.target_indexes = []
        self.meanings = []
        self.logger = logger
        self.corpus_lines = 0

        meaning_to_sentence = get_meaning_to_sentence(path_to_corpus=corpus_path)

        for meaning, sentence_target_dict_list in meaning_to_sentence.items():
            for sentence_index, sentence_target_dict in enumerate(sentence_target_dict_list):
                sentence = sentence_target_dict["sentence"]
                sentence_tokens = []
                target_index = sentence_target_dict["target_index"]
                for tok_index, tok in enumerate(sentence):
                    token_text = tok
                    token_pos = 'n'
                    token_lemma = 'unknown'
                    if tok_index == target_index:
                        token_id = "target"
                    else:
                        token_id = "unknown"
                    sentence_tokens.append(Token(token_id=token_id, text=token_text, pos=token_pos, lemma=token_lemma))
                sentence_tokens.insert(0, Token(text='[CLS]', token_id='unknown'))
                sentence_tokens.append(Token(text='[SEP]', token_id='unknown'))
                new_wsd_tokens = get_new_wsd_tokens(wsd_tokens=sentence_tokens, tokenizer=tokenizer)
                target_index_list = get_target_indexes(target_token_id="target", wsd_tokens=new_wsd_tokens,
                                                       df_index=meaning+"_"+str(sentence_index))
                tokenized_sentence = tokenizer.tokenize(' '.join(sentence))
                example = InputExample(guid=self.corpus_lines, tokens_a=tokenized_sentence)
                feature = convert_example_to_feature(example=example, tokenizer=tokenizer,
                                                     max_seq_length=73, logger=self.logger)

                target_index_list = pad_sequences([target_index_list], padding="post", value=-1, maxlen=73)

                self.input_ids.append(feature.input_ids)
                self.input_masks.append(feature.input_mask)
                self.segment_ids.append(feature.segment_ids)
                self.target_indexes.append(target_index_list)
                self.meanings.append(meaning)
                self.corpus_lines += 1

        self.segment_ids = np.asarray(self.segment_ids)
        self.input_masks = np.asarray(self.input_masks)
        self.input_ids = np.asarray(self.input_ids)
        self.target_indexes = np.asarray(self.target_indexes)
        self.meanings = np.asarray(self.meanings)

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                self.target_indexes[item],
                self.meanings[item])


def get_target_indexes(target_token_id, wsd_tokens, df_index):
    t_indexes = []
    for index2, token_obj in enumerate(wsd_tokens):
        if token_obj.token_id == target_token_id:
            t_indexes.append(index2)
    assert len(t_indexes) > 0, 'no target token index for %s' % df_index
    return t_indexes


def get_target_index(target_token_id, wsd_tokens, df_index):
    target_index = None
    for index2, token_obj in enumerate(wsd_tokens):
        if token_obj.token_id == target_token_id:
            target_index = index2
            break
    assert isinstance(target_index, int), 'no target token index for %s' % df_index
    return target_index


def get_new_wsd_tokens(wsd_tokens, tokenizer):
    new_wsd_tokens = []
    for tok in wsd_tokens:
        split_text = tokenizer.tokenize(tok.text)
        if len(split_text) > 1:
            for text in split_text:
                new_wsd_tokens.append(Token(token_id=tok.token_id, text=text, pos=tok.pos, lemma=tok.lemma))
        else:
            new_wsd_tokens.append(Token(token_id=tok.token_id, text=tok.text, pos=tok.pos, lemma=tok.lemma))
    return new_wsd_tokens


def get_meanings_from_context_corpus(path_to_file):
    # "./resources/lstm_input_linux.txt"
    meanings = set()
    with open(path_to_file, 'r') as f:
        for line in f:
            line = line.split()
            for token in line[1:]:
                if '---eng' in token:
                    meaning_token = token
                    meaning_token = meaning_token.split('-')
                    meaning_token = '-'.join(meaning_token[3:])
                    meanings.add(meaning_token)
                    break
        f.close()
    return meanings


def get_lemma_per_meaning(meanings, dataframe_list):
    meaning_to_lemma = dict()
    for meaning in meanings:
        found_meaning = False
        for dataframe in dataframe_list:
            for index, instance in dataframe.iterrows():
                if meaning in instance.candidate_meanings:
                    meaning_to_lemma[meaning] = instance.target_lemma
                    found_meaning = True
                    break
            if found_meaning:
                break
    return meaning_to_lemma


def get_meaning_to_sentence(path_to_corpus):
    meaning_to_sentence = dict()
    # "./resources/lstm_input_linux.txt"
    with open(path_to_corpus, 'r') as f:
        for line in f:
            line = copy.deepcopy(line).split()
            for token_index, token in enumerate(line):
                if "---" in token:
                    meaning = '-'.join(token.split('-')[3:])
                    if token_index == 2:
                        target_index = 0
                    else:
                        target_index = token_index - 2
                    del line[token_index]
                    if meaning in meaning_to_sentence:
                        meaning_to_sentence[meaning].append({"sentence": line[1:], "target_index": target_index})
                    else:
                        meaning_to_sentence[meaning] = [{"sentence": line[1:], "target_index": target_index}]
                    break
        f.close()
    return meaning_to_sentence