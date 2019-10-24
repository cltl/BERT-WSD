from torch.utils.data import DataLoader, RandomSampler
from wsd_datasets_classes import Token
from bert_input_helper import get_meaning_to_sentence, get_target_index, get_target_indexes
import collections
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import copy



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


def create_target_word_embeddings_from_dataframe(path_to_dataframe, tokenizer, model,
                                                 target_word_vector_method="average",
                                                 final_vector_method="full_list"):

    assert target_word_vector_method == "average" or target_word_vector_method == "sum", \
        "You can only choose between summing the target token word pieces or averaging them!"

    assert final_vector_method == "full_list" or final_vector_method == "average", \
        "You can either choose to leave the target token embeddings " \
        "for a meaning as a list or choose 'average' to create a " \
        "one-to-one mapping between a mapping and its vector"

    dataframe = pd.read_pickle(path_to_dataframe)
    layer_indexes = [-1, -2, -3, -4]
    meanings_to_vec = {}
    total_length = len(dataframe)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tqdm(total=total_length, desc="Creating context embeddings") as pbar:
        for index, instance in dataframe.iterrows():
            sentence = copy.deepcopy(instance.sentence)
            sentence_tokens = copy.deepcopy(instance.sentence_tokens)
            sentence_tokens.insert(0, Token(text='[CLS]', token_id='unknown'))
            sentence_tokens.append(Token(text='[SEP]', token_id='unknown'))
            new_wsd_tokens = get_new_wsd_tokens(sentence_tokens, tokenizer)
            target_indexes = get_target_indexes(instance.token_ids[0], new_wsd_tokens, 0)
            gold_meanings = copy.deepcopy(instance.source_wn_engs)

            tokenized_sentence = tokenizer.tokenize(sentence)

            tokenized_sentence.insert(0, '[CLS]')
            tokenized_sentence.append('[SEP]')
            input_ids = np.asarray(tokenizer.convert_tokens_to_ids(tokenized_sentence)) \
                .reshape(1, len(tokenized_sentence))

            input_mask = [1] * len(tokenized_sentence)
            input_mask = np.asarray(input_mask).reshape(1, len(tokenized_sentence))
            input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
            input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)

            with torch.no_grad():
                all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
                all_encoder_layers = all_encoder_layers

            all_out_features = []
            for i, token in enumerate(tokenized_sentence):
                all_layers = []
                for j, layer_index in enumerate(layer_indexes):
                    layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()
                    layers = collections.OrderedDict()
                    layers["index"] = layer_index
                    layers["values"] = [
                        round(x.item(), 6) for x in layer_output[0][i]
                    ]
                    all_layers.append(layers)
                out_features = collections.OrderedDict()
                out_features["token"] = token
                out_features["layers"] = all_layers
                all_out_features.append(out_features)

            token_average_list = list()
            for feature_index, feature in enumerate(all_out_features):
                layers = feature["layers"]
                layer_values = []
                for layer in layers:
                    values = layer['values']
                    layer_values.append(values)

                context_vector_values = np.sum(layer_values, axis=0)
                token_average_list.append(context_vector_values)

            temp_list = []
            for token_index, token_vector in enumerate(token_average_list):
                if token_index in target_indexes:
                    temp_list.append(token_vector)

            for meaning in gold_meanings:
                assert len(temp_list) > 0, "Temp list is empty at {}".format(meaning + "_" + str(index))

                for item in temp_list:
                    assert isinstance(item, np.ndarray), "Temp list has nan vector(s) for {}".format(
                        meaning + "_" + str(index))

                if target_word_vector_method == "average":
                    context_vector = np.average(temp_list, axis=0)
                elif target_word_vector_method == "sum":
                    context_vector = np.sum(temp_list, axis=0)

                if meaning in meanings_to_vec:
                    meanings_to_vec[meaning].append(context_vector)
                else:
                    meanings_to_vec[meaning] = [context_vector]

            pbar.update(1)

    if final_vector_method == "averaging":
        for meaning, vec_list in meanings_to_vec:
            meanings_to_vec[meaning] = np.average(vec_list, axis=0)

    return meanings_to_vec



def create_target_word_embeddings_from_textfile(path_to_file, tokenizer, model,
                                                target_word_vector_method="average",
                                                final_vector_method="full_list"):

    assert target_word_vector_method == "average" or target_word_vector_method == "sum", \
        "You can only choose between summing the target token word pieces or averaging them!"

    assert final_vector_method == "full_list" or final_vector_method == "average", \
        "You can either choose to leave the target token embeddings " \
        "for a meaning as a list or choose 'average' to create a " \
        "one-to-one mapping between a mapping and its vector"

    meaning_to_sentence = get_meaning_to_sentence(path_to_corpus=path_to_file)

    layer_indexes = [-1, -2, -3, -4]
    meanings_to_vec = {}
    total_length = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for meaning, sentence_target_dict_list in meaning_to_sentence.items():
        total_length += len(sentence_target_dict_list)

    with tqdm(total=total_length, desc="Creating context embeddings") as pbar:
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
                new_wsd_tokens = get_new_wsd_tokens(sentence_tokens, tokenizer)
                target_indexes = get_target_indexes("target", new_wsd_tokens, 0)

                tokenized_sentence = tokenizer.tokenize(' '.join(sentence))
                tokenized_sentence.insert(0, '[CLS]')
                tokenized_sentence.append('[SEP]')
                input_ids = np.asarray(tokenizer.convert_tokens_to_ids(tokenized_sentence)) \
                    .reshape(1, len(tokenized_sentence))
                input_mask = [1] * len(tokenized_sentence)
                input_mask = np.asarray(input_mask).reshape(1, len(tokenized_sentence))
                input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
                input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)

                with torch.no_grad():
                    all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
                    all_encoder_layers = all_encoder_layers

                all_out_features = []
                for i, token in enumerate(tokenized_sentence):
                    all_layers = []
                    for j, layer_index in enumerate(layer_indexes):
                        layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()
                        layers = collections.OrderedDict()
                        layers["index"] = layer_index
                        layers["values"] = [
                            round(x.item(), 6) for x in layer_output[0][i]
                        ]
                        all_layers.append(layers)
                    out_features = collections.OrderedDict()
                    out_features["token"] = token
                    out_features["layers"] = all_layers
                    all_out_features.append(out_features)

                token_average_list = list()
                for feature_index, feature in enumerate(all_out_features):
                    layers = feature["layers"]
                    layer_values = []
                    for layer in layers:
                        values = layer['values']
                        layer_values.append(values)

                    context_vector_values = np.sum(layer_values, axis=0)
                    token_average_list.append(context_vector_values)

                temp_list = []
                for token_index, token_vector in enumerate(token_average_list):
                    if token_index in target_indexes:
                        temp_list.append(token_vector)

                assert len(temp_list) > 0, "Temp list is empty at {}".format(meaning+"_"+str(sentence_index))

                for item in temp_list:
                    assert isinstance(item, np.ndarray), "Temp list has nan vector(s) for {}".format(meaning+"_"+str(sentence_index))

                if target_word_vector_method == "average":
                    context_vector = np.average(temp_list, axis=0)
                elif target_word_vector_method == "sum":
                    context_vector = np.sum(temp_list, axis=0)


                if meaning in meanings_to_vec:
                    meanings_to_vec[meaning].append(context_vector)
                else:
                    meanings_to_vec[meaning] = [context_vector]

                pbar.update(1)

    if final_vector_method == "averaging":
        for meaning, vec_list in meanings_to_vec:
            meanings_to_vec[meaning] = np.average(vec_list, axis=0)

    return meanings_to_vec


def create_context_embeddings_from_textfile(path_to_file, tokenizer, model,
                                            get_target_word_embedding_only, is_context_embedding,
                                            vector_method="full_list"):

    if get_target_word_embedding_only:
        print("Please note that since only the target word embedding will be used to represent a meaning, "
              "then the paramater 'is_context_embedding' will be ignored ")

    meaning_to_sentence = get_meaning_to_sentence(path_to_corpus=path_to_file)
    layer_indexes = [-1, -2, -3, -4]
    meanings_to_vec = {}
    total_length = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for meaning, sentence_target_dict_list in meaning_to_sentence.items():
        total_length += len(sentence_target_dict_list)

    with tqdm(total=total_length, desc="Creating context embeddings") as pbar:
        for meaning, sentence_target_dict_list in meaning_to_sentence.items():
            for sentence_target_dict in sentence_target_dict_list:
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
                new_wsd_tokens = get_new_wsd_tokens(sentence_tokens, tokenizer)
                target_indexes = get_target_index("target", new_wsd_tokens, 0)

                tokenized_sentence = tokenizer.tokenize(' '.join(sentence))
                tokenized_sentence.insert(0, '[CLS]')
                tokenized_sentence.append('[SEP]')
                input_ids = np.asarray(tokenizer.convert_tokens_to_ids(tokenized_sentence))\
                    .reshape(1, len(tokenized_sentence))
                input_mask = [1] * len(tokenized_sentence)
                input_mask = np.asarray(input_mask).reshape(1, len(tokenized_sentence))
                input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
                input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)

                with torch.no_grad():
                    all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
                    all_encoder_layers = all_encoder_layers

                all_out_features = []
                for i, token in enumerate(tokenized_sentence):
                    all_layers = []
                    for j, layer_index in enumerate(layer_indexes):
                        layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()
                        layers = collections.OrderedDict()
                        layers["index"] = layer_index
                        layers["values"] = [
                            round(x.item(), 6) for x in layer_output[0][i]
                        ]
                        all_layers.append(layers)
                    out_features = collections.OrderedDict()
                    out_features["token"] = token
                    out_features["layers"] = all_layers
                    all_out_features.append(out_features)

                token_average_list = list()
                for feature_index, feature in enumerate(all_out_features):
                    token = feature['token']

                    if token == '[CLS]' or token == '[SEP]' or (feature_index in target_indexes) \
                            and is_context_embedding and not get_target_word_embedding_only:
                        continue

                    layers = feature["layers"]
                    layer_values = []
                    for layer in layers:
                        values = layer['values']
                        layer_values.append(values)

                    context_vector_values = np.sum(layer_values, axis=0)
                    token_average_list.append(context_vector_values)

                    if not is_context_embedding and not get_target_word_embedding_only and token == '[CLS]':
                        break

                if is_context_embedding and not get_target_word_embedding_only:
                    context_vector = np.average(token_average_list, axis=0)
                elif not is_context_embedding and not get_target_word_embedding_only:
                    context_vector = token_average_list[0]
                elif get_target_word_embedding_only:
                    temp_list = []
                    for token_index, token_vector in enumerate(token_average_list):
                        if token_index in target_indexes:
                            temp_list.append(token_vector)
                    context_vector = np.average(temp_list, axis=0)
                if not isinstance(context_vector, np.ndarray):
                    print()
                if meaning in meanings_to_vec:
                    meanings_to_vec[meaning].append(context_vector)
                else:
                    meanings_to_vec[meaning] = [context_vector]

                pbar.update(1)

    if vector_method == "averaging":
        for meaning, vec_list in meanings_to_vec:
            meanings_to_vec[meaning] = np.average(vec_list, axis=0)

    return meanings_to_vec


def get_meaning_to_features_list(path_to_corpus, model, tokenizer):

    meaning_to_sentence = get_meaning_to_sentence(path_to_corpus=path_to_corpus)
    layer_indexes = [-1, -2, -3, -4]
    meanings_to_vec = {}
    total_length = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for meaning, sentence_target_dict_list in meaning_to_sentence.items():
        total_length += len(sentence_target_dict_list)

    with tqdm(total=total_length, desc="Creating features") as pbar:
        for meaning, sentence_target_dict_list in meaning_to_sentence.items():
            for sentence_target_dict in sentence_target_dict_list:
                sentence = sentence_target_dict["sentence"]
                tokenized_sentence = tokenizer.tokenize(' '.join(sentence))
                tokenized_sentence.insert(0, '[CLS]')
                tokenized_sentence.append('[SEP]')
                input_ids = np.asarray(tokenizer.convert_tokens_to_ids(tokenized_sentence)) \
                    .reshape(1, len(tokenized_sentence))
                input_mask = [1] * len(tokenized_sentence)
                input_mask = np.asarray(input_mask).reshape(1, len(tokenized_sentence))
                input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
                input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)

                with torch.no_grad():
                    all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
                    all_encoder_layers = all_encoder_layers

                all_out_features = []
                for i, token in enumerate(tokenized_sentence):
                    all_layers = []
                    for j, layer_index in enumerate(layer_indexes):
                        layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()
                        layers = collections.OrderedDict()
                        layers["index"] = layer_index
                        layers["values"] = [
                            round(x.item(), 6) for x in layer_output[0][i]
                        ]
                        all_layers.append(layers)
                    out_features = collections.OrderedDict()
                    out_features["token"] = token
                    out_features["layers"] = all_layers
                    all_out_features.append(out_features)

                features_list = list()
                for feature_index, feature in enumerate(all_out_features):
                    token = feature["token"]

                    if token == '[CLS]' or token == '[SEP]':
                        continue

                    layers = feature["layers"]
                    layer_values = []
                    for layer in layers:
                        values = layer['values']
                        layer_values.append(values)

                    context_vector_values = np.sum(layer_values, axis=0)
                    features_list.append(context_vector_values)

                features_list = np.asarray(features_list)
                if meaning in meanings_to_vec:
                    meanings_to_vec[meaning].append(features_list)
                else:
                    meanings_to_vec[meaning] = [features_list]

                pbar.update(1)

    return meanings_to_vec


def create_embeddings_from_context_dataset(context_dataset, model, tokenizer, is_context_embedding,
                                           vector_method="average",
                                           bert_variation="BertModel", batch_size=32):

    assert bert_variation == "BertForMaskedLM" or bert_variation == "BertModel", \
        "BERT model variation must be 'BertForMaskedLM' or 'BertModel' from the pytorch_pretrained_bert module!"

    train_sampler = RandomSampler(context_dataset)
    train_dataloader = DataLoader(context_dataset, sampler=train_sampler, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_indexes = [-1, -2, -3, -4]
    meanings_to_vec = dict()

    with tqdm(total=len(train_dataloader), desc="Creating context embeddings") as pbar:
        for step, batch in enumerate(train_dataloader):
            input_ids, input_mask, segment_ids, target_indexes, meanings = batch
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            target_indexes = target_indexes.detach().cpu().tolist()
            meanings = meanings
            with torch.no_grad():
                all_encoded_layers, pooled_output = model(input_ids, segment_ids, input_mask)

            for input_ids_index, input_id in enumerate(input_ids.detach().cpu().tolist()):
                all_out_features = []
                tokenized_sentence = tokenizer.convert_ids_to_tokens(input_id)
                for i, token in enumerate(tokenized_sentence):
                    all_layers = []
                    for j, layer_index in enumerate(layer_indexes):
                        layer_output = all_encoded_layers[int(layer_index)].detach().cpu().numpy()
                        layers = collections.OrderedDict()
                        layers["index"] = layer_index
                        layers["values"] = [
                            round(x.item(), 6) for x in layer_output[0][i]
                        ]
                        all_layers.append(layers)
                    out_features = collections.OrderedDict()
                    out_features["token"] = token
                    out_features["layers"] = all_layers
                    all_out_features.append(out_features)

                token_average_list = list()
                for feature_index, feature in enumerate(all_out_features):
                    token = feature['token']
                    if (token == '[CLS]' or token == '[SEP]' or (feature_index in target_indexes[input_ids_index][0])) \
                            and is_context_embedding:
                        continue

                    if token == '[PAD]':
                        break

                    layers = feature["layers"]
                    layer_values = []
                    for layer in layers:
                        values = layer['values']
                        layer_values.append(values)

                    context_vector_values = np.sum(layer_values, axis=0)
                    token_average_list.append(context_vector_values)
                    if not is_context_embedding and token == '[CLS]':
                        break

                if is_context_embedding:
                    context_vector = np.average(token_average_list, axis=0)
                else:
                    context_vector = token_average_list[0]

                meaning = meanings[input_ids_index]
                if meaning in meanings_to_vec:
                    meanings_to_vec[meaning].append(context_vector)
                else:
                    meanings_to_vec[meaning] = [context_vector]
            pbar.update(1)

    if vector_method == "average":
        for meaning, vec_list in meanings_to_vec.items():
            meanings_to_vec[meaning] = np.average(vec_list, axis=0)

    return meanings_to_vec


def create_context_embeddings_from_dataframe(dataframe, tokenizer, model):
    layer_indexes = [-1, -2, -3, -4]
    meanings = {}
    for index, instance in dataframe.iterrows():
        sentence = copy.deepcopy(instance.sentence)
        source_wn_engs = copy.deepcopy(instance.source_wn_engs)
        original_sentence_tokens = copy.deepcopy(instance.sentence_tokens)
        original_sentence_tokens.insert(0, Token(token_id="unknown", text="[CLS]"))
        original_sentence_tokens.append(Token(token_id="unknown", text="[SEP]"))
        n_wsd_tokens = get_new_wsd_tokens(wsd_tokens=original_sentence_tokens, tokenizer=tokenizer)
        target_indexes = get_target_index(instance.token_ids[0], n_wsd_tokens, index)

        tokenized_sentence = tokenizer.tokenize(sentence)
        tokenized_sentence.insert(0, '[CLS]')
        tokenized_sentence.append('[SEP]')
        input_ids = np.asarray(tokenizer.convert_tokens_to_ids(tokenized_sentence)).reshape(1, len(tokenized_sentence))
        input_mask = [1] * len(tokenized_sentence)
        input_mask = np.asarray(input_mask).reshape(1, len(tokenized_sentence))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)

        with torch.no_grad():
            all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
            all_encoder_layers = all_encoder_layers

        all_out_features = []
        for i, token in enumerate(tokenized_sentence):
            all_layers = []
            for j, layer_index in enumerate(layer_indexes):
                layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()
                layers = collections.OrderedDict()
                layers["index"] = layer_index
                layers["values"] = [
                    round(x.item(), 6) for x in layer_output[0][i]
                ]
                all_layers.append(layers)
            out_features = collections.OrderedDict()
            out_features["token"] = token
            out_features["layers"] = all_layers
            all_out_features.append(out_features)

        token_average_list = list()
        for feature_index, feature in enumerate(all_out_features):
            token = feature['token']

            if token == '[CLS]' or token == '[SEP]' or (feature_index in target_indexes):
                continue

            layers = feature["layers"]
            layer_values = []
            for layer in layers:
                values = layer['values']
                layer_values.append(values)

            summed_values = np.sum(layer_values, axis=0)
            token_average_list.append(summed_values)

        context_vector = np.average(token_average_list, axis=0)
        for source_wn_eng in source_wn_engs:
            if source_wn_eng in meanings:
                meanings[source_wn_eng].append(context_vector)
            else:
                meanings[source_wn_eng] = [context_vector]
    return meanings


def get_targetword_embedding_per_sentence(tokenized_sentence, tokenizer, model, target_index_list, layer_index_list
                                          , device,  vector_method="average"):

    assert vector_method == "average" or vector_method == "sum",\
        "The paramater 'vector_method' should be set to 'average' (default) or 'sum'"

    input_ids = np.asarray(tokenizer.convert_tokens_to_ids(tokenized_sentence)).reshape(1, len(tokenized_sentence))
    input_mask = [1] * len(tokenized_sentence)
    input_mask = np.asarray(input_mask).reshape(1, len(tokenized_sentence))
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)

    with torch.no_grad():
        all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
        all_encoder_layers = all_encoder_layers

    all_out_features = []
    for i, token in enumerate(tokenized_sentence):
        all_layers = []
        for j, layer_index in enumerate(layer_index_list):
            layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()
            layers = collections.OrderedDict()
            layers["index"] = layer_index
            layers["values"] = [
                round(x.item(), 6) for x in layer_output[0][i]
            ]
            all_layers.append(layers)
        out_features = collections.OrderedDict()
        out_features["token"] = token
        out_features["layers"] = all_layers
        all_out_features.append(out_features)

    token_average_list = list()
    for feature_index, feature in enumerate(all_out_features):
        layers = feature["layers"]
        layer_values = []
        for layer in layers:
            values = layer['values']
            layer_values.append(values)

        summed_values = np.sum(layer_values, axis=0)
        token_average_list.append(summed_values)
    target_word_vectors = []
    for token_index, token_vector in enumerate(token_average_list):
        if token_index in target_index_list:
            target_word_vectors.append(token_vector)

    if len(target_word_vectors) == 0:
        print()
    if vector_method == "average":
        target_word_vector = np.average(target_word_vectors, axis=0)
    elif vector_method == "sum":
        target_word_vector = np.sum(target_word_vectors, axis=0)

    return target_word_vector


def get_context_vector_per_sentence(tokenized_sentence, tokenizer, model,
                                    target_index_list, layer_index_list, is_context_embedding):

    input_ids = np.asarray(tokenizer.convert_tokens_to_ids(tokenized_sentence)).reshape(1, len(tokenized_sentence))
    input_mask = [1] * len(tokenized_sentence)
    input_mask = np.asarray(input_mask).reshape(1, len(tokenized_sentence))
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.long)

    with torch.no_grad():
        all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask, )
        all_encoder_layers = all_encoder_layers

    all_out_features = []
    for i, token in enumerate(tokenized_sentence):
        all_layers = []
        for j, layer_index in enumerate(layer_index_list):
            layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()
            layers = collections.OrderedDict()
            layers["index"] = layer_index
            layers["values"] = [
                round(x.item(), 6) for x in layer_output[0][i]
            ]
            all_layers.append(layers)
        out_features = collections.OrderedDict()
        out_features["token"] = token
        out_features["layers"] = all_layers
        all_out_features.append(out_features)

    token_average_list = list()
    for feature_index, feature in enumerate(all_out_features):
        token = feature['token']

        if (token == '[CLS]' or token == '[SEP]' or (feature_index in target_index_list)) and is_context_embedding:
            continue

        layers = feature["layers"]
        layer_values = []
        for layer in layers:
            values = layer['values']
            layer_values.append(values)

        summed_values = np.sum(layer_values, axis=0)
        token_average_list.append(summed_values)

        if not is_context_embedding and token == '[CLS]':
            break

    if is_context_embedding:
        context_vector = np.average(token_average_list, axis=0)
    else:
        context_vector = token_average_list[0]
    return context_vector