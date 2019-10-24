from create_context_embeddings import get_context_vector_per_sentence, get_targetword_embedding_per_sentence
from bert_input_helper import get_target_indexes, get_target_index, get_new_wsd_tokens
from pytorch_pretrained_bert import BertTokenizer, BertModel
from wsd_datasets_classes import Token
from nltk.corpus import stopwords
from collections import Counter
from scipy import spatial
import pandas as pd
import operator
import torch
import copy


def get_similarity_differences(test_dataframe):
    final_dict = dict()
    for index, instance in test_dataframe.iterrows():
        final_dict[index] = []
        bert_output = instance.bert_output
        source_engs = instance.source_wn_engs
        chosen_conf = instance.chosen_meaning_confidence
        meaning_to_confidence = instance.meaning2confidence
        for source in source_engs:
            final_dict[index].append(
                {"chosen_meaning": bert_output, "chosen_meaning_confidence": meaning_to_confidence[bert_output],
                 "source_meaning": source, "source_confidence": meaning_to_confidence[source],
                 "confidence_difference": chosen_conf - meaning_to_confidence[source]})

    return final_dict


def perform_wsd_on_test(test_dataframe, meanings, model, tokenizer,
                        layer_indexes, use_context_embeddings, without_stop_words, target_word_embeddings_only):
    test_dataframe["bert_output"] = [None for _ in range(len(test_dataframe))]
    test_dataframe["meaning2confidence"] = [None for _ in range(len(test_dataframe))]
    test_dataframe["wsd_strategy"] = [None for _ in range(len(test_dataframe))]
    test_dataframe['chosen_meaning_confidence'] = [None for _ in range(len(test_dataframe))]
    stop_words = set(stopwords.words('english'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for index, instance in test_dataframe.iterrows():
        sentence = copy.deepcopy(instance.sentence)
        sentence_tokens = copy.deepcopy(instance.sentence_tokens)
        target_index = get_target_index(instance.token_ids[0], sentence_tokens, index)
        if without_stop_words:
            temp_sentence_tokens = []
            for token_index, token in enumerate(sentence_tokens):
                if not (token.text in stop_words) or token_index == target_index:
                    temp_sentence_tokens.append(token)
            sentence_tokens = copy.deepcopy(temp_sentence_tokens)

        sentence_tokens.insert(0, Token(text='[CLS]', token_id='unknown'))
        sentence_tokens.append(Token(text='[SEP]', token_id='unknown'))
        new_sentence_tokens = get_new_wsd_tokens(wsd_tokens=sentence_tokens, tokenizer=tokenizer)
        tokenized_sentence = tokenizer.tokenize(sentence)

        tokenized_sentence.insert(0, '[CLS]')
        tokenized_sentence.append('[SEP]')

        target_indexes = get_target_indexes(instance.token_ids[0], new_sentence_tokens, index)

        if not target_word_embeddings_only:
            context_vector = get_context_vector_per_sentence(tokenized_sentence=tokenized_sentence,
                                                             tokenizer=tokenizer,
                                                             model=model,
                                                             target_index_list=target_indexes,
                                                             layer_index_list=layer_indexes,
                                                             is_context_embedding=use_context_embeddings)
        else:
            context_vector = get_targetword_embedding_per_sentence(tokenized_sentence=tokenized_sentence,
                                                                   tokenizer=tokenizer,
                                                                   model=model,
                                                                   target_index_list=target_indexes,
                                                                   layer_index_list=layer_indexes,
                                                                   device=device)

        candidate_meanings = copy.deepcopy(instance.candidate_meanings)
        found_meaning = False
        meaning_similarities = dict()
        for candidate_meaning in candidate_meanings:
            meaning_similarities[candidate_meaning] = []
            if candidate_meaning in meanings:
                found_meaning = True
                similarity = 1 - spatial.distance.cosine(meanings[candidate_meaning], context_vector)
                meaning_similarities[candidate_meaning].append(similarity)
            else:
                meaning_similarities[candidate_meaning] = float(0)

        wsd_strategy = "bert"

        for meaning, similarity_list in meaning_similarities.items():
            if isinstance(similarity_list, list):
                similarity_list.sort(reverse=True)
                meaning_similarities[meaning] = similarity_list[0]

        if found_meaning:
            sorted_meanings = sorted(meaning_similarities.items(), key=operator.itemgetter(1), reverse=True)
            same_confidence = [i for i, v in enumerate(sorted_meanings) if v[1] == sorted_meanings[0][1]]
            if len(same_confidence) > 1 and 0 in same_confidence:
                temp_list = list()
                for idx in same_confidence:
                    temp_list.append(sorted_meanings[idx][0])
                temp_dict = {}
                for m in temp_list:
                    sense_rank = candidate_meanings.index(m)
                    temp_dict[m] = sense_rank
                sorted_ranks = sorted(temp_dict.items(), key=operator.itemgetter(1))
                bert_output = sorted_ranks[0][0]
            else:
                bert_output = sorted_meanings[0][0]
            chosen_meaning_confidence = sorted_meanings[0][1]
        else:
            bert_output = candidate_meanings[0]
            chosen_meaning_confidence = meaning_similarities[candidate_meanings[0]]
            wsd_strategy = "mfs_fallback"

        if len(candidate_meanings) == 1:
            bert_output = candidate_meanings[0]
            wsd_strategy = "monosemous"
            chosen_meaning_confidence = meaning_similarities[candidate_meanings[0]]


        test_dataframe.at[index, 'bert_output'] = bert_output
        test_dataframe.at[index, 'wsd_strategy'] = wsd_strategy
        test_dataframe.at[index, "meaning2confidence"] = meaning_similarities
        test_dataframe.at[index, "chosen_meaning_confidence"] = chosen_meaning_confidence

    return test_dataframe


if __name__ == "__main__":
    wsd_df_path = 'example_files/test.df'
    save_path = 'example_files/test.df.wsd'
    wsd_level = 'synset'
    meanings_path = 'TODO'

    wsd_df = pd.read_pickle(wsd_df_path)

    #meanings = pd.read_pickle(meanings_path)
    meanings = {}

    bert_model_variation = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_model_variation, do_lower_case=True)
    model = BertModel.from_pretrained(bert_model_variation)

    # TODO: ask whether this is ok
    #model.load_state_dict(torch.load(model_path))

    # TODO: ask whether this is ok
    #model.cuda()
    #model.eval()

    layer_indexes = [-1, -2, -3, -4]
    wsd_df = perform_wsd_on_test(test_dataframe=wsd_df,
                                 model=model,
                                 tokenizer=tokenizer,
                                 meanings=meanings,
                                 layer_indexes=layer_indexes,
                                 use_context_embeddings=False,
                                 without_stop_words=False,
                                 target_word_embeddings_only=True)

    wsd_df.to_pickle(save_path)

    wsd_strategies = Counter(wsd_df['wsd_strategy'])

    print("Size of dataset: ", len(wsd_df))
    print("Overall WSD strategies: ", wsd_strategies)
