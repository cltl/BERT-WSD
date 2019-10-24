from collections import defaultdict

import pandas
from nltk.corpus import wordnet as wn

import wn_utils


naf_pos_to_wn_pos = {
        'N' : 'n',
        'V' : 'v',
        'G' : 'a',
        'A' : 'r'
    }

class Token:
    def __init__(self, token_id, text, pos=None, lemma=None):
        self.token_id = token_id
        self.text = text
        self.pos = pos
        self.lemma = lemma


class NAF:


    def __init__(self,
                 doc,
                 naf_pos_to_consider={'N', 'V', 'G', 'A'},
                 use_pos_in_candidate_selection=True,
                 wn_version='30',
                 verbose=0,
                 ):
        self.naf_pos_to_consider = naf_pos_to_consider
        self.use_pos_in_candidate_selection = use_pos_in_candidate_selection
        self.verbose = verbose

        self.doc_name = self.get_doc_name(doc)

        self.sentid2token_objs = self.load_sentid2token_objs(doc)

        self.df = self.create_df(wn_version)

    def __str__(self):
        info = []

        attrs = ['doc_name']

        for attr in attrs:
            info.append(f'KEY: {attr}: {getattr(self, attr)}')

        return '\n'.join(info)


    def get_doc_name(self, doc):
        return doc.find('nafHeader/fileDesc').get('title')


    def load_sentid2token_objs(self, doc):
        """

        :param lxml.etree._ElementTree doc: NAF file loaded with etree.parse()

        :rtype: dict
        :return: mapping of sent_id -> list of Token objs
        """

        sentid2token_objs = defaultdict(list)

        wf_els = doc.xpath('text/wf')
        term_els = doc.xpath('terms/term')

        # for now, I assume that NAF files should have the same number of wf and term elements
        assert len(wf_els) == len(term_els), f'mismatch in number of wf and term elements'

        ignored_pos = set()
        for wf_el, term_el in zip(wf_els, term_els):
            token_id = term_el.get('id')
            text = wf_el.text
            lemma = term_el.get('lemma')
            sent_id = int(wf_el.get('sent'))

            naf_pos = term_el.get('pos')
            if naf_pos not in self.naf_pos_to_consider:
                ignored_pos.add(naf_pos)

            wn_pos = naf_pos_to_wn_pos.get(naf_pos, None)

            if wn_pos is None:
                if self.verbose >= 3:
                    print(f'could not map {naf_pos} to {wn_pos}')

            token_obj = Token(token_id=token_id,
                              text=text,
                              pos=wn_pos,
                              lemma=lemma)
            sentid2token_objs[sent_id].append(token_obj)


        if self.verbose >= 2:
            print(f'skipped pos labels: {ignored_pos}')
            print(f'found {len(sentid2token_objs)} sentences')

        return sentid2token_objs


    def create_df(self, wn_version):

        list_of_lists = []
        headers = ['doc_name', 'pos',
                   'sentence', 'sentence_tokens',
                   'target_lemma', 'token',
                   'token_ids',
                   'candidate_meanings']

        for sent_id, token_objs in self.sentid2token_objs.items():

            sentence = ' '.join([token_obj.text
                                 for token_obj in token_objs])

            for token_obj in token_objs:

                if self.use_pos_in_candidate_selection:
                    if not token_obj.pos:
                        continue

                # TODO: precompute
                if self.use_pos_in_candidate_selection:
                    the_pos = token_obj.pos
                else:
                    the_pos = None
                synsets = wn_utils.candidate_selection(wn=wn,
                                                       token=token_obj.text,
                                                       target_lemma=token_obj.lemma,
                                                       pos=the_pos)
                synset_identifiers = [wn_utils.synset2identifier(synset, wn_version)
                                      for synset in synsets]


                if synset_identifiers:
                    one_row = [self.doc_name,
                               token_obj.pos,
                               sentence,
                               token_objs,
                               token_obj.lemma,
                               token_obj.text,
                               [token_obj.token_id],
                               synset_identifiers]
                    list_of_lists.append(one_row)


        df = pandas.DataFrame(list_of_lists, columns=headers)
        return df

if __name__ == '__main__':
    import pickle
    from lxml import etree
    from datetime import datetime

    import naf_utils
    from utils import time_in_correct_format

    start_time = time_in_correct_format(datetime.now())

    # load NAF
    doc = etree.parse('example_files/World Chess Championship 1984.naf')
    naf_obj = NAF(doc, verbose=2)

    # TODO: process with BERT

    end_time = time_in_correct_format(datetime.now())

    naf_utils.add_wsd_header(doc,
                             start_time=start_time,
                             end_time=end_time
                             )

    with open('example_files/test.df', 'wb') as outfile:
        pickle.dump(naf_obj.df, outfile)
    # TODO: enrich NAF




