
def candidate_selection(wn,
                        token,
                        target_lemma,
                        pos=None,
                        gold_lexkeys=set(),
                        debug=False):
    """
    return candidate synsets of a token

    :param str token: the token
    :param str targe_lemma: a lemma
    :param str pos: supported: n, v, r, a. If None, candidate selection is limited by one pos
    :param str gold_lexkeys: {'congress%1:14:00::'}

    :rtype: tuple
    :return: (candidate_synsets, 
              gold_in_candidates)
    """

    if token.istitle():
        candidate_synsets = wn.synsets(token, pos)

        if not candidate_synsets:
            candidate_synsets = wn.synsets(target_lemma, pos)

    else:
        candidate_synsets = wn.synsets(target_lemma, pos)

    return candidate_synsets



def synset2identifier(synset, wn_version):
    """
    return synset identifier of
    nltk.corpus.reader.wordnet.Synset instance

    :param nltk.corpus.reader.wordnet.Synset synset: a wordnet synset
    :param str wn_version: supported: '171 | 21 | 30'

    :rtype: str
    :return: eng-VERSION-OFFSET-POS (n | v | r | a)
    e.g.
    """
    offset = str(synset.offset())
    offset_8_char = offset.zfill(8)

    pos = synset.pos()
    if pos in {'j', 's'}:
        pos = 'a'

    identifier = 'eng-{wn_version}-{offset_8_char}-{pos}'.format_map(locals())

    return identifier