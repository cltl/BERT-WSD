from lxml import etree
from collections import defaultdict


def add_wsd_header(doc, start_time, end_time):
    """
    add WSD header to NAF

    :param lxml.etree._ElementTree doc: NAF file loaded with etree.parse()

    """
    naf_header = doc.find('nafHeader')

    ling_proc = etree.SubElement(naf_header, "linguisticProcessors")
    ling_proc.set("layer", "WSD")
    lp = etree.SubElement(ling_proc, "lp")
    lp.set("beginTimestamp", start_time)
    lp.set('endTimestamp', end_time)
    lp.set('name', "BERT-WSD")
    lp.set('version', "1.0")


def enrich_naf_with_wsd_output(doc, df, resource):
    """

    :param lxml.etree._ElementTree doc: NAF file loaded with etree.parse()
    :param df: wsd_datasets_classes.NAF.df
    """
    # load dict mapping t1 -> synset identifier -> confidence
    tid_to_synsetid_to_conf = defaultdict(dict)
    for index, row in df.iterrows():
        tid = row['token_ids'][0]
        synset = row['bert_output']
        confidence = row['chosen_meaning_confidence']
        tid_to_synsetid_to_conf[tid][synset] = str(confidence)

    # loop and update
    for term_el in doc.xpath('terms/term'):
        t_id = term_el.get('id')

        if t_id in tid_to_synsetid_to_conf:
            ext_refs_el = etree.SubElement(term_el, 'externalReferences')

            for meaning, confidence in tid_to_synsetid_to_conf[t_id].items():
                etree.SubElement(ext_refs_el,
                                 'externalRef',
                                 attrib={'resource' : resource,
                                         'reference' : meaning,
                                         'confidence' : confidence})




if __name__ == "__main__":
    import pickle
    from lxml import etree
    from datetime import datetime

    import naf_utils
    from utils import time_in_correct_format

    start_time = time_in_correct_format(datetime.now())

    # load NAF
    doc = etree.parse('example_files/World Chess Championship 1984.naf')
    df = pickle.load(open('example_files/test.df.wsd', 'rb'))
    resource = 'WN-3.0'

    end_time = time_in_correct_format(datetime.now())

    naf_utils.add_wsd_header(doc,
                             start_time=start_time,
                             end_time=end_time)

    enrich_naf_with_wsd_output(doc, df, resource)


