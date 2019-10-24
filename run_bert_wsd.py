"""
Usage:
  run_bert_wsd.py (--input_folder=<input_folder> | --input_path=<input_path>) \
  --meanings_path=<meanings_path> \
  --naf_pos=<naf_pos> --use_pos_in_candidate_selection=<use_pos_in_candidate_selection> \
  --verbose=<verbose>

Options:
  --input_folder=<input_folder> input folder (*.naf will be used)
  --input_path=<input_path> path to NAF files (inefficient to only run one but it is possible)
  --meanings_path=<meanings_path> path to where synsets embeddings are stored
  --naf_pos=<naf_pos> NAF parts of speech to consider, e.g., "N-V-G-A"
  --use_pos_in_candidate_selection=<use_pos_in_candidate_selection> if "yes" make use of
  part of speech tag in looking up synsets in WordNet, else use only the lemma
  --verbose=<verbose>


Example:
    python run_bert_wsd.py --input_folder="example_files" --meanings_path="example_files/meanings.bin" \
    --naf_pos="N-V-G-A" --use_pos_in_candidate_selection="yes" --verbose=2
"""
from docopt import docopt
from glob import glob
from datetime import datetime
import pickle

from lxml import etree
from pytorch_pretrained_bert import BertTokenizer, BertModel

from utils import time_in_correct_format
import wsd_datasets_classes
import perform_wsd
import naf_utils

# load arguments
arguments = docopt(__doc__)
print()
print('PROVIDED ARGUMENTS')
print(arguments)
print()

verbose = int(arguments['--verbose'])
wsd_level = 'synset'
resource = 'WN-3.0'
bert_model_variation = 'bert-base-uncased' # in future version, this can be an option
meanings_path = arguments['--meanings_path']
meanings = pickle.load(open(meanings_path, 'rb'))

naf_pos = set(arguments['--naf_pos'].split('-'))
use_pos_in_candidate_selection = arguments['--use_pos_in_candidate_selection'] == 'yes'

# iterable
if arguments['--input_folder']:
    naf_iterable = glob(f'{arguments["--input_folder"]}/*naf')
elif arguments['--input_path']:
    naf_iterable = [arguments['--input_path']]

# load Bert
tokenizer = BertTokenizer.from_pretrained(bert_model_variation, do_lower_case=True)
model = BertModel.from_pretrained(bert_model_variation)

# TODO: ask whether this is ok
#model.cuda()
#model.eval()

for naf_path in naf_iterable:

    output_path = naf_path + '.wsd'

    start_time = time_in_correct_format(datetime.now())

    doc = etree.parse(naf_path)
    naf_obj = wsd_datasets_classes.NAF(doc,
                                       naf_pos_to_consider=naf_pos,
                                       use_pos_in_candidate_selection=use_pos_in_candidate_selection,
                                       verbose=verbose)

    layer_indexes = [-1, -2, -3, -4]
    wsd_df = perform_wsd.perform_wsd_on_test(test_dataframe=naf_obj.df,
                                             model=model,
                                             tokenizer=tokenizer,
                                             meanings=meanings,
                                             layer_indexes=layer_indexes,
                                             use_context_embeddings=False,
                                             without_stop_words=False,
                                             target_word_embeddings_only=True)

    end_time = time_in_correct_format(datetime.now())

    naf_utils.add_wsd_header(doc,
                             start_time=start_time,
                             end_time=end_time)

    naf_utils.enrich_naf_with_wsd_output(doc=doc,
                                         df=wsd_df,
                                         resource=resource)

    doc.write(output_path,
              encoding='utf-8',
              pretty_print=True,
              xml_declaration=True)






