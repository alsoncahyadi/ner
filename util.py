import os
import string
import re
import ast
import dill
from nltk.stem.snowball import SnowballStemmer
from nltk.chunk import conlltags2tree
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

NLTK_BASE_PATH = "/home/alson/Desktop/ner/NLTK"
GMB_IOB_TRIAD_PATH = NLTK_BASE_PATH + "/obj/gmb-iob-triad.pkl"
GMB_IOB_NLTK_FORMAT_PATH = NLTK_BASE_PATH + "/obj/gmb-iob-nltk-format.pkl"
GMB_IOB_TREE_PATH = NLTK_BASE_PATH + "/obj/gmb-iob-tree.pkl"
GMB_RANDOM_5000_TRAIN_PATH = NLTK_BASE_PATH + "/obj/random/data/data_gmb_5000_random_train.pkl"
GMB_RANDOM_2000_TEST_PATH = NLTK_BASE_PATH + "/obj/random/data/data_gmb_2000_random_test.pkl"
ID_IOB_TRIAD_PATH = NLTK_BASE_PATH + "/obj/id/data/id_words+tags+iobs.pkl"
ID_POS_TAGGED_PATH = NLTK_BASE_PATH + "/obj/id/data/id_words+tags.pkl"



def get_named_entities_from_file(filename):
    myfile = open(filename, "r")
    sentences = myfile.readlines()
    named_entities = []
    
    for sentence in sentences:
        data = ast.literal_eval(sentence)
        named_entities.append(data)

    return named_entities

def features(tokens, index, history):
    """
    `tokens`  = a POS-tagged sentence [(w1, t1), ...]
    `index`   = the index of the token we want to extract features for
    `history` = the previous predicted IOB tags
    """
 
    # init the stemmer
    stemmer = SnowballStemmer('english')
 
    # Pad the sequence with placeholders
    tokens = [('[START2]', '[START2]'), ('[START1]', '[START1]')] + list(tokens) + [('[END1]', '[END1]'), ('[END2]', '[END2]')]
    history = ['[START2]', '[START1]'] + list(history)
 
    # shift the index with 2, to accommodate the padding
    index += 2
 
    word, pos = tokens[index]
    prevword, prevpos = tokens[index - 1]
    prevprevword, prevprevpos = tokens[index - 2]
    nextword, nextpos = tokens[index + 1]
    nextnextword, nextnextpos = tokens[index + 2]
    previob = history[index - 1]
    contains_dash = '-' in word
    contains_dot = '.' in word
    allascii = all([True for c in word if c in string.ascii_lowercase])
 
    allcaps = word == word.capitalize()
    capitalized = word[0] in string.ascii_uppercase
 
    prevallcaps = prevword == prevword.capitalize()
    prevcapitalized = prevword[0] in string.ascii_uppercase
 
    nextallcaps = prevword == prevword.capitalize()
    nextcapitalized = prevword[0] in string.ascii_uppercase
 
    return {
        'word': word,
        'lemma': stemmer.stem(word),
        'pos': pos,
        'all-ascii': allascii,
 
        'next-word': nextword,
        'next-lemma': stemmer.stem(nextword),
        'next-pos': nextpos,
 
        'next-next-word': nextnextword,
        'nextnextpos': nextnextpos,
 
        'prev-word': prevword,
        'prev-lemma': stemmer.stem(prevword),
        'prev-pos': prevpos,
 
        'prev-prev-word': prevprevword,
        'prev-prev-pos': prevprevpos,
 
        'prev-iob': previob,
 
        'contains-dash': contains_dash,
        'contains-dot': contains_dot,
 
        'all-caps': allcaps,
        'capitalized': capitalized,
 
        'prev-all-caps': prevallcaps,
        'prev-capitalized': prevcapitalized,
 
        'next-all-caps': nextallcaps,
        'next-capitalized': nextcapitalized,
    }

def to_conll_iob(annotated_sentence):
    """
    `annotated_sentence` = list of triplets [(w1, t1, iob1), ...]
    Transform a pseudo-IOB notation: O, PERSON, PERSON, O, O, LOCATION, O
    to proper IOB notation: O, B-PERSON, I-PERSON, O, O, B-LOCATION, O
    """
    proper_iob_tokens = []
    for idx, annotated_token in enumerate(annotated_sentence):
        tag, word, ner = annotated_token
 
        if ner != 'O':
            if idx == 0:
                ner = "B-" + ner
            elif annotated_sentence[idx - 1][2] == ner:
                ner = "I-" + ner
            else:
                ner = "B-" + ner
        proper_iob_tokens.append((tag, word, ner))
    return proper_iob_tokens
 
 
def read_gmb(corpus_root):
    for root, dirs, files in os.walk(corpus_root):
        for filename in files:
            if filename.endswith(".tags"):
                with open(os.path.join(root, filename), 'rb') as file_handle:
                    file_content = file_handle.read().decode('utf-8').strip()
                    annotated_sentences = file_content.split('\n\n')
                    for annotated_sentence in annotated_sentences:
                        annotated_tokens = [seq for seq in annotated_sentence.split('\n') if seq]
 
                        standard_form_tokens = []
 
                        for idx, annotated_token in enumerate(annotated_tokens):
                            annotations = annotated_token.split('\t')
                            word, tag, ner = annotations[0], annotations[1], annotations[3]
 
                            if ner != 'O':
                                ner = ner.split('-')[0]
 
                            if tag in ('LQU', 'RQU'):   # Make it NLTK compatible
                                tag = "``"
 
                            standard_form_tokens.append((word, tag, ner))
 
                        conll_tokens = to_conll_iob(standard_form_tokens)
 
                        # Make it NLTK Classifier compatible - [(w1, t1, iob1), ...] to [((w1, t1), iob1), ...]
                        # Because the classfier expects a tuple as input, first item input, second the class
                        yield [((w, t), iob) for w, t, iob in conll_tokens]

def read_gmb_as_iob_triads(corpus_root):
    for root, dirs, files in os.walk(corpus_root):
        for filename in files:
            if filename.endswith(".tags"):
                with open(os.path.join(root, filename), 'rb') as file_handle:
                    file_content = file_handle.read().decode('utf-8').strip()
                    annotated_sentences = file_content.split('\n\n')
                    for annotated_sentence in annotated_sentences:
                        annotated_tokens = [seq for seq in annotated_sentence.split('\n') if seq]
 
                        standard_form_tokens = []
 
                        for idx, annotated_token in enumerate(annotated_tokens):
                            annotations = annotated_token.split('\t')
                            word, tag, ner = annotations[0], annotations[1], annotations[3]
 
                            if ner != 'O':
                                ner = ner.split('-')[0]
 
                            if tag in ('LQU', 'RQU'):   # Make it NLTK compatible
                                tag = "``"
 
                            standard_form_tokens.append((word, tag, ner))
 
                        conll_tokens = to_conll_iob(standard_form_tokens)
 
                        # Make it NLTK Classifier compatible - [(w1, t1, iob1), ...] to [((w1, t1), iob1), ...]
                        # Because the classfier expects a tuple as input, first item input, second the class
                        yield conll_tokens

def read_gmb_ner(corpus_root):
    for root, dirs, files in os.walk(corpus_root):
        for filename in files:
            if filename.endswith(".tags"):
                with open(os.path.join(root, filename), 'rb') as file_handle:
                    file_content = file_handle.read().decode('utf-8').strip()
                    annotated_sentences = file_content.split('\n\n')
                    for annotated_sentence in annotated_sentences:
                        annotated_tokens = [seq for seq in annotated_sentence.split('\n') if seq]
 
                        standard_form_tokens = []
 
                        for idx, annotated_token in enumerate(annotated_tokens):
                            annotations = annotated_token.split('\t')
                            word, tag, ner = annotations[0], annotations[1], annotations[3]
 
                            if ner != 'O':
                                ner = ner.split('-')[0]
 
                            standard_form_tokens.append((word, tag, ner))
 
                        conll_tokens = to_conll_iob(standard_form_tokens)
                        yield conlltags2tree(conll_tokens)
 
def shape(word):
    word_shape = 'other'
    if re.match('[0-9]+(\.[0-9]*)?|[0-9]*\.[0-9]+$', word):
        word_shape = 'number'
    elif re.match('\W+$', word):
        word_shape = 'punct'
    elif re.match('[A-Z][a-z]+$', word):
        word_shape = 'capitalized'
    elif re.match('[A-Z]+$', word):
        word_shape = 'uppercase'
    elif re.match('[a-z]+$', word):
        word_shape = 'lowercase'
    elif re.match('[A-Z][a-z]+[A-Z][a-z]+[A-Za-z]*$', word):
        word_shape = 'camelcase'
    elif re.match('[A-Za-z]+$', word):
        word_shape = 'mixedcase'
    elif re.match('__.+__$', word):
        word_shape = 'wildcard'
    elif re.match('[A-Za-z0-9]+\.$', word):
        word_shape = 'ending-dot'
    elif re.match('[A-Za-z0-9]+\.[A-Za-z0-9\.]+\.$', word):
        word_shape = 'abbreviation'
    elif re.match('[A-Za-z0-9]+\-[A-Za-z0-9\-]+.*$', word):
        word_shape = 'contains-hyphen'
 
    return word_shape

from nltk.stem.snowball import SnowballStemmer
 
 
stemmer =  {
    'en': SnowballStemmer('english'),
    'id': StemmerFactory().create_stemmer()
}
 
LANGUAGE_CODE = 'id'
def ner_features(tokens, index, history):
    """
    `tokens`  = a POS-tagged sentence [(w1, t1), ...]
    `index`   = the index of the token we want to extract features for
    `history` = the previous predicted IOB tags
    """
 
    # Pad the sequence with placeholders
    tokens = [('__START2__', '__START2__'), ('__START1__', '__START1__')] + list(tokens) + [('__END1__', '__END1__'), ('__END2__', '__END2__')]
    history = ['__START2__', '__START1__'] + list(history)
 
    # shift the index with 2, to accommodate the padding
    index += 2
 
    word, pos = tokens[index]
    prevword, prevpos = tokens[index - 1]
    prevprevword, prevprevpos = tokens[index - 2]
    nextword, nextpos = tokens[index + 1]
    nextnextword, nextnextpos = tokens[index + 2]
    previob = history[-1]
    prevpreviob = history[-2]
 
    return {
        'word': word,
        # 'lemma': stemmer[LANGUAGE_CODE].stem(word),
        'pos': pos,
        'pos[:2]': pos[:2],
        'shape': shape(word),
        'word.lower()': str(word.lower()),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),

        '+1:word': nextword,
        '+1:pos': nextpos,
        # '+1:lemma': stemmer[LANGUAGE_CODE].stem(nextword),
        '+1:shape': shape(nextword),
        '+1:word.lower()': nextword.lower(),
        '+1:word.isupper()': nextword.isupper(),
        '+1:word.istitle()': nextword.istitle(),
        '+1:word.isdigit()': nextword.isdigit(),

        '+2:word': nextnextword,
        '+2:pos': nextnextpos,
        # '+2:lemma': stemmer[LANGUAGE_CODE].stem(nextnextword),
        '+2:shape': shape(nextnextword),
        '+2:word.lower()': nextnextword.lower(),
        '+2:word.isupper()': nextnextword.isupper(),
        '+2:word.istitle()': nextnextword.istitle(),
        '+2:word.isdigit()': nextnextword.isdigit(),

        '-1:word': prevword,
        '-1:pos': prevpos,
        '-1:iob': previob,
        '-1:shape': shape(prevword),
        '-1:word.lower()': prevword.lower(),
        '-1:word.isupper()': prevword.isupper(),
        '-1:word.istitle()': prevword.istitle(),
        '-1:word.isdigit()=': prevword.isdigit(),

        '-2:word': prevprevword,
        '-2:pos': prevprevpos,
        # '-2:lemma': stemmer[LANGUAGE_CODE].stem(prevprevword),
        '-2:iob': prevpreviob,
        '-2:shape': shape(prevprevword),
        '-2:word.lower()': prevprevword.lower(),
        '-2:word.isupper()': prevprevword.isupper(),
        '-2:word.istitle()': prevprevword.istitle(),
        '-2:word.isdigit()': prevprevword.isdigit(),
    }
 
def my_modified_scorer(chunker, iob_tagged_datatest, **kwargs):
    correct_entities = 0
    total_entities = 0
    for iob_tagged_tokens in list(iob_tagged_datatest):
        words, tags, iob_tags = list(zip(*iob_tagged_tokens))
        pos_tagged_tokens = list(zip(words, tags))
        predicted = chunker.parse_to_iob_tagged_tokens(pos_tagged_tokens)
        
        datatest_entities = get_entities_from_iob_tagged_tokens(iob_tagged_tokens)
        predicted_entities = get_entities_from_iob_tagged_tokens(predicted)

        if kwargs.get('verbose', 0):
            print("\nACTUAL   :", iob_tagged_tokens)
            print("         :", datatest_entities)
            print("         :", len(datatest_entities), "entities")
            print("PREDICTED:", predicted)
            print("         :", predicted_entities)

        total_entities += len(datatest_entities)
        for datatest_entity in datatest_entities:
            if datatest_entity in predicted_entities:
                correct_entities += 1

    if total_entities:
        return correct_entities / total_entities
    else:
        print("Dividing by 0, returning 0")
        return 0

def mitie_scorer(extractor, iob_tagged_datatest):
    correct_entities = 0
    total_entities = 0
    for iob_tagged_tokens in list(iob_tagged_datatest):
        words, tags, iob_tags = list(zip(*iob_tagged_tokens))
        pos_tagged_tokens = list(zip(words, tags))
        predicted = extractor.extract_entities(pos_tagged_tokens)
        
        datatest_entities = get_entities_from_iob_tagged_tokens(iob_tagged_tokens)
        predicted_entities = get_entities_from_iob_tagged_tokens(predicted)
        
        # print("\nACTUAL   :", iob_tagged_tokens)
        # print("         :", datatest_entities)
        # print("         :", len(datatest_entities), "entities")
        # print("PREDICTED:", predicted)
        # print("         :", predicted_entities)

        total_entities += len(datatest_entities)
        for datatest_entity in datatest_entities:
            if datatest_entity in predicted_entities:
                correct_entities += 1

    if total_entities:
        return correct_entities / total_entities
    else:
        print("Dividing by 0, returning 0")
        return 0

def get_entities_from_iob_tagged_tokens(iob_tagged_tokens):
    #entity = (word, tag)
    #example: ('Wonder Woman', 'movie_title', token pos)
    current_entity = None
    current_entity_words = None
    current_entity_name = None
    current_entity_pos = None
    all_entities = []
    for i, (word, tag, iob_tag) in enumerate(iob_tagged_tokens):
        iob = iob_tag[:2]
        entity_name = iob_tag[2:]
        if current_entity_words:
            if iob == "I-": #concatenate word if in chunk
                current_entity_words += " " + word
            else: #append to list if chunk stops
                current_entity = (current_entity_words, current_entity_name, current_entity_pos)
                all_entities.append(current_entity)
                current_entity = None
                current_entity_words = None
                current_entity_name = None
                current_entity_pos = None
        if iob == "B-": #beginning of chunk
            current_entity_name = entity_name
            current_entity_words = word
            current_entity_pos = i

    #check if the last word is in chunk
    if current_entity_words:
        current_entity = (current_entity_words, current_entity_name, current_entity_pos)
        all_entities.append(current_entity)
    return all_entities

def get_entities_from_iob_tagged_tokens1(iob_tagged_tokens):
    #entity = (word, tag)
    #example: ('Wonder Woman', 'movie_title', token begin, token end + 1)
    current_entity = None
    current_entity_words = None
    current_entity_name = None
    current_entity_pos = None
    all_entities = []
    for i, (word, tag, iob_tag) in enumerate(iob_tagged_tokens):
        iob = iob_tag[:2]
        entity_name = iob_tag[2:]
        if current_entity_words:
            if iob == "I-": #concatenate word if in chunk
                current_entity_words += " " + word
            else: #append to list if chunk stops
                current_entity = (current_entity_words, current_entity_name, current_entity_pos, i)
                all_entities.append(current_entity)
                current_entity = None
                current_entity_words = None
                current_entity_name = None
                current_entity_pos = None
        if iob == "B-": #beginning of chunk
            current_entity_name = entity_name
            current_entity_words = word
            current_entity_pos = i

    #check if the last word is in chunk
    if current_entity_words:
        current_entity = (current_entity_words, current_entity_name, current_entity_pos, len(iob_tagged_tokens))
        all_entities.append(current_entity)
    return all_entities

def save_object(obj, filename):
    with open(filename, 'wb') as output_file:
        dill.dump(obj, output_file)

def load_object(filename):
    with open(filename, 'rb') as input_file:
        return dill.load(input_file)

def print_here(x, y, text):
    sys.stdout.write("\x1b7\x1b[%d;%df%s\x1b8" % (x, y, text))
    sys.stdout.flush()