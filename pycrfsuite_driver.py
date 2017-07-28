from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import time
import pycrfsuite
from util import *
from threading import Thread


NUM_THREAD = 4

def word2features(tokens, index, history):
    """
	`tokens`  = a POS-tagged sentence [(w1, t1), ...]
	`index`   = the index of the token we want to extract features for
	`history` = the previous predicted IOB tags
	"""

    # Pad the sequence with placeholders
    tokens = [('__START2__', '__START2__'), ('__START1__', '__START1__')] + list(tokens) + [('__END1__', '__END1__'),
                                                                                            ('__END2__', '__END2__')]
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

    return [
        'word=' + word,
        'lemma=' + stemmer.stem(word),
        'pos=' + pos,
        'pos[:2]=' + str(pos[:2]),
        'shape=' + shape(word),
        'word.lower()=' + str(word.lower()),
        'word.isupper()=' + str(word.isupper()),
        'word.istitle()=' + str(word.istitle()),
        'word.isdigit()=' + str(word.isdigit()),

        '+1:word=' + nextword,
        '+1:pos=' + nextpos,
        '+1:lemma=' + stemmer.stem(nextword),
        '+1:shape=' + shape(nextword),
        '+1:word.lower()=' + str(nextword.lower()),
        '+1:word.isupper()=' + str(nextword.isupper()),
        '+1:word.istitle()=' + str(nextword.istitle()),
        '+1:word.isdigit()=' + str(nextword.isdigit()),

        '+2:word=' + nextnextword,
        '+2:pos=' + nextnextpos,
        '+2:lemma=' + stemmer.stem(nextnextword),
        '+2:shape=' + shape(nextnextword),
        '+2:word.lower()=' + str(nextnextword.lower()),
        '+2:word.isupper()=' + str(nextnextword.isupper()),
        '+2:word.istitle()=' + str(nextnextword.istitle()),
        '+2:word.isdigit()=' + str(nextnextword.isdigit()),

        '-1:word=' + prevword,
        '-1:pos=' + prevpos,
        '-1:lemma=' + stemmer.stem(prevword),
        '-1:iob=' + previob,
        '-1:shape=' + shape(prevword),
        '-1:word.lower()=' + str(prevword.lower()),
        '-1:word.isupper()=' + str(prevword.isupper()),
        '-1:word.istitle()=' + str(prevword.istitle()),
        '-1:word.isdigit()=' + str(prevword.isdigit()),

        '-2:word=' + prevprevword,
        '-2:pos=' + prevprevpos,
        '-2:lemma=' + stemmer.stem(prevprevword),
        '-2:iob=' + prevpreviob,
        '-2:shape=' + shape(prevprevword),
        '-2:word.lower()=' + str(prevprevword.lower()),
        '-2:word.isupper()=' + str(prevprevword.isupper()),
        '-2:word.istitle()=' + str(prevprevword.istitle()),
        '-2:word.isdigit()=' + str(prevprevword.isdigit()),
    ]


def sent2features(iob_tagged_sentence, iob_tagged_sentences, feature_detector):
    words, tags, iob_tags = list(zip(*iob_tagged_sentence))
    X_sent, y_sent = [], []
    for index in range(len(iob_tagged_sentence)):
        X_sent.append(feature_detector(iob_tagged_sentences, index, history=iob_tags[:index]))
        y_sent.append(iob_tags[index])

    return X_sent, y_sent

def to_dataset_for_single_sentence(index, X, y, parsed_sentences, feature_detector):
    for iob_tagged in parsed_sentences:
        words, tags, iob_tags = list(zip(*iob_tagged))

        tagged = list(zip(words, tags))

        X_sent, y_sent = sent2features(iob_tagged, tagged, feature_detector)
        X[index].append(X_sent)
        y[index].append(y_sent)

def to_dataset(parsed_sentences, feature_detector):
    """
    Transform a list of tagged sentences into a scikit-learn compatible POS dataset
    :param parsed_sentences:
    :param feature_detector:
    :return:
    """
    X, y = [], []
    for iob_tagged in parsed_sentences:
        words, tags, iob_tags = list(zip(*iob_tagged))

        tagged = list(zip(words, tags))

        X_sent, y_sent = sent2features(iob_tagged, tagged, feature_detector)
        X.append(X_sent)
        y.append(y_sent)

    return X, y


def score_sentences_thread(index, results, iob_tagged_datatest, y_pred):
    total_entities = 0
    correct_entities = 0
    for i, iob_tagged_tokens in enumerate(iob_tagged_datatest):
        words, tags, iob_tags = list(zip(*iob_tagged_tokens))
        predicted_iob_tokens = list(zip(words, tags, y_pred[i]))

        datatest_entities = get_entities_from_iob_tagged_tokens(iob_tagged_tokens)
        predicted_entities = get_entities_from_iob_tagged_tokens(predicted_iob_tokens)

        # print("\nACTUAL   :", iob_tagged_tokens)
        # print("         :", datatest_entities)
        # print("         :", len(datatest_entities), "entities")
        # print("PREDICTED:", predicted_iob_tokens)
        # print("         :", predicted_entities)

        total_entities += len(datatest_entities)
        for datatest_entity in datatest_entities:
            if datatest_entity in predicted_entities:
                correct_entities += 1

    results[index] = (total_entities, correct_entities)


def my_modified_crfsuite_scorer(tagger, iob_tagged_datatest, feature_detector):
    print(" >> Converting to data test")
    X_test, y_test = to_dataset(iob_tagged_datatest, word2features)
    print(" >> Predicting labels")
    y_pred = [tagger.tag(xseq) for xseq in X_test]

    print(" >> Calculating Accuracy")
    iob_tagged_datatest_len_per_thread = -(-len(iob_tagged_datatest) // 4)
    threads = [None] * NUM_THREAD
    results = [None] * NUM_THREAD
    for i in range(NUM_THREAD):
        start_index = i * iob_tagged_datatest_len_per_thread
        end_index = (i+1) * iob_tagged_datatest_len_per_thread
        threads[i] = Thread(target=score_sentences_thread, args=(
            i, results, iob_tagged_datatest[start_index : end_index], y_pred[start_index : end_index]
        ))
        threads[i].start()

    for thread in threads:
        thread.join()

    total_entities_in_sentence, correct_entities_in_sentence = zip(*results)

    total_entities = sum(total_entities_in_sentence)
    correct_entities = sum(correct_entities_in_sentence)

    if total_entities:
        accuracy = (correct_entities / total_entities) * 100
        print(" > Accuracy: {0:.10f}%".format(accuracy))
    else:
        print("Dividing by 0")
    print(" > Report:")
    print(bio_classification_report(y_test, y_pred))


def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.
    
    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
    )


def train(data_train, model_file_path):
    print("\n=== TRAINING {} DATA ===".format(len(data_train)))
    print(" > Extracting features")
    X_train, y_train = to_dataset(data_train, word2features)
    trainer = pycrfsuite.Trainer(verbose=False)
    print(" > Appending data trains")
    for i in range(len(X_train)):
        trainer.append(X_train[i], y_train[i])

    trainer.set_params({
        'algorithm': 'lbfgs',
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 100,  # stop earlier

        # include transitions that are possible, but not observed
        'all_possible_transitions': True
    })
    print(" > Start training. . .")
    master_begin_time = time.time()
    trainer.train(model_file_path)
    master_end_time = time.time()
    print(" > Training done in {} seconds".format(master_end_time - master_begin_time))
    print(" > Model saved in {}".format(model_file_path))


def test(data_test, model_file_path, feature_detector):
    print("\n=== TESTING {} DATA ===".format(len(data_test)))
    tagger = pycrfsuite.Tagger()
    tagger.open(model_file_path)
    master_begin_time = time.time()
    my_modified_crfsuite_scorer(tagger, data_test, feature_detector)
    master_end_time = time.time()
    print(" > Testing done in {} seconds".format(master_end_time - master_begin_time))


gmb_iob_triads = load_object(GMB_IOB_TRIAD_PATH)

train_size = 5000
test_size = 1000
data_train = gmb_iob_triads[:train_size]
data_test = gmb_iob_triads[train_size: train_size + test_size]

model_file_path = 'obj/gmb_{}.crfsuite'.format(train_size, test_size)
# train(data_train, model_file_path)
test(data_test, model_file_path, word2features)
