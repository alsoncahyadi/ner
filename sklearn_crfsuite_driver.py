import time
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.grid_search import RandomizedSearchCV
from util import *

import sklearn_crfsuite
from sklearn_crfsuite import metrics
from threading import Thread
from collections import Counter


NUM_THREAD = 4

"""
    Python version: 3.5.3
    DEPENDENCIES:
        sudo apt install libicu-dev
        pip install pyicu
    
        pip install polyglot
        pip install sklearn_crfsuite
        pip install scipy
        pip install sklearn
        pip install dill
        pip install nltk
        pip install Sastrawi
        
        polyglot download embeddings2.id pos2.id
"""

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))


def print_top_transitions(crf):
    print("\n\n >> Top likely transitions:")
    print_transitions(Counter(crf.transition_features_).most_common(20))

    print("\n >> Top unlikely transitions:")
    print_transitions(Counter(crf.transition_features_).most_common()[-20:])


def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))


def print_top_state_features(crf):
    print("\n\n Top positive:")
    print_state_features(Counter(crf.state_features_).most_common(30))

    print("\n Top negative:")
    print_state_features(Counter(crf.state_features_).most_common()[-30:])


def sent2features(iob_tagged_sentence, iob_tagged_sentences, feature_detector):
    words, tags, iob_tags = list(zip(*iob_tagged_sentence))
    X_sent, y_sent = [], []
    for index in range(len(iob_tagged_sentence)):
        X_sent.append(feature_detector(iob_tagged_sentences, index, history=iob_tags[:index]))
        y_sent.append(iob_tags[index])

    return X_sent, y_sent


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


def bio_classification_report(crf, y_test, y_pred):
    labels = list(crf.classes_)
    labels.remove('O')

    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    return metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=4
    )


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


def my_modified_crfsuite_scorer(crf, iob_tagged_datatest, feature_detector):
    print(" >> Converting to data test")
    X_test, y_test = to_dataset(iob_tagged_datatest, feature_detector)
    print(" >> Predicting labels")
    y_pred = crf.predict(X_test)

    #Threading
    print(" >> Calculating Accuracy")
    iob_tagged_datatest_len_per_thread = -(-len(iob_tagged_datatest) // 4)
    threads = [None] * NUM_THREAD
    results = [None] * NUM_THREAD
    for i in range(NUM_THREAD):
        start_index = i * iob_tagged_datatest_len_per_thread
        end_index = (i + 1) * iob_tagged_datatest_len_per_thread
        threads[i] = Thread(target=score_sentences_thread, args=(
            i, results, iob_tagged_datatest[start_index: end_index], y_pred[start_index: end_index]
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
    print(bio_classification_report(crf, y_test, y_pred))


def train(data_train, model_file_path):
    print("\n=== TRAINING {} DATA ===".format(len(data_train)))
    print(" > Extracting features")
    X_train, y_train = to_dataset(data_train, ner_features)
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    print(" > Start training. . .")
    master_begin_time = time.time()
    crf.fit(X_train, y_train)
    master_end_time = time.time()
    print(" > Training done in {} seconds".format(master_end_time - master_begin_time))
    save_object(crf, model_file_path)
    print(" > Model saved in {}".format(model_file_path))


def k_fold_validation(data_train, model_file_path, **kwargs):
    print("\n=== CROSS VALIDATING {} DATA WITH k: {} ===".format(len(data_train), kwargs.get('k', 4)))
    print(" > Extracting features")
    X_train, y_train = to_dataset(data_train, ner_features)
    print(" > Start training. . .")
    master_begin_time = time.time()
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=100,
        all_possible_transitions=True
    )
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
    }

    labels = ['B-gpe', 'B-org', 'I-org', 'B-geo', 'I-geo', 'B-tim', 'I-tim', 'B-per',
              'I-per', 'B-art', 'I-art', 'I-gpe', 'B-eve', 'I-eve', 'B-nat', 'I-nat']

    # use the same metric for evaluation
    f1_scorer = make_scorer(
        metrics.flat_f1_score,
        average='weighted', labels=labels
    )

    # search
    rs = RandomizedSearchCV(crf, params_space,
                            cv=kwargs.get('k', 4),
                            verbose=1,
                            n_jobs=-1,
                            n_iter=kwargs.get('n_iter', 10),
                            scoring=f1_scorer)
    rs.fit(X_train, y_train)
    master_end_time = time.time()
    print(" > Training done in {} seconds".format(master_end_time - master_begin_time))

    crf = rs.best_estimator_
    print(' >> best params:', rs.best_params_)
    print(' >> best CV score:', rs.best_score_)
    print(' >> model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))
    save_object(crf, model_file_path)
    print(" > Model saved in {}".format(model_file_path))


def test(data_test, model_file_path, feature_detector):
    print("\n=== TESTING {} DATA ===".format(len(data_test)))
    crf = load_object(model_file_path)
    master_begin_time = time.time()
    my_modified_crfsuite_scorer(crf, data_test, feature_detector)
    master_end_time = time.time()
    print(" > Testing done in {} seconds".format(master_end_time - master_begin_time))
    print("\n=== WHAT THE CLASSIFIER LEARNED ===")
    print_top_state_features(crf)
    print_top_transitions(crf)

if __name__ == "__main__":
    # gmb_iob_triads = load_object(GMB_IOB_TRIAD_PATH)

    train_size = 400
    test_size = 80
    # data_train = load_object(GMB_IOB_TRIAD_PATH)[:7000]
    # data_test = load_object(GMB_RANDOM_2000_TEST_PATH)

    data_train = load_object(ID_IOB_TRIAD_PATH)[:train_size]
    data_test = load_object(ID_IOB_TRIAD_PATH)[train_size : test_size + train_size]

    # data_train = list(nltk.corpus.conll2000.iob_sents('train.txt'))
    # data_test = list(nltk.corpus.conll2000.iob_sents('test.txt'))
    # train_size = len(data_train)
    # test_size = len(data_test)

    model_file_path = 'obj/id/sklearn-5fold50-no+1.crfsuite'.format(train_size, test_size)
    train(data_train, model_file_path)
    k_fold_validation(data_train, model_file_path, k=5, n_iter = 75)
    test(data_test, model_file_path, ner_features)
