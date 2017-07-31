import sklearn_crfsuite
import time
import random
import scipy.stats

from collections import Counter
from threading import Thread

from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import metrics

from ViraAdapter import ViraAdapter
from util import *


class TrainCRFOnVira:
    """
    USAGE:
        1. Construct the class
            => The file specified will be directly converted to readable IOB dataset (CoNLL format)
        2. Train with 'train' or 'train_with_RandomizedSearchCV' method
            => The trained classifier will be saved as the attribute '_crf'
        2. Test with 'test' method
            => The tested accuracy will be saved as the attribute 'test_score'
    """

    def __init__(self, vira_json_file_path, train_size_percentage):
        """
        :param vira_json_file_path: The file path to the json file
        :param train_size_percentage: train size percentage. The test size percentage would be 1 - train_size_percentage
        """
        self.vira_adapter = ViraAdapter()
        self.vira_json_file_path = vira_json_file_path
        self.iob_sentences = self.vira_adapter.jsonfile2iobsentences(vira_json_file_path)

        self._train_size_percentage = train_size_percentage
        self._test_size_percentage = 1 - train_size_percentage
        self._train_size = int(self._train_size_percentage * len(self.iob_sentences))
        self._test_size = len(self.iob_sentences) - self._train_size
        #for consistent train and test
        random.seed(0)
        self._sampling_index_list = random.sample(range(len(self.iob_sentences)), len(self.iob_sentences))

        self.data_train = []
        self.data_test = []
        self._initialize_data()
        self.prediction_reports = [None] * len(self.data_test)


        self._crf = None
        self.accuracy = -1

    def _initialize_data(self):
        for train_sampling_index in self._sampling_index_list[:self._train_size]:
            self.data_train.append(self.iob_sentences[train_sampling_index])

        for test_sampling_index in self._sampling_index_list[self._train_size : self._train_size+self._test_size]:
            self.data_test.append(self.iob_sentences[test_sampling_index])

    def _sent2features(self, iob_tagged_sentence, iob_tagged_sentences, feature_detector):
        words, tags, iob_tags = list(zip(*iob_tagged_sentence))
        X_sent, y_sent = [], []
        for index in range(len(iob_tagged_sentence)):
            X_sent.append(feature_detector(iob_tagged_sentences, index, history=iob_tags[:index]))
            y_sent.append(iob_tags[index])

        return X_sent, y_sent

    def _to_dataset(self, parsed_sentences, feature_detector):
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

            X_sent, y_sent = self._sent2features(iob_tagged, tagged, feature_detector)
            X.append(X_sent)
            y.append(y_sent)

        return X, y

    def _print_transitions(self, trans_features):
        for (label_from, label_to), weight in trans_features:
            print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

    def _print_top_transitions(self, crf):
        print("\n\n >> Top likely transitions:")
        self._print_transitions(Counter(crf.transition_features_).most_common(20))

        print("\n >> Top unlikely transitions:")
        self._print_transitions(Counter(crf.transition_features_).most_common()[-20:])

    def _print_state_features(self, state_features):
        for (attr, label), weight in state_features:
            print("%0.6f %-8s %s" % (weight, label, attr))

    def _print_top_state_features(self, crf):
        print("\n\n >> Top positive:")
        self._print_state_features(Counter(crf.state_features_).most_common(30))

        print("\n >> Top negative:")
        self._print_state_features(Counter(crf.state_features_).most_common()[-30:])

    def _bio_classification_report(self, crf, y_test, y_pred):
        labels = list(crf.classes_)
        labels.remove('O')

        sorted_labels = sorted(
            labels,
            key=lambda name: (name[1:], name[0])
        )
        return metrics.flat_classification_report(
            y_test, y_pred, labels=sorted_labels, digits=4
        )

    def _score_sentences_thread(self, index, results, iob_tagged_datatest, y_pred, start_index, end_index):
        total_entities = 0
        correct_entities = 0
        print(start_index, ":", end_index)
        for i, iob_tagged_tokens in enumerate(iob_tagged_datatest[start_index:end_index]):
            words, tags, iob_tags = list(zip(*iob_tagged_tokens))
            predicted_iob_tokens = list(zip(words, tags, y_pred[start_index + i]))

            words = list(words)
            tags = list(tags)
            iob_tags = list(iob_tags)

            datatest_entities = get_entities_from_iob_tagged_tokens(iob_tagged_tokens)
            predicted_entities = get_entities_from_iob_tagged_tokens(predicted_iob_tokens)


            report_str = "================================= Sentence #{} ========================================\n".\
                format(start_index + i)
            report_str += "> Text: '{}'\n\n".format(" ".join(words))
            report_str += " ## ACTUAL ##\n"
            report_str += "> IOB tagged tokens:\n{}\n".format(str(iob_tagged_tokens))
            report_str += "> Entities:\n{}\n\n".format(datatest_entities)
            report_str += " ## PREDICTED ##\n"
            report_str += "> IOB tagged tokens:\n{}\n".format(predicted_iob_tokens)
            report_str += "> Entities:\n{}\n".format(predicted_entities)
            report_str += "======================================================================================\n\n"

            is_predicted_true = True
            total_entities += len(datatest_entities)
            for datatest_entity in datatest_entities:
                if datatest_entity in predicted_entities:
                    correct_entities += 1
                else:
                    is_predicted_true = False


            self.prediction_reports[start_index + i] = {
                'report_text': report_str,
                'is_predicted_true': is_predicted_true
            }

        results[index] = (total_entities, correct_entities)

    def train(self):
        print("\n=== TRAINING {} DATA ===".format(len(self.data_train)))
        print(" > Extracting features")
        X_train, y_train = self._to_dataset(self.data_train, ner_features)
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
        self._crf = crf
        print(" > Model saved as class attribute (self._crf)")
        # save_object(crf, model_file_path)
        # print(" > Model saved in {}".format(model_file_path))

    def train_with_RandomizeSearchCV(self, **kwargs):
        """
        fitting (k * n_iter) times

        :param kwargs:
            k => how many folds to do k-fold
            n_iter => iterations of different hyper parameters
        :return:
        """
        print("\n=== CROSS VALIDATING {} DATA WITH k: {} ===".format(len(self.data_train), kwargs.get('k', 5)))
        print(" > Extracting features")
        X_train, y_train = self._to_dataset(self.data_train, ner_features)
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
                                cv=kwargs.get('k', 5),
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
        self._crf = crf
        print(" > Model saved as class attribute (self._crf)")
        # save_object(crf, model_file_path)
        # print(" > Model saved in {}".format(model_file_path))

    def test(self, **kwargs):
        NUM_THREAD = 4

        print("\n=== TESTING {} DATA ===".format(len(self.data_test)))
        if not self._crf:
            raise RuntimeError("You haven't trained any classifier! Train by running this class' 'train' method")
        crf = self._crf
        print("\n  ## WHAT THE CLASSIFIER LEARNED ##")
        self._print_top_state_features(crf)
        self._print_top_transitions(crf)
        master_begin_time = time.time()
        print("\n  ## CALCULATING ACCURACY ##")
        print(" >> Converting to data test")
        X_test, y_test = self._to_dataset(self.data_test, ner_features)
        print(" >> Predicting labels")
        y_pred = crf.predict(X_test)

        # Threading
        print(" >> Calculating Accuracy")
        iob_tagged_datatest_len_per_thread = -(-len(self.data_test) // 4)
        threads = [None] * NUM_THREAD
        results = [None] * NUM_THREAD
        for i in range(NUM_THREAD):
            start_index = i * iob_tagged_datatest_len_per_thread
            end_index = (i + 1) * iob_tagged_datatest_len_per_thread
            threads[i] = Thread(target=self._score_sentences_thread, args=(
                i, results, self.data_test, y_pred, start_index, end_index
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
            self.accuracy = accuracy
        else:
            print("There are no entities in the data test (Dividing by 0)")
        print(" > Report:")
        print(self._bio_classification_report(crf, y_test, y_pred))
        master_end_time = time.time()
        print(" > Testing done in {} seconds".format(master_end_time - master_begin_time))

        # Write report text
        report_folder_path = kwargs.get('report_folder_path', 'report.txt')
        with open(report_folder_path + "/true.txt", 'w') as true_report_file:
            for prediction_report in self.prediction_reports:
                if prediction_report['is_predicted_true']:
                    true_report_file.write(prediction_report['report_text'])
        with open(report_folder_path + "/false.txt", 'w') as false_report_file:
            for prediction_report in self.prediction_reports:
                if not prediction_report['is_predicted_true']:
                    false_report_file.write(prediction_report['report_text'])
        with open(report_folder_path + "/all.txt", 'w') as all_report_file:
            for prediction_report in self.prediction_reports:
                all_report_file.write(prediction_report['report_text'])
        print(" > Report file saved in {} folder".format(report_folder_path))


if __name__ == "__main__":
    vira_trainer = TrainCRFOnVira("data/vira_example.json", 0.8)
    """
    Choose either <b>ONE</b> of vira_trainer.train() or vira_trainer.train_with_RandomizeSearchCV
    : vira_trainer.train() : This is a normal trainer
    : vira_trainer.train_with_RandomizeSearchCV : This is a parameter optimizer, AND cross validator.
                                                  For full lists of modifieable variables, read the documentation
                                                  under it's declaration.
                                                  The default is:
                                                    k       : 5
                                                    n_iter  : 10
    """
    vira_trainer.train()
    # vira_trainer.train_with_RandomizeSearchCV()
    vira_trainer.test(
        print_to_text=True,
        report_folder_path='report_vira/'
    )
