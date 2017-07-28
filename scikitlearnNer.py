import itertools
import time
import dill
import sys

from util import *
from nltk import tree2conlltags
from nltk.chunk import ChunkParserI, conlltags2tree
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.naive_bayes import MultinomialNB
from copy import deepcopy

class ScikitLearnChunker(ChunkParserI):
 
    @classmethod
    def to_dataset(cls, parsed_sentences, feature_detector):
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
 
            for index in range(len(iob_tagged)):
                X.append(feature_detector(tagged, index, history=iob_tags[:index]))
                y.append(iob_tags[index])

        return X, y
 
    @classmethod
    def get_minibatch(cls, parsed_sentences, feature_detector, batch_size=350):
        batch = list(itertools.islice(parsed_sentences, batch_size))
        X, y = cls.to_dataset(batch, feature_detector)
        return X, y
 
    @classmethod
    def train(cls, parsed_sentences, feature_detector, all_classes, **kwargs):
        X, y = cls.get_minibatch(parsed_sentences, feature_detector, kwargs.get('batch_size', 500))
        vectorizer = DictVectorizer(sparse=False)
        vectorizer.fit(X)
        
        # clf = MultinomialNB()
        # clf = SGDClassifier(verbose=0, n_jobs=-1, n_iter=kwargs.get('n_iter', 5))
        # clf = Perceptron(verbose=0, n_jobs=-1, n_iter=kwargs.get('n_iter', 10))
        clf = MLPClassifier(
            hidden_layer_sizes= (7,7,7,7,7,7,7,7),
            verbose= 0,
            activation= 'relu',
            solver= 'adam',
            random_state= 0,
        )

        cnt = 1
        master_begin_time = time.time()
        num_tokens = 0
        while len(X):
            # print(len(parsed_sentences))
            num_tokens += len(X)
            begin_time = time.time()
            X = vectorizer.transform(X)
            clf.partial_fit(X, y, all_classes)
            end_time = time.time()
            print("Batch [# {0}] done in [{1:.2f} seconds]".format(cnt, end_time - begin_time), flush=True, end="\r")
            X, y = cls.get_minibatch(parsed_sentences, feature_detector, kwargs.get('batch_size', 500))
            cnt += 1
        master_end_time = time.time()
        print("All batches ({0:d} tokens) done in [[{1:.2f} seconds]]".format(num_tokens, master_end_time - master_begin_time))
 
        clf = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', clf)
        ])
 
        return cls(clf, feature_detector)

    @classmethod
    def train_loaded_classifier(cls, pipe, parsed_sentences, feature_detector, all_classes, **kwargs):
        X, y = cls.get_minibatch(parsed_sentences, feature_detector, kwargs.get('batch_size', 500))
        vectorizer = pipe.named_steps['vectorizer']
        vectorizer.transform(X)

        clf = pipe.named_steps['classifier']

        cnt = 0
        master_begin_time = time.time()
        num_tokens = 0
        while len(X):
            # print(len(parsed_sentences))
            num_tokens += len(X)
            begin_time = time.time()
            X = vectorizer.transform(X)
            clf.partial_fit(X, y, all_classes)
            end_time = time.time()
            print("Batch [# {0}] done in [{1:.2f} seconds] ({2} tokens)".format(
                cnt, end_time - begin_time, len(X)), flush=True, end="\r"
            )
            X, y = cls.get_minibatch(parsed_sentences, feature_detector, kwargs.get('batch_size', 500))
            cnt += 1
        master_end_time = time.time()
        print("{0} batches ({1:d} tokens) done in [[{2:.2f} seconds]]".format(
            cnt ,num_tokens, master_end_time - master_begin_time)
        )
 
        clf = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', clf)
        ])
 
        return cls(clf, feature_detector)#, vectorizer)

    @classmethod
    def load(cls, classifier_path, feature_detector):
        with open(classifier_path, 'rb') as input_file:
            clf = dill.load(input_file)
            return cls(clf, feature_detector)#, None)

    @classmethod
    def save(cls, classifier, feature_detector):
        with open( + "/classifier.pkl",) as output_file:
            pass

    def __init__(self, classifier, feature_detector):#, vectorizer):
        self._classifier = classifier
        self._feature_detector = feature_detector
        # self._vectorizer = vectorizer
        # self._pipeline = Pipeline([
        #     ('vectorizer', self._vectorizer),
        #     ('classifier', self._classifier)
        # ])
 
    def parse(self, tokens):
        """
        Chunk a tagged sentence
        :param tokens: List of words [(w1, t1), (w2, t2), ...]
        :return: chunked sentence: nltk.Tree
        """
        history = []
        iob_tagged_tokens = []
        for index, (word, tag) in enumerate(tokens):
            iob_tag = self._classifier.predict([self._feature_detector(tokens, index, history)])[0]
            history.append(iob_tag)
            iob_tagged_tokens.append((word, tag, iob_tag))
 
        return conlltags2tree(iob_tagged_tokens)

    def parse_to_iob_tagged_tokens(self, tokens):
        history = []
        iob_tagged_tokens = []
        for index, (word, tag) in enumerate(tokens):
            iob_tag = self._classifier.predict([self._feature_detector(tokens, index, history)])[0]
            history.append(iob_tag)
            iob_tagged_tokens.append((word, tag, iob_tag))
        return iob_tagged_tokens
 
    def score(self, parsed_sentences):
        """
        Compute the accuracy of the tagger for a list of test sentences
        :param parsed_sentences: List of parsed sentences: nltk.Tree
        :return: float 0.0 - 1.0
        """
        X_test, y_test = self.__class__.to_dataset(parsed_sentences, self._feature_detector)
        score_float = self._classifier.score(X_test, y_test)
        return score_float

    def score_with_modified_scorer(self, parsed_sentences):
        print("Converting raw data test")
        X_test, y_test = self.__class__.to_dataset(parsed_sentences, self._feature_detector)
        print("Scoring {} entries".format(len(X_test)))
        return self.scorer(self._classifier, X_test, y_test)

    def print_evaluation(self, test_samples_tree):
        print("\nTesting {} samples".format(len(test_samples_tree)))
        score = self.evaluate(test_samples_tree)
        print(score)

def auto_train(classifier_label, start_iteration, data_train, data_test, **kwargs):
    master_reader = list(data_train)
    print("Training {} data".format(len(master_reader)))
    i = start_iteration
    print("STARTING")

    while(True):
        print("\n======================= #{} ========================".format(i))
        #CREATING DATATEST
        reader = deepcopy(master_reader)
        if kwargs.get("is_alternate", False):
            if i % 2 == 0:
                print("Even iteration: reversed dataset")
                reader = list(reversed(reader))

        #LOADING PREVIOUS MODEL
        previous_classifier_name = classifier_label + str(i-1)
        pa_ner = ScikitLearnChunker.load("obj/id/scikit-classifier-" + previous_classifier_name + ".pkl", feature_detector=ner_features)

        #TRAINING
        all_classes = kwargs.get("all_classes", ['O', 'B-per', 'I-per', 'B-gpe', 'I-gpe',
                                                 'B-geo', 'I-geo', 'B-org', 'I-org', 'B-tim', 'I-tim',
                                                 'B-art', 'I-art', 'B-eve', 'I-eve', 'B-nat', 'I-nat'])
        pa_ner = ScikitLearnChunker.train_loaded_classifier(pa_ner._classifier, itertools.islice(reader,5000),
                                                            feature_detector=ner_features,
                                                            all_classes=all_classes,
                                                            batch_size=kwargs.get('batch_size', 300), n_iter=10
                                                            )
            
        #SCORING
        test_reader = data_test
        accuracy = my_modified_scorer(pa_ner, test_reader, verbose=kwargs.get('verbose', 0)) * 100
        print("\n             <### Score Details ###>             ")
        print("  > Classifier               : {}".format(type(pa_ner._classifier.named_steps["classifier"]).__name__)) #prints the classifier type
        print("  > Modified Accuracy        : {0:.10f}%".format(accuracy, len(data_test)))
        # print("  > Built in score           : {}".format(pa_ner.score(data_test)))
        print("===================================================".format(i))

        #SAVING NEW CLASSIFIER
        new_classifier_name = classifier_label + str(i)
        save_object(pa_ner._classifier, "obj/id/scikit-classifier-" + new_classifier_name + ".pkl")

        #WRITING ACCURACY ON FILE
        with open("report/{}.csv".format(classifier_label[:len(classifier_label)-1]), "a+") as report_file:
            report_file.write("{0}; {1:.10f}%\r\n".format(i, accuracy))

        i += 1

def train_perceptron(data_train, data_test, **kwargs):
    is_train = True
    is_load = False
    reader = list(data_train)
    pa_ner = None
    opt = input("Do you want to load classifier? (y/n): ")
    if opt == "y":
        is_load = True
        classifier_name = input("Input classifier name: ")
        print("Loading classifier from obj/id/scikit-classifier-" + classifier_name + ".pkl")
        pa_ner = ScikitLearnChunker.load("obj/id/scikit-classifier-" + classifier_name + ".pkl", feature_detector=ner_features)
        print("Load successful")
        opt = input("Do you still want to train? (y/n): ")
        if opt == "y":
            is_train = True
        else:
            is_train = False
    
    if is_train:
        all_classes = kwargs.get("all_classes", ['O', 'B-per', 'I-per', 'B-gpe', 'I-gpe',
                       'B-geo', 'I-geo', 'B-org', 'I-org', 'B-tim', 'I-tim',
                       'B-art', 'I-art', 'B-eve', 'I-eve', 'B-nat', 'I-nat'])

        train_size = int(input("Input Training Size: "))
        print("Training. . .")
        if is_load:
            pa_ner = ScikitLearnChunker.train_loaded_classifier(pa_ner._classifier, itertools.islice(reader, train_size),
                                                                feature_detector=ner_features,
                                                                all_classes=all_classes,
                                                                batch_size=350,
                                                                n_iter=10
                                                                )
        else:
            print("HEEE")
            pa_ner = ScikitLearnChunker.train(itertools.islice(reader, train_size),
                                              feature_detector=ner_features,
                                              all_classes=all_classes,
                                              batch_size=350,
                                              n_iter=10
                                              )
        print("Done Training")
        
    #SCORING
    print("\n\nReading data test")
    test_reader = data_test
    accuracy = my_modified_scorer(pa_ner, test_reader) * 100
    print(" > Modified Accuracy : {0:.10f}%".format(accuracy))
    # print("\nOriginal:")
    # print("First 2000:")
    # pa_ner.print_evaluation([conlltags2tree(datum) for datum in test_reader[:2000]])
    # print("\nLast 2000:")
    # pa_ner.print_evaluation([conlltags2tree(datum) for datum in test_reader[-2000:]])

    #SAVE FILE
    opt = input("Do you want to save model? (y/n): ")
    if "y" in opt:
        classifier_name = input("Input classifier name: ")
        save_object(pa_ner._classifier, "obj/id/scikit-classifier-" + classifier_name + ".pkl")
        print("Classifier saved as obj/id/scikit-classifier-" + classifier_name + ".pkl")

        #WRITING ACCURACY ON FILE
        with open("report/{}.csv".format(classifier_name), "w+") as report_file:
            report_file.write("{0}; {1:.10f}%\r\n".format(1, accuracy))


def score_classifier_on_all_datatest(clf):
    print("\n==== Score Details ====")
    print(type(clf._classifier.named_steps["classifier"])) #prints the classifier type
    test_reader = load_object(GMB_IOB_TRIAD_PATH)
    accuracy = my_modified_scorer(clf, test_reader)
    print(" > Modified Accuracy: {0:.10f}%".format(accuracy*100))
    print("=======================")


if __name__ == "__main__":
    # data_train = load_object(GMB_RANDOM_5000_TRAIN_PATH)
    # data_test = load_object(GMB_RANDOM_2000_TEST_PATH)

    data_train = list(load_object(ID_IOB_TRIAD_PATH)[:400])
    data_test = list(load_object(ID_IOB_TRIAD_PATH)[400:480])

    train_perceptron(data_train, data_test,
                     all_classes=['O', 'B-LOCATION', 'I-LOCATION', 'B-ORGANIZATION',
                                  'I-ORGANIZATION', 'B-PERSON', 'I-PERSON']
                     )

    auto_train(
        "id_mlp_8-", 2, data_train, data_test,
        is_alternate=False,
        all_classes=['O', 'B-LOCATION', 'I-LOCATION', 'B-ORGANIZATION',
                    'I-ORGANIZATION', 'B-PERSON', 'I-PERSON'],
        batch_size= 200,
        verbose=0
    )

