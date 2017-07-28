MITIELIB_PATH = "/home/alson/Desktop/ner/MITIE-master/mitielib"
MITIEMODEL_PARENT_PATH = "/home/alson/Desktop/ner/MITIE-master/MITIE-models/english"
print("Importing. . .")
# sys.path.append(MITIELIB_PATH)
from mitie import *

import pickle
from util import *
from nltk.tag import ClassifierBasedTagger
from nltk.corpus import state_union
# from nltk.tokenize import PunktSentenceTokenizer
from nltk import word_tokenize, pos_tag, ne_chunk
import nltk
import ast


def load_gmb_corpus():
    print("Loading GMB corpus. . .")
    data = load_object(GMB_IOB_TRIAD_PATH)
    print("Done loading GMB corpus! ! !\n")
    return data


def train(data_train, file_label):
    trainer = ner_trainer(MITIEMODEL_PARENT_PATH + "/total_word_feature_extractor.dat")

    num_samples_to_train = int(input("Input number of training samples: "))

    print("Adding data train samples to trainer. . .")
    for i, datum_train in enumerate(data_train[:num_samples_to_train]):
        sentence, _, _ = list(zip(*datum_train))
        sample = ner_training_instance(sentence)
        # entity: ('Wonder Woman', 'movie_title', token begin, token end + 1)
        entities = get_entities_from_iob_tagged_tokens1(datum_train)
        # print(entities)
        for entity in entities:
            sample.add_entity(xrange(entity[2], entity[3]), entity[1])
        trainer.add(sample)
    print("Done adding data train samples! ! !\n")

    print("Training model. . .")
    trainer.num_threads = 4
    ner = trainer.train()
    print("Done training! ! !\n")

    ner.save_to_disk("obj/random/mitie-" + file_label + ".dat")


def test(data_test):
    file_label = input("Input model label to be tested: ")
    ner = named_entity_extractor("obj/random/mitie-" + str(file_label) + ".dat")

    correct_entities = 0
    total_entities = 0

    for datum_test in data_test:
        words, tags, labels = zip(*datum_test)
        raw_predicted_entities = ner.extract_entities(words)
        predicted_entities = [mitieentity2myentity(predicted_entity, words) for predicted_entity in
                              raw_predicted_entities]
        datatest_entities = get_entities_from_iob_tagged_tokens1(datum_test)

        # print("\nACTUAL   :", datatest_entities)
        # print("PREDICTED:", predicted_entities)

        total_entities += len(datatest_entities)
        for datatest_entity in datatest_entities:
            if datatest_entity in predicted_entities:
                correct_entities += 1

    if total_entities:
        print("Accuracy: {0:.10f}%".format((correct_entities / total_entities) * 100))
    else:
        print("Dividing by 0")


def mitieentity2myentity(raw_entity, tokens):
    words_in_chunk = [word for word in tokens[raw_entity[0].start: raw_entity[0].stop]]
    return (" ".join(words_in_chunk), raw_entity[1], raw_entity[0].start, raw_entity[0].stop)


# train()
print("Done importing! ! !\n")

# data = load_gmb_corpus()[5000:1000]
data_train = load_object(ID_IOB_TRIAD_PATH)[:400]
data_test = load_object(ID_IOB_TRIAD_PATH)[400:480]

train(data_train, "id")
test(data_test)
