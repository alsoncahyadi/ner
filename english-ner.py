print("Importing. . .")
from util import read_gmb, features
import pickle
from collections import Iterable
from nltk.tag import ClassifierBasedTagger
from nltk.chunk import ChunkParserI
import nltk
from nltk.corpus import state_union
# from nltk.tokenize import PunktSentenceTokenizer
from nltk import word_tokenize, pos_tag, ne_chunk
print("Done importing! ! !")

def sandbox():
	sample_sents = state_union.sents("2005-GWBush.txt")
	for sent in sample_sents[:2]:
		print(sent)
		nered = ne_chunk(pos_tag(sent))
		print(nered)
		print("				===== END =====					\n")

class NamedEntityChunker(ChunkParserI):
    def __init__(self, train_sents, **kwargs):
        assert isinstance(train_sents, Iterable)
 
        self.feature_detector = features
        self.tagger = ClassifierBasedTagger(
            train=train_sents,
            feature_detector=features,
            **kwargs)
 
    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)
 
        # Transform the result from [((w1, t1), iob1), ...] 
        # to the preferred list of triplets format [(w1, t1, iob1), ...]
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]
 
        # Transform the list of triplets to nltk.Tree format
        return conlltags2tree(iob_triplets)

corpus_root = "gmb-2.2.0"

print("Reading corpus. . .")
reader = read_gmb(corpus_root)
print("Converting corpus to list of data. . .")
data = list(reader)
print("Creating training and test samples. . .")
training_samples = data[:int(len(data) * 0.9)]
test_samples = data[int(len(data) * 0.9):]

print("#train: " + str(len(training_samples)))
print("#test : " + str(len(test_samples)))

print("Creating Classifier. . .")
chunker = NamedEntityChunker(training_samples[:2000])
