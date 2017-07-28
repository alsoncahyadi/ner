from util import *
import pickle
from collections import Iterable
from nltk.tag import ClassifierBasedTagger
from nltk.chunk import ChunkParserI, conlltags2tree, tree2conlltags
import nltk
from nltk.corpus import state_union
# from nltk.tokenize import PunktSentenceTokenizer
from nltk import word_tokenize, pos_tag, ne_chunk
import ast
print("Done importing! ! !\n")

DATA_TRAIN_FILE_NAME = "dataTrain.txt"

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

	def parse_to_iob_tagged_tokens(self, tagged_sent):
		chunks = self.tagger.tag(tagged_sent)

		# Transform the result from [((w1, t1), iob1), ...] 
		# to the preferred list of triplets format [(w1, t1, iob1), ...]
		iob_triplets = [(w, t, c) for ((w, t), c) in chunks]

		# Transform the list of triplets to nltk.Tree format
		return iob_triplets

	def test_from_file(self, filename):
		test_samples = get_named_entities_from_file(filename)
		print("Testing from {}".format(filename))
		score = self.evaluate([conlltags2tree([(w, t, iob) for (w, t), iob in iobs]) for iobs in test_samples[:500]])
		print("Testing done.")
		print(score)

def test_gmb():
	corpus_root = "gmb-2.2.0"

	# print("Reading corpus. . .")
	# reader = read_gmb(corpus_root)
	# print("Converting corpus to list of data. . .")
	# data = list(reader)
	# print("Creating training and test samples. . .")
	data = load_object(GMB_IOB_NLTK_FORMAT_PATH)
	training_samples = data[:int(len(data) * 0.9)]
	test_samples = data[int(len(data) * 0.9):]

	print("#train: " + str(len(training_samples)))
	print("#test : " + str(len(test_samples)))

	print("Training Classifier. . .")
	chunker = NamedEntityChunker(training_samples[:2000])
	print("Testing Classifier. . .")
	print("\n> Evaluating first 500 entries")
	score = chunker.evaluate([conlltags2tree([(w, t, iob) for (w, t), iob in iobs]) for iobs in test_samples[:500]])
	print(score)
	print("\n> Evaluating last 500 entries")
	score = chunker.evaluate([conlltags2tree([(w, t, iob) for (w, t), iob in iobs]) for iobs in test_samples[-500:]])
	print(score)
	print("\n> Evaluating first 500 entries with modified scoring system")
	print("Accuracy: {}%".format(my_modified_scorer(chunker, ([(w, t, iob) for (w, t), iob in iobs] for iobs in test_samples[:500])) * 100))

def train_from_file(filename):
	training_samples = get_named_entities_from_file(filename)
	new_chunker = NamedEntityChunker(training_samples)
	return new_chunker

def test_from_file(filename, chunker):
	test_samples = get_named_entities_from_file(filename)
	print("Testing from {}".format(filename))
	score = chunker.evaluate([conlltags2tree([(w, t, iob) for (w, t), iob in iobs]) for iobs in test_samples[:500]])
	print("Testing done, accuracy: {0:.2f}%\n".format(score.accuracy() * 100))

test_gmb()

# chunker = train_from_file(DATA_TRAIN_FILE_NAME)
# chunker.test_from_file(DATA_TRAIN_FILE_NAME)
# tagged_sent = pos_tag(word_tokenize("last samurai's schedule please"))
# parse_result = chunker.parse(tagged_sent)
# print(parse_result)