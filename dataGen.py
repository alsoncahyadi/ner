from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
import pprint
import ast
from types import *

DATA_TRAIN_FILE_NAME = "dataTrain.txt"
TYPE_LIST = "<class 'list'>"

def check_ner(filename):
	print("Checking data train: {}".format(filename))
	myfile = open(filename, "r")
	sentences = myfile.readlines()
	for i, sentence in enumerate(sentences):
		data = ast.literal_eval(sentence)
		if not isinstance(data, list):
			print("line {}: Not a list, {} instead".format(i+1, type(sentence)))
			continue
		for j, word in enumerate(data):
			if not isinstance(word, tuple):
				print("line {}, word {}: Not a tuple, {} instead".format(i+1, j+1, type(word)))

def input_sentences():
	sentence = input("\nSentence: ")
	while (sentence != "quit"):
		tagged_sentence = pos_tag(word_tokenize(sentence))
		print("Tagged: {}".format(str(tagged_sentence)))
		named_entity = input_entity(tagged_sentence)
		print("After Named: {}".format(str(named_entity)))

		with open(DATA_TRAIN_FILE_NAME, "a") as myfile:
		    myfile.write(str(named_entity) + "\n")
		sentence = input("\nSentence: ")

def input_entity(tagged_sentence):
	#[((word, POStag), IOBtag)]
	print("ENTITY NAMING")
	named_entities = []
	for (word, tag) in tagged_sentence:
		entity_name = input("{} | {} | entity name: ".format(word[1], word[0]))
		named_entity = ((word, tag), entity_name)
		named_entities.append(named_entity)
	return named_entities

def input_entity1(tagged_sentence):
	#[(word, POStag, IOBtag)]
	print("ENTITY NAMING")
	named_entities = []
	for (word, tag) in tagged_sentence:
		entity_name = input("{} | {} | entity name: ".format(word[1], word[0]))
		named_entity = (word, tag, entity_name)
		named_entities.append(named_entity)
	return named_entities

check_ner(DATA_TRAIN_FILE_NAME)
input_sentences()