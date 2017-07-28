from bs4 import BeautifulSoup
from polyglot.text import Text
from util import *
import re
import itertools

def print_sentences(sentences):
    for i, sent in enumerate(sentences):
        print("\n\n======================== {} =========================".format(i))
        for word in sent:
            print(word)

def generate_iob_labelled_list():
    with open("ViraAdapter/enamex_file.txt") as fp:
        contents = fp.readlines()

    sentences_with_iob = []
    # sentences_with_pos_tag = load_object("obj/id/data/id_words+tags.pkl")

    for i, content in enumerate(contents):
        # Preprocess so tokenize work properly
        content = content.replace('<ENAMEX TYPE="', ' <ENAMEX TYPE="')
        content = content.replace("</ENAMEX>", "</ENAMEX> ")
        print("===",i,"===")
        soup = BeautifulSoup(content, "lxml")
        soup = soup.body
        if soup.p:
            soup = soup.p
        cnt = 0
        words, pos_tags = [], []
        text = Text(soup.text)
        text.language = "id"
        text.pos_tags
        for word in text.words:
            words.append(word.string)
            pos_tags.append(word.pos_tag)
        sentence_with_iob = []
        for content in soup.contents:
            if content.name == "enamex":
                content_text = Text(content.string)
                content_text.language = 'id'
                for j, word in enumerate(content_text.words):
                    iob_label = ""
                    if j == 0:
                        iob_label += "B-"
                    else:
                        iob_label += "I-"
                    iob_label += content['type']
                    sentence_with_iob.append((words[cnt], pos_tags[cnt], iob_label))
                    print((words[cnt], pos_tags[cnt], iob_label))
                    cnt += 1
            else:
                content_text = Text(content.string)
                content_text.language = 'id'
                for word in content_text.words:
                    print(cnt, word)
                    sentence_with_iob.append((words[cnt], pos_tags[cnt], "O"))
                    cnt += 1
        sentences_with_iob.append(sentence_with_iob)

    save_object(sentences_with_iob, "coba.pkl")

# do()
def generate_postagged_list():
    with open("../training_data_new.txt") as fp:
        contents = fp.readlines()
    sentences = []
    for content in contents:
        sentence = []
        soup = BeautifulSoup(content, "lxml")
        text = Text(soup.text)
        text.language = "id"
        text.pos_tags
        for word in text.words:
            sentence.append((word, word.pos_tag))
        sentences.append(sentence)
    save_object(sentences, ID_POS_TAGGED_PATH)



# generate_postagged_list()
generate_iob_labelled_list()
# iobs = load_object("coba.pkl")
# print_sentences(iobs)
