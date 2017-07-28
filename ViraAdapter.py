import ast

from bs4 import BeautifulSoup
from polyglot.text import Text
from util import *
from sklearn_crfsuite import *

ENAMEX_FILE_NAME = 'enamex_file.txt'


class ViraAdapter:
    def __init__(self):
        self.iob_sentences = []
        self.enamex_string = ""
        self.json_string = ""

    def jsonstring2enamexstring(self, string_in):
        print("Converting Json String => Enamex String")
        self.iob_sentences = []
        self.json_string = string_in
        json_data = ast.literal_eval(string_in)
        enamex_string = ""
        for json_datum in json_data:
            # Extract entities from the json datum
            json_entities = json_datum['entities']
            entities = [
                entity
                for entity in json_entities
                if entity.get('start')
            ]
            currently_processed_entity = None
            if len(entities) != 0:
                currently_processed_entity = entities.pop(0)

            for i, character in enumerate(json_datum['text']):
                if currently_processed_entity:
                    if (i == currently_processed_entity['start']):
                        enamex_string += '<ENAMEX TYPE="{}">'.format(currently_processed_entity['entity'])

                enamex_string += character

                if currently_processed_entity:
                    if (i+1 == currently_processed_entity['end']):
                        enamex_string += "</ENAMEX>"
                        if len(entities) != 0:
                            currently_processed_entity = entities.pop(0)
                        else:
                            currently_processed_entity = None
            enamex_string += "\n"
        self.enamex_string = enamex_string
        return enamex_string


    def jsonfile2enamexstring(self, file_path):
        print("Converting Json File => Enamex String")
        with open(file_path, 'r') as input_file:
            string_in = input_file.read()
        self.jsonstring2enamexstring(string_in)

    def jsonfile2iobsentences(self, file_path):
        print("Converting Json File => IOB Sentences")
        self.jsonfile2enamexstring(file_path)
        return self.enamexstring2iobsentences(self.enamex_string)

    def enamexstring2iobsentences(self, string_in):
        print("Converting Enamex String => IOB Sentences")
        contents = string_in.split("\n")
        # Remove the last element (empty string because of endline on every last line)
        contents.pop(len(contents)-1)

        sentences_with_iob = []
        # sentences_with_pos_tag = load_object("obj/id/data/id_words+tags.pkl")

        for i, content in enumerate(contents):
            # Preprocess so tokenize work properly
            content = content.replace('<ENAMEX TYPE="', ' <ENAMEX TYPE="')
            content = content.replace("</ENAMEX>", "</ENAMEX> ")
            # print("===", i, "===")
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
                        # print(cnt, (words[cnt], pos_tags[cnt], iob_label))
                        cnt += 1
                else:
                    content_text = Text(content.string)
                    content_text.language = 'id'
                    for word in content_text.words:
                        # print(cnt, word)
                        sentence_with_iob.append((words[cnt], pos_tags[cnt], "O"))
                        cnt += 1
            sentences_with_iob.append(sentence_with_iob)
        self.iob_sentences = sentences_with_iob
        return sentences_with_iob


# Testing
if __name__ == "__main__":
    vira2iob = ViraAdapter()
    vira2iob.jsonfile2iobsentences("data/vira_example.json")
    print(vira2iob.iob_sentences)