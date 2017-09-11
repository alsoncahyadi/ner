# Named Entity Recognition Exploration
This is a thorough documentation of my exploration on NER.
## Table of contents
1. Planning phase
	1. Dataset
	2. Tools comparison
2. Implementation phase
	1. Preparing the dataset
	2. Training with MITIE
	3. Training with Scikit Learn
	4. Training with sklearn-crfsuite

## Prerequisites

## Planning phase

## Implementation phase

### Preparing the dataset
#### CoNLL structure
The dataset structure used on the interface is a CoNLL type. Which has three levels
1. Sentences level (list of sentences)
```python
```
2. Words/Sentence level (list of words)
3. Word level (a set of word, POS tag, and IOB label)

This is an example of the CoNLL structure described above
```python
[ # Sentences level
	[ # Words/Sentence level
		('I', 'NN', 'O'), # Word level
		(),
		()
	],
	[
		(),
		(),
		()
	]
]
```

The procedure defined to convert read GMB dataset and directly convert it to CoNLL dataset is in the file ```/util.py```

#### Scikit Learn's readable dataset structure
The structure of which scikit learn is able to read is the mapping of features (X) and the tag (y).

### Training with MITIE

### Training with Scikit Learn

### Training with sklearn-crfsuite