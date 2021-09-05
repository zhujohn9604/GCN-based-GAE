import itertools as it
import re
from collections import Counter, defaultdict
#from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from tqdm import tqdm
import torch


def text_standardize(text):
    """
    fixes some issues the spacy tokenizer had on books corpus
    also does some whitespace standardization
    """
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('―', '-')
    text = text.replace('…', '...')
    text = text.replace('\\n', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('´', "'")
    text = re.sub(r'''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
    text = re.sub(r'\s*\n\s*', ' \n ', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text.strip()


def normalize_text(text, lower=True):
    text = str(text)
    text = text_standardize(text)
    if lower:
        text = text.lower()
    return ' '.join(filter(None, (''.join(c for c in w if c.isalnum())
                                  for w in text.split())))


class AlphaNumericTextPreprocessor(object):

    def __init__(self, max_features=None, lowercase=True, max_length=None,
                 stop_words=None, drop_unknown=False, dtype=None):

        self._max_features = max_features
        self._lowercase = lowercase
        self._max_length = max_length

        if isinstance(stop_words, str):
            if stop_words == 'english':
                self._stop_words = ENGLISH_STOP_WORDS
            else:
                raise ValueError("Stop words should be 'english', or a list or set!")
        elif isinstance(stop_words, (list, set, frozenset)):
            self._stop_words = set(stop_words)
        elif stop_words is None:
            self._stop_words = stop_words

        self._dtype = dtype
        self._drop_unknown = drop_unknown

        if drop_unknown:
            self._offset, self.unk_idx_ = 1, None
        else:
            self._offset, self.unk_idx_ = 2, 1

        self.padding_idx_ = 0

        self.vocabulary_ = None

    def _tokenize(self, raw_string):
        tokens = normalize_text(raw_string, lower=self._lowercase).split()
        if self._stop_words:
            tokens = [token for token in tokens if token not in self._stop_words]
        return tokens

    def _build_vocab(self, flat_tokens):
        counter = Counter(flat_tokens)
        most_common = counter.most_common(self._max_features)

        vocabulary = {word: i + self._offset for i, (word, _) in enumerate(most_common)}
        if not self._drop_unknown:
            vocabulary = defaultdict(lambda: 1, vocabulary)
        return vocabulary

    def _apply_vocab(self, doc):
        if self._drop_unknown:
            doc = [token for token in doc if token in self.vocabulary_]

        doc = [self.vocabulary_[token] for token in doc]
        return doc

    def _pad(self, batch):
        if self._max_length is not None:
            max_length = self._max_length
            batch = [doc[:max_length] for doc in batch]
        else:
            max_length = max(map(len, batch))

        batch = [doc + [self.padding_idx_] * (max_length - len(doc)) for doc in batch]

        return batch

    def fit(self, raw_documents):
        """Build a vocabulary"""
        docs = [self._tokenize(raw_doc) for raw_doc in raw_documents]
        self.vocabulary_ = self._build_vocab(it.chain.from_iterable(docs))

    def transform(self, raw_documents):
        docs = [self._tokenize(raw_doc) for raw_doc in raw_documents]
        docs_ix = [self._apply_vocab(doc) for doc in docs]
        docs_ix = self._pad(docs_ix)

        if self._dtype is None:
            return docs_ix

        return self._dtype(docs_ix)

    def fit_transform(self, raw_documents):
        print('Tokenizing...')
        docs = [self._tokenize(raw_doc) for raw_doc in tqdm(raw_documents, ncols=60, leave=True)]
        print('Building vocabulary...')
        self.vocabulary_ = self._build_vocab(it.chain.from_iterable(docs))
        print('Transforming into indices...')
        docs_ix = [self._apply_vocab(doc) for doc in tqdm(docs, ncols=60, leave=True)]
        docs_ix = self._pad(docs_ix)

        if self._dtype is None:
            return docs_ix

        return self._dtype(docs_ix)
