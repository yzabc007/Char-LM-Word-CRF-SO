import sys
import os
import numpy as np
from collections import Counter
import cPickle
import codecs

from utils import *


def load_sentences(file_path):
    """
    Load sentences from file
    """
    sents = []
    sent = []
    with open(file_path) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line == '':
                sents.append(sent)
                sent = []
                continue
            line = line.split()
            assert(len(line) >= 2)
            sent.append(line)
    return sents


def load_conll2003(path, zeros=None):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    # for line in codecs.open(path, 'r', 'utf8'):
    for line in open(path):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            if len(word) < 2:
                print(line)
                print(word)
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def word_mapping_glove(sents, Model_Parameters):
    #  ,embed_file, input_dim, lower=True
    words = [x[0].lower() if Model_Parameters['word_lower'] else x[0] for sent in sents for x in sent]
    vocab = Counter(words)

    def load_glove_vec(embed_filename_, vocab_, input_dim):
        """
    	Loads word vecs from gloVe
    	"""
        # vocab is a counter
        word_vecs_ = {}
        with open(embed_filename_) as f:
            for idx, line in enumerate(f):
                L = line.split()
                word = L[0].lower()
                if word in vocab_:
                    word_vecs_[word] = np.array(L[1::], dtype=np.float32)
                    assert(len(word_vecs_[word]) == input_dim)
        return word_vecs_

    def add_unknown_words(word_vecs_, vocab_, min_df=5, embed_dim=300, unk_token='<UNK>'):
        """
    	For words that occur in at least min_df documents, create a separate word vector.
    	0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    	"""
        vocab_size = len([x[0] for x in vocab.items() if x[1] >= min_df or x[1] in word_vecs_]) + 1
        drange = np.sqrt(6. / (np.sum([vocab_size, embed_dim])))
        for word in vocab_:
            if word not in word_vecs_ and vocab_[word] >= min_df:
                # word_vecs_[word] = np.random.uniform(-drange, drange, embed_dim)
                word_vecs_[word] = np.random.uniform(-0.25, 0.25, embed_dim)

        # word_vecs_[unk_token] = np.random.uniform(-drange, drange, embed_dim)
        word_vecs_[unk_token] = np.random.uniform(-0.25, 0.25, embed_dim)

    def get_embed_W(word_vecs_, embed_dim=300):
        """
    	Get word matrix. W[i] is the vector for word indexed by i
    	"""
        vocab_size = len(word_vecs_)
        word_to_id_ = {}
        id_to_word_ = {}
        W = np.zeros((vocab_size + 1, embed_dim))
        W[0] = np.zeros(embed_dim)
        i = 1
        for word in word_vecs_:
            W[i] = word_vecs_[word]
            word_to_id_[word] = i
            id_to_word_[i] = word
            i += 1
        return W, word_to_id_, id_to_word_

    word_vecs = load_glove_vec(Model_Parameters['pre_trained_path'], vocab, Model_Parameters['word_input_dim'])

    add_unknown_words(word_vecs, vocab, embed_dim=Model_Parameters['word_input_dim'])

    W, word_to_id, id_to_word = get_embed_W(word_vecs, embed_dim=Model_Parameters['word_input_dim'])

    print(W.shape)
    print('Find %i unique words on Glove (%i in total)' % (len(word_to_id), len(vocab)))
    return vocab, word_to_id, id_to_word, W


def word_mapping_random(sents):
    words = [x[0] for sent in sents for x in sent]
    vocab = Counter(words)
    vocab['<UNK>'] = sys.maxint
    sorted_vocab = vocab.most_common()
    word_to_id = {x[0]: idx for idx, x in enumerate(sorted_vocab)}
    id_to_word = {v: k for k, v in word_to_id.items()}
    print('Find %i unique words on training set (%i in total).' % (len(word_to_id), len(words)))
    return vocab, word_to_id, id_to_word


def tag_mapping(sents):
    tags = [x[-1] for sent in sents for x in sent]
    tag_vocab = Counter(tags).most_common()
    tag_to_id = {x[0]: idx for idx, x in enumerate(tag_vocab)}
    id_to_tag = {v: k for k, v in tag_to_id.items()}
    print('Find %i tags.' % (len(tag_to_id)))
    return tag_vocab, tag_to_id, id_to_tag


def char_mapping(sents):
    chars_list = list(" ".join([" ".join([w[0] for w in s]) for s in sents]))
    char_vocab = Counter(chars_list)
    sorted_char_vocab = char_vocab.most_common()
    char_to_id = {x[0]: idx + 1 for idx, x in enumerate(sorted_char_vocab)}
    char_to_id['<UNK>'] = len(char_to_id) + 1
    char_to_id['<s>'] = len(char_to_id) + 1
    char_to_id['</s>'] = len(char_to_id) + 1
    id_to_char = {v: k for k, v in char_to_id.items()}
    assert(' ' in char_to_id)
    # print '!!!!!!!', char_to_id[' ']
    print('Find %i character.' % (len(char_to_id)))
    return char_vocab, char_to_id, id_to_char


def lm_vocab_mapping(Model_Parameters, train_sents):
    vocab = []
    for sent in train_sents:
        vocab += [x[0].lower() if Model_Parameters['word_lower'] else x[0] for x in sent]
    sorted_vocab = Counter(vocab).most_common(Model_Parameters['lm_max_vocab'])
    final_vocab = [x[0] for x in sorted_vocab]
    word_to_id = {x: idx for idx, x in enumerate(final_vocab)}
    word_to_id[Model_Parameters['start']] = len(word_to_id)
    word_to_id[Model_Parameters['end']] = len(word_to_id)
    word_to_id[Model_Parameters['unk']] = len(word_to_id)
    id_to_word = {v: k for k, v in word_to_id.items()}

    return word_to_id, id_to_word


def insert_singletons(words, word_to_id, singletons, unk, lower, p=0.5):
    """
    Replace singletons by the unknown word with a probability p.
    """
    def f(x): return x.lower() if lower else x
    new_words = []
    for w in words:
        if w in singletons and np.random.uniform() < p:
            new_words.append(word_to_id[unk])
        else:
            new_words.append(word_to_id[f(w) if f(w) in word_to_id else unk])
    return new_words


def make_idx_data(Model_Parameters, sents, word_to_id, tag_to_id):
    '''
    convert strings to ids
    '''
    def f(x): return x.lower() if Model_Parameters['word_lower'] else x
    data = []
    for sent in sents:
        cur_sent_dict = {}
        seq_words = [w[0] for w in sent]
        if Model_Parameters['insert_singletons']:
            singletons = Model_Parameters['singletons']
            word_ids = insert_singletons(seq_words, word_to_id, singletons, Model_Parameters['unk'], Model_Parameters['word_lower'] )
        else:
            word_ids = [word_to_id[f(w) if f(w) in word_to_id else Model_Parameters['unk']] for w in seq_words]
        tag_ids = [tag_to_id[w[-1]] for w in sent]
        assert(len(word_ids) == len(tag_ids))
        # basic inputs, word id, sent lenght, tag id
        cur_sent_dict['seq_words'] = seq_words
        cur_sent_dict['word_ids'] = word_ids
        cur_sent_dict['tag_ids'] = tag_ids

        # add char id
        if Model_Parameters['char_encode'] != 'None' and not Model_Parameters['use_hier_char']:
            char_ids = []
            char_to_id = Model_Parameters['char_to_id']
            for idx, word in enumerate(seq_words):
                cur_chars = [char_to_id[c if c in char_to_id else '<UNK>'] for c in word]
                char_ids.append(cur_chars)

            cur_sent_dict['char_ids'] = char_ids
        elif Model_Parameters['use_hier_char']:
            # add one extra space character for each word
            char_ids = []
            char_to_id = Model_Parameters['char_to_id']
            for idx, word in enumerate(seq_words):
                if idx != len(seq_words) - 1:
                    cur_chars = [char_to_id[c if c in char_to_id else '<UNK>'] for c in word] + [char_to_id[' ']]
                else:
                    cur_chars = [char_to_id[c if c in char_to_id else '<UNK>'] for c in word]
                char_ids.append(cur_chars)

            cur_sent_dict['char_ids'] = char_ids
            # add char lm inputs, next/previous char, only valid when using hier_char model
            if Model_Parameters['char_lm']:
                flat_char_ids = [c for sent in char_ids for c in sent]
                cur_sent_dict['char_lm_forward'] = flat_char_ids[1:] + [char_to_id['</s>']]
                cur_sent_dict['char_lm_backward'] = [char_to_id['<s>']] + flat_char_ids[:-1]

        if Model_Parameters['word_lm']:
            lm_word_to_id = Model_Parameters['lm_word_to_id']
            next_words = seq_words[1::] + [Model_Parameters['end']]
            prev_words = [Model_Parameters['start']] + seq_words[:-1]
            cur_sent_dict['forward_words'] = [lm_word_to_id[f(w) if f(w) in lm_word_to_id else Model_Parameters['unk']] for w
                                              in next_words]
            cur_sent_dict['backward_words'] = [lm_word_to_id[f(w) if f(w) in lm_word_to_id else Model_Parameters['unk']] for w
                                               in prev_words]

        data.append(cur_sent_dict)
    return data
