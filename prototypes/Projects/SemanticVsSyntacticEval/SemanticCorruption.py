 

import nltk
from nltk.corpus import wordnet as wn
import pattern.en as en

import itertools
import random
import copy

#--------UTIL
def n_wise(n, seq): #TODO move this into an util.py file
    """ Generalisation of pairwise"""
    from collections import deque
    prevs = deque(maxlen=n)
    for ii,ele in enumerate(seq,1):
        prevs.append(ele)
        if ii>=n:
            yield tuple(prevs)
        



#------------ Load the POS tagger

from nltk.parse import stanford
from nltk.tag.stanford import POSTagger

import os
os.environ['CLASSPATH'] = '/home/wheel/oxinabox/nltk_data/standford_models/stanford-postagger/stanford-postagger.jar'
os.environ['STANFORD_MODELS'] = '/home/wheel/oxinabox/nltk_data/standford_models/stanford-postagger/models/'


standford_pos_tagger = POSTagger("english-bidirectional-distsim.tagger")
def pos_tag(words):
    return standford_pos_tagger.tag(words)[0]
    
def tokenize_and_tag(sent):
    tokens = nltk.tokenize.word_tokenize(sent)
    return tokens, pos_tag(tokens)



#--------- Fixing up things 

def fix_indefinite_articles(words):
    """Alters a list of words in place so that the indefinate articles are correct. Eg replacing "An man" with "A man" """
    for ii in range(0,len(words)-1): #don't do the last word, as it can't ne an 'an' or an 'a'
        if words[ii] in frozenset(['an','a','An','A']):
            referenced_form = en.referenced(words[ii+1]) #Smarter than simple vowel match eg "a yak" not "an yak"
            replacement_article = referenced_form.split()[0] #'an' or 'a'
            if words[ii][0]=='A':
                replacement_article[0] == 'A' #Uppercase it
            words[ii]=replacement_article

    return words
    
    
def unstem_fun(pos_tag):
    """
    Handles the restemming of a particular POS tag after it has been converted to a Stem via wordnet lemmaisation.

    pos_tag is from https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html eg VBD
    
    This can be extended as required from https://www.nodebox.net/code/index.php/Linguistics
    """

    unstem_funs = {frozenset(['NNS', 'NNPS']) : en.pluralize,
                  frozenset(['RBR', 'JJR']) : en.comparative,
                  frozenset(['JJS']) : en.superlative, #Skip RBS, as ("Most") not changed by WordNet
                  frozenset(['VBD', 'VBN']) :  lambda w: en.conjugate(w, en.PAST), # A lot more of these can be made with en.conugate
                  frozenset(['VBG']) :  lambda w: en.conjugate(w,en.PRESENT,aspect=en.PROGRESSIVE ),
                 }
    
    for category in unstem_funs.keys():
        if pos_tag in category:
            return unstem_funs[category]
    else:
        return lambda x: x

valid_words = set(open("/usr/share/dict/british-english-insane").read().split())    #Use the SCOWL wordlist british-insane http://wordlist.aspell.net/scowl-readme/
def is_real_word(word):
    return word in valid_words
    


#----------- Thinking about short phrases, eg Chief Financial Officer
def is_in_wordnet(word):
    """Check that the word is valid, by seeing if it is in wordnet"""
    return len(wn.synsets(word))>0

def get_phrases(tagged_sent, max_phrase_length):
    for phrase_len in range(2, max_phrase_length+1):
        for tuple_of_tagged_words in n_wise(phrase_len, tagged_sent):
            words, tags = zip(*tuple_of_tagged_words)
            possible_phrase = "_".join(words)
            if is_in_wordnet(possible_phrase): #If there are any, then it is a phrase
                phrase = possible_phrase.split("_")
                pos = tags[-1]
                yield(phrase, pos)
                
def get_phrases_indexes(tagged_sent, max_phrase_length):
    phrase_indexes = set()
    for phrase_len in range(max_phrase_length,1,-1): #Go from largest to smallest to keep information
        for indexes in n_wise(phrase_len, range(len(tagged_sent))):
            tagged_words = [tagged_sent[index] for index in indexes]
            if not(any([index in phrase_indexes for index in indexes])): #If we haven't already got it covered
                words, tags = zip(*tagged_words)
                possible_phrase = "_".join(words)
                if is_in_wordnet(possible_phrase): #If there are any, then it is a phrase
                    phrase_indexes.update(indexes)
    return phrase_indexes
                    
def get_tagged_phrases(tagged_sent, max_phrase_length):
    tagged_phrase_sent = list(tagged_sent)
    for phrase_len in range(max_phrase_length,1,-1): #Go from largest to smallest to keep information
        for indexes in n_wise(phrase_len, range(len(tagged_sent))):
            tagged_words = [tagged_phrase_sent[index] for index in indexes]
            if not(any([tagged_word is None for tagged_word in tagged_words])):
                words, tags = zip(*tagged_words)
                possible_phrase = "_".join(words)
                if is_in_wordnet(possible_phrase): #If there are any, then it is a phrase
                    for index in indexes:
                        tagged_phrase_sent[index] = None #Blank them out with Nones which we will remove later
                    pos = tags[-1] #Use final tag, it will be the one we need for handling plurals
                    tagged_phrase_sent[indexes[0]] = (possible_phrase, pos)
    return [tagged_phrase for tagged_phrase in tagged_phrase_sent if not tagged_phrase is None]
                    



#-------------- Get all the ways we can corrupt it

def get_all_antonyms(word, pos=None):
    synsets = wn.synsets(word, pos=pos)
    for synset in synsets:
        for lemma in synset.lemmas():
            for anto in lemma.antonyms():
                yield anto.name()

def get_all_synonyms(word, pos=None):
    synsets = wn.synsets(word, pos=pos)
    for synset in synsets:
        for lemma_name in synset.lemma_names():
            yield lemma_name

                

#These constants define the types that I am interested in, as well as what POS tags they have for what wordnet tags
NOUN_POS_TAGS = frozenset(["NN", "NNS"])
ADJ_POS_TAGS = frozenset(["JJ","JJS", "JJR", "VBN"]) #VBN is here because it is hard to tell the difference between a VERB PAST PARTICPANT and an ADJECTIVE
VERB_POS_TAGS = frozenset(["VB","VBS", "VBN","VBG", "VBD"]) 
ADVERB_POS_TAGS = frozenset(["RB","RBS"])



BANNED_AUXILIARY_VERBS = frozenset(["be", "am", "are", "is", "was", "were", "being", "can", "could", "do", "did", "does", "doing", "have", "had", "has", "having", "may", "might", "must", "shall", "should", "will", "would"])
    #Changing these words tends to have huge impact on sentence, and they are had to change correctly

def get_pos_sub_function(pos_tag_set, wordnet_tag, sub_generator):
    def get_subs(tagged_words, skip_indexes=set()):
        for ii,(pword,p_pos_tag) in enumerate(tagged_words):
            if p_pos_tag in pos_tag_set and not(pword in BANNED_AUXILIARY_VERBS) and not(ii in skip_indexes):
                unstem = unstem_fun(p_pos_tag)

                sub = set(sub_generator(pword, wordnet_tag))
                sub = map(unstem,sub)
                sub = filter(is_real_word, sub) #Must be in wordnet -- ensure it is a real word, not a mistake introduced by unstemming.
                sub = filter(lambda w:not('_' in w), sub) #some WordNet lemmas are not single words. We don't use them.
                sub = filter(lambda w:not(w==pword), sub) #No subs that make no change
                sub = list(sub)
                if len(sub)>0:
                    yield(ii, sub)
    return get_subs

#-------



#Define the functions: all take sequence of words as parameter
get_noun_synonyms = get_pos_sub_function(NOUN_POS_TAGS, wn.NOUN, get_all_synonyms)
get_adj_synonyms = get_pos_sub_function(ADJ_POS_TAGS, wn.ADJ, get_all_synonyms)
get_verb_synonyms = get_pos_sub_function(VERB_POS_TAGS, wn.VERB, get_all_synonyms)
get_adverb_synonyms = get_pos_sub_function(ADVERB_POS_TAGS, wn.ADV, get_all_synonyms)

get_noun_antos = get_pos_sub_function(NOUN_POS_TAGS, wn.NOUN, get_all_antonyms)
get_adj_antos = get_pos_sub_function(ADJ_POS_TAGS, wn.ADJ, get_all_antonyms)
get_verb_antos = get_pos_sub_function(VERB_POS_TAGS, wn.VERB, get_all_antonyms)
get_adverb_antos = get_pos_sub_function(ADVERB_POS_TAGS, wn.ADV, get_all_antonyms)



#-------------------------- Do the Corupting

def all_antonym_corruptions(tagged_words):
    return itertools.chain(get_adj_antos(tagged_words),
                                get_noun_antos(tagged_words),
                                get_adverb_antos(tagged_words),
                                get_verb_antos(tagged_words),
                               )



def leveled_semantic_corrupt_sentences(sent, get_corruptions, skip_indexes=set()):
    words = nltk.tokenize.word_tokenize(sent)
    tagged_words = pos_tag(words)
    return leveled_semantic_corrupt_sentences_from_pretagged(words,tagged_words,get_corruptions,skip_indexes)
    
def leveled_semantic_corrupt_sentences_from_pretagged(words, tagged_words, get_corruptions,skip_indexes=set()):
    corruptions = list(get_corruptions(tagged_words,skip_indexes))
    random.shuffle(corruptions)
    
    for corrupt_index, antos in corruptions:
        words[corrupt_index] = random.sample(antos,1)[0]
        fix_indefinite_articles(words)
        yield " ".join(words)
    
