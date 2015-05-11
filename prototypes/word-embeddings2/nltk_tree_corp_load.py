def load_treebank_cnf_sents():
    import nltk.corpus
    xs=nltk.corpus.treebank.parsed_sents()
    for x in xs:
        x.chomsky_normal_form()
        yield x
