def load_treebank_cnf_sents(reader):
    import nltk.corpus
    xs=reader.parsed_sents()
    for x in xs:
        try:
            x.chomsky_normal_form()   
            yield x
        except GeneratorExit:
                break
        except:
            pass
        
