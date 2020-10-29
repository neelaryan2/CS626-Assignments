import spacy
from nltk import Tree
from spacy import displacy

en_nlp = spacy.load('en')

sentences = [
	'I told him that I am going.',
	'He flew to the field and kicked and died painfully.',
	'He walked and kicked but talked cheerfully.',
	'The young and wild kid fell down.',
	'He went into the wild and through the civilization and towards the sun.',
	'The big and scary house shined under the sun.',
	'He eats mangoes and he plays but she runs fast.',
	'He ran quickly over a truck with a car.',
	'He gave me a slap swiftly.',
]

doc = en_nlp(sentences[-1])

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_

for token in doc:
    print(token.text, token.dep_, token.head.text, token.pos_, [child for child in token.children])

[to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]

def make_VP(node, wh_word=None):
    dict_VP = {}
    dict_VP['V'] = Tree('V', [node.text])
    for child in node.children:
        if child.dep_ == 'aux':
            dict_VP['aux'] = Tree('AUX', [child.text])
        if child.dep_ == 'dative':
            dict_VP['NP2'] = make_NP(child)
        if child.dep_ == 'dobj':
            dict_VP['NP1'] = make_NP(child)
        if child.dep_ in 'advmod' and (child.text != wh_word or wh_word == None):
            if 'RB' in dict_VP.keys():
                dict_VP['RB'].append(Tree('RB', [child.text]))
            else:
                dict_VP['RB'] = [Tree('RB', [child.text])]
        if child.dep_ == 'prep':
            if 'PP' in dict_VP.keys():
                dict_VP['PP'].append(make_PP(child))
            else:
                dict_VP['PP'] = [make_PP(child)]
        if child.dep_ in ['advcl', 'ccomp', 'xcomp']:
            wh_word = None
            wh_type = None
            for gchild in child.children:
                if gchild.dep_ == 'mark' or gchild.dep_ == 'advmod':
                    wh_word = gchild.text
                    wh_type = gchild.pos_
                    break
            sbar = make_S(child, True, wh_word)[0]
            if 'SBAR' in dict_VP.keys():
                dict_VP['SBAR'].append(Tree('SBAR', [Tree(wh_type, [wh_word]), sbar]))
            else:
                dict_VP['SBAR'] = [Tree('SBAR', [Tree(wh_type, [wh_word]), sbar])]
    VP = VP_d2t(dict_VP)
    return VP


def VP_d2t(dict_VP):
    children = []
    if 'aux' in dict_VP.keys():
        children.append(dict_VP['aux'])
    children.append(dict_VP['V'])
    if 'NP2' in dict_VP.keys():
        children.append(dict_VP['NP2'])
    if 'NP1' in dict_VP.keys():
        children.append(dict_VP['NP1'])
    if 'RB' in dict_VP.keys():
        children += dict_VP['RB']
    if 'PP' in dict_VP.keys():
        children += dict_VP['PP']
    if 'SBAR' in dict_VP.keys():
        children += dict_VP['SBAR']
    final = Tree('VP', children)
    return final


def make_ADJP(node, init=True):
    cc_found = False
    jjword = node.text
    cc_node = None
    cc_conj = None
    for child in node.children:
        if child.dep_ == 'cc':
            cc_found = True
            cc_node = Tree('CC', [child.text])
        if child.dep_ == 'conj':
            cc_conj = make_ADJP(child, False)
    if cc_found:
        children = [Tree('JJ', [jjword]), cc_node]
        [children.append(x) for x in cc_conj]
        return Tree('ADJP', children)
    elif init:
        return Tree('JJ', [jjword])
    else:
        return Tree('ADJP', [Tree('JJ', [jjword])])


def make_NP(node, init=True):
    dict_NP = {}
    cc_node = None
    conj_NP = None
    dict_NP['N'] = Tree('N', [node.text])
    for child in node.children:
        if child.dep_ == 'det':
            dict_NP['DET'] = Tree('DET', [child.text])
        if child.dep_ == 'amod':
            if 'JJ' in dict_NP.keys():
                dict_NP['JJ'].append(make_ADJP(child))
            else:
                dict_NP['JJ'] = [make_ADJP(child)]
        if child.dep_ == 'prep':
            if 'PP' in dict_NP.keys():
                dict_NP['PP'].append(make_PP(child))
            else:
                dict_NP['PP'] = [make_PP(child)]

    for child in node.children:
        if child.dep_ == 'cc':
            cc_node = Tree('CC', [child.text])
        if child.dep_ == 'conj':
            conj_NP = make_NP(child, False)

    NP = NP_d2t(dict_NP)
    if cc_node != None and conj_NP != None:
        finalNP = Tree('NP', [NP, cc_node] + list(conj_NP))
        return finalNP
    elif cc_node != None:
        finalNP = Tree('NP', [NP, cc_node])
        return finalNP
    elif conj_NP != None:
        finalNP = Tree('NP', [NP] + list(conj_NP))
        return finalNP
    if init:
        return NP
    else:
        return Tree('NP', [NP])


def NP_d2t(dict_NP):
    children = []
    if 'DET' in dict_NP.keys():
        children.append(dict_NP['DET'])
    if 'JJ' in dict_NP.keys():
        children += dict_NP['JJ']
    children.append(dict_NP['N'])
    if 'PP' in dict_NP.keys():
        children += dict_NP['PP']

    final = Tree('NP', children)
    return final


def make_PP(node, init=True):
    np_obj = None
    cc_node = None
    conj_node = None
    cc_found = False
    for child in node.children:
        if child.dep_ == 'pobj':
            np_obj = make_NP(child)
        if child.dep_ == 'cc':
            cc_found = True
            cc_node = Tree('CC', [child.text])
        if child.dep_ == 'conj':
            conj_node = make_PP(child, False)
    if cc_found:
        children = [Tree('PP', [Tree('IN', [node.text]), np_obj])]
        [children.append(x) for x in conj_node]
        return Tree('PP', children)
    elif init:
        return Tree('PP', [Tree('IN', [node.text]), np_obj])
    else:
        return Tree('PP', [Tree('PP', [Tree('IN', [node.text]), np_obj])])


def make_S(node, init=True, wh_word=None):
    VP = make_VP(node, wh_word)
    if len(list(VP)) > 1:
        VP = Tree('VP', [VP])
    full_sent = [VP]

    cc_node = None
    conj_node = None
    cctemp = None
    subexists = False
    for child in node.children:
        if child.dep_ == 'cc':
            cc_node = Tree('CC', [child.text])
    for child in node.children:
        if child.dep_ == 'conj':
            subexists = False
            for gchild in child.children:
                if gchild.dep_ == 'nsubj':
                    subexists = True
                    break
            if True:
                conj_node, conj_subexists = make_S(child, False)
            else:
                conj_node = make_VP(child)

    for child in node.children:
        if child.dep_ == 'nsubj':
            subexists = True
            NP = make_NP(child)
    if subexists:
        full_sent = [NP] + full_sent
    if cc_node != None:
        if conj_subexists or conj_node == None:
            full_sent.append(cc_node)
        elif subexists:
            newvplist = []
            [newvplist.append(x) for x in full_sent[1]]
            newvplist.append(cc_node)
            full_sent[1] = Tree(full_sent[1].label(), newvplist)
        else:
            newvplist = []
            [newvplist.append(x) for x in full_sent[0]]
            newvplist.append(cc_node)
            full_sent[0] = Tree(full_sent[0].label(), newvplist)

    if conj_node != None:
        if conj_subexists:
            [full_sent.append(x) for x in conj_node]
        elif subexists:
            #add cc and conj_node to chidlren of vp at pos 1
            newvplist = []
            [newvplist.append(x) for x in full_sent[1]]
            [newvplist.append(x) for x in conj_node]
            full_sent[1] = Tree(full_sent[1].label(), newvplist)
        else:
            newvplist = []
            [newvplist.append(x) for x in full_sent[0]]
            [newvplist.append(x) for x in conj_node]
            full_sent[0] = Tree(full_sent[0].label(), newvplist)
    if init:
        if subexists:
            if len(full_sent) > 2:

                final = Tree('S', [Tree('S', [full_sent[0], full_sent[1]])] + full_sent[2:])
            else:
                if len(list(full_sent[1])) == 1:
                    full_sent[1] = list(full_sent[1])[0]
                final = Tree('S', full_sent)
        else:
            final = Tree('S', full_sent)
        return final, subexists
    else:
        if subexists:
            if len(full_sent) > 2:
                final = Tree('S', [Tree('S', [full_sent[0], full_sent[1]])] + full_sent[2:])
            else:
                final = Tree('S', [Tree('S', full_sent)])
        else:
            final = full_sent[0]
        return final, subexists

[make_S(sent.root)[0].pretty_print() for sent in doc.sents]
