import spacy
from nltk import Tree
from spacy import displacy

s=input('Enter sentence: ')
en_nlp = spacy.load('en')
# doc=en_nlp("He is charming")
doc=en_nlp(s)
# doc=en_nlp("The quick brown fox jumps over the lazy dog")
# doc=en_nlp("The man to whom I talked was wearing a hat")
# doc=en_nlp("I ate the apple which fell from the tree")
# doc=en_nlp("The seats were sold out fast")
# doc=en_nlp("My parents have been very helpful with my work")
# doc=en_nlp("His car zoomed past me")
# doc=en_nlp("Book the flight through Houston")
# doc=en_nlp("The man who I met told me that he is leaving")
# doc=en_nlp("I am going where there is happiness")
# doc=en_nlp("The bag which I liked was stolen")
# doc=en_nlp("The woman whom I gave a slap ran away")
# doc=en_nlp("The man who was tall shook my hand")
# doc=en_nlp("I told him that I am going")
# doc=en_nlp("He flew to the field and kicked and died painfully")
# doc=en_nlp("He walked and kicked but talked cheerfully")
# doc=en_nlp("The young and wild kid fell down")
# doc=en_nlp("He went into the wild and through the civilization and towards the sun")
# doc = en_nlp("The big and scary house shined under the sun")	
# doc = en_nlp("He eats mangoes and he plays but she runs fast")
# doc = en_nlp("He ran quickly over a truck with a car")
# doc = en_nlp("He gave me a slap swiftly")

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_
# for token in doc:
#     print(token.text, token.dep_, token.head.text, token.pos_,
#             [child for child in token.children])

[to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]

# displacy.serve(doc, style='dep')
# print(list(doc.sents)[0].root.orth_)
# print()

def make_VP(node,wh_word=None,noun_dobj_sbar=False,dobj_word=None):
	dict_VP = {}
	dict_VP['V'] = Tree('V', [node.text])
	for child in node.children:
		if child.dep_=='attr':
			dict_VP['ATTR']=make_NP(child)
		if child.dep_=='aux':
			dict_VP['aux'] =dict_VP.get('aux',[])+[Tree('AUX', [child.text])]
		if child.dep_=='dative':
			dict_VP['NP2'] = make_NP(child)
		if child.dep_=='acomp':
			dict_VP['ACOMP']=make_ADJP(child)
		if child.dep_=='neg':
			dict_VP['NEG']=Tree(child.pos_,[child.text])
		if child.dep_=='dobj' and not (noun_dobj_sbar and child.text==dobj_word):
			dict_VP['NP1'] = make_NP(child)
		if child.dep_ in 'advmod' and (child.text!=wh_word or wh_word==None):
			if 'RB' in dict_VP.keys():
				dict_VP['RB'].append(Tree('RB', [child.text]))
			else:
				dict_VP['RB'] = [Tree('RB', [child.text])]
		if child.dep_=='prep':
			if 'PP' in dict_VP.keys():
				dict_VP['PP'].append(make_PP(child))
			else:
				dict_VP['PP'] = [make_PP(child)]
		if child.dep_ in ['advcl','ccomp','xcomp']:
			wh_word=None
			wh_type=None
			for gchild in child.children:
				if gchild.dep_=='mark' or gchild.dep_=='advmod':
					wh_word=gchild.text
					wh_type=gchild.pos_
					break
			# nc=list(child)
			# nc=[x for x in child.children if x.text!=wh_word]
			# sbar=Tree(child.dep_,nc)
			sbar=make_S(child,True,wh_word)[0]
			# sbartree=list(sbar)
			# print(sbartree)
			# sbartree=[print(x) for x in sbartree if list(x)[0]!=wh_word]
			# sbar=Tree(sbar.label(),sbartree)
			# sbar.pretty_print
			if 'SBAR' in dict_VP.keys():
				dict_VP['SBAR'].append(Tree('SBAR',[Tree(wh_type,[wh_word]),sbar]))
			else:
				dict_VP['SBAR']=[Tree('SBAR',[Tree(wh_type,[wh_word]),sbar])]
	VP = VP_d2t(dict_VP)
	return VP


def VP_d2t(dict_VP):
	children = []
	# children.append(Tree('V', [node.text]))
	if 'aux' in dict_VP.keys():
		children+=dict_VP['aux']
	# print(dict_VP['V'][0])
	if dict_VP['V'][0] in ['am','is','were','will','are','was']:
		children.append(dict_VP['V'])
		if 'NEG' in dict_VP.keys():
			children.append(dict_VP['NEG'])
	else:
		if 'NEG' in dict_VP.keys():
			children.append(dict_VP['NEG'])
		children.append(dict_VP['V'])
	if 'ATTR' in dict_VP.keys():
		children.append(dict_VP['ATTR'])
	if 'NP2' in dict_VP.keys():
		children.append(dict_VP['NP2'])
	if 'NP1' in dict_VP.keys():
		children.append(dict_VP['NP1'])
	if 'ACOMP' in dict_VP.keys():
		children.append(dict_VP['ACOMP'])
	if 'RB' in dict_VP.keys():
		children+=dict_VP['RB']
	if 'PP' in dict_VP.keys():
		children+=dict_VP['PP']
	if 'SBAR' in dict_VP.keys():
		children+=dict_VP['SBAR']
	final = Tree('VP', children)
	return final

def make_ADJP(node,init=True):
	cc_found=False
	jjword=node.text
	cc_node=None
	cc_conj=None
	acomp_found=False
	acomp_node=None
	for child in node.children:
		if child.dep_=='cc':
			cc_found=True
			cc_node=Tree('CC',[child.text])
		if child.dep_=='conj':
			cc_conj=make_ADJP(child,False)
		if child.dep_=='advmod' or child.dep_=='neg':
			# print('yay')
			acomp_found=True
			acomp_node=Tree(child.pos_,[child.text])
	if cc_found:
		children=[Tree('JJ',[jjword]),cc_node]
		[children.append(x) for x in cc_conj]
		if acomp_found:
			children=[acomp_node]+children
		return Tree('ADJP',children)
	elif init:
		if acomp_found:
			return Tree('ADJP',[acomp_node,Tree('JJ',[jjword])])
		else:
			return Tree('JJ',[jjword])
	else:
		if acomp_found:
			return Tree('ADJP',[acomp_node,Tree('JJ',[jjword])])
		else:
			return Tree('ADJP',[Tree('JJ',[jjword])])

def make_NP(node,init=True):
	dict_NP={}
	cc_node=None
	conj_NP=None
	dict_NP['N'] = Tree('N', [node.text])
	for child in node.children:
		if child.dep_=='det':
			dict_NP['DET']=Tree('DET', [child.text])
		if child.dep_=='amod':
			if 'JJ' in dict_NP.keys():
				dict_NP['JJ'].append(make_ADJP(child))
			else:
				dict_NP['JJ'] = [make_ADJP(child)]
		if child.dep_=='prep':
			if 'PP' in dict_NP.keys():
				dict_NP['PP'].append(make_PP(child))
			else:
				dict_NP['PP'] = [make_PP(child)]
		if child.dep_=='poss':
			dict_NP['POSS']=[Tree('PRP$',[child.text])]
		if child.dep_=='relcl':
			# print('yay')
			wh_word=None
			wh_type=None
			sbar=None
			for gchild in child.children:
				# print(gchild.text+gchild.dep_)
				if gchild.dep_=='nsubj' and gchild.text in ['who','which','whom']:

					wh_word=gchild.text
					wh_type=gchild.pos_
					sbar=make_S(child,True,wh_word=None,noun_dobj_sbar=False,noun_sub_sbar=True,nsubj_word=wh_word)[0]
					break
				elif gchild.dep_=='dobj' and gchild.text in ['whom','which','who']:
					wh_word=gchild.text
					wh_type=gchild.pos_
					sbar=make_S(child,True,wh_word=None,noun_dobj_sbar=True,noun_sub_sbar=False,dobj_word=wh_word)[0]
					break
			if sbar!=None:
				dict_NP['relcl']=[Tree('sbar',[Tree(wh_type,[wh_word]),sbar])]
			
		
	for child in node.children:
		if child.dep_=='cc':
			cc_node=Tree('CC',[child.text])
		if child.dep_=='conj':
			conj_NP=make_NP(child,False)

	NP = NP_d2t(dict_NP)
	if cc_node!=None and conj_NP!=None:
		# finalNP=Tree('NP',[NP,cc_node,conj_NP])
		finalNP=Tree('NP',[NP,cc_node]+list(conj_NP))
		return finalNP
		# return finalNP if init else Tree('NP',[finalNP])
	elif cc_node!=None:
		finalNP=Tree('NP',[NP,cc_node])
		return finalNP
		# return finalNP if init else Tree('NP',[finalNP])
	elif conj_NP!=None:
		# finalNP=Tree('NP',[NP,conj_NP])
		finalNP=Tree('NP',[NP]+list(conj_NP))
		return finalNP
		# return finalNP if init else Tree('NP',[finalNP])
	# print('NP')
	# NP.pretty_print()
	if init:
		return NP
	else:
		return Tree('NP',[NP])


def NP_d2t(dict_NP):
	children=[]
	if 'POSS' in dict_NP.keys():
		children.append(dict_NP['POSS'][0])
	if 'DET' in dict_NP.keys():
		children.append(dict_NP['DET'])
	if 'JJ' in dict_NP.keys():
		children+=dict_NP['JJ']
	children.append(dict_NP['N'])
	if 'PP' in dict_NP.keys():
		children+=dict_NP['PP']
	if 'relcl' in dict_NP.keys():
		children+=dict_NP['relcl']

	final = Tree('NP', children)
	return final


def make_PP(node,init=True):
	# children = []
	# children.append(Tree('IN', [node.text]))
	np_obj=None
	cc_node=None
	conj_node=None
	cc_found=False
	for child in node.children:
		if child.dep_=='pobj':
			np_obj=make_NP(child)
		if child.dep_=='cc':
			cc_found=True
			cc_node=Tree('CC',[child.text])
		if child.dep_=='conj':
			conj_node=make_PP(child,False)
	if cc_found:
		# print('yay')
		# print(conj_node)
		children=[Tree('PP',[Tree('IN', [node.text]),np_obj])]
		children.append(cc_node)
		[children.append(x) for x in conj_node]
		return Tree('PP',children)
	elif init:
		return Tree('PP', [Tree('IN', [node.text]),np_obj])
	else:
		return Tree('PP',[Tree('PP', [Tree('IN', [node.text]),np_obj])])


def make_S(node,init=True,wh_word=None,noun_sub_sbar=False,noun_dobj_sbar=False,dobj_word=None,nsubj_word=None):
	VP = make_VP(node,wh_word,noun_dobj_sbar=noun_dobj_sbar,dobj_word=dobj_word)
	# VP.pretty_print()
	if len(list(VP))>1:
		VP=Tree('VP',[VP])
	# print(len(list(VP)))
	full_sent=[VP]
	tmpsubjword=None
	cc_node=None
	conj_node=None
	cctemp=None
	subexists=False
	for child in node.children:
		if child.dep_=='cc':
			cc_node=Tree('CC',[child.text])
	for child in node.children:
		if child.dep_=='conj':
			subexists=False
			for gchild in child.children:
				if gchild.dep_=='nsubj':
					subexists=True
					break
			if True:
				conj_node,conj_subexists=make_S(child,False)
				# conj_node.pretty_print()
			else:
				conj_node=make_VP(child)

			# full_sent.append(Stemp)
	for child in node.children:
		if child.dep_=='nsubj':
			tmpsubjword=child.text
			subexists=True
			# if wh_word!=None and wh_word!=
			NP = make_NP(child)
	# print(full_sent)
	if subexists and not (noun_sub_sbar and tmpsubjword==nsubj_word and tmpsubjword!=None and nsubj_word!=None):
		full_sent=[NP]+full_sent
	if cc_node!=None:
		if conj_subexists or conj_node==None:
			# [full_sent.append(x) for x in cc_node]
			full_sent.append(cc_node)
		elif subexists:
			newvplist=[]
			[newvplist.append(x) for x in full_sent[1]]
			newvplist.append(cc_node)
			full_sent[1]=Tree(full_sent[1].label(),newvplist)
		else:
			newvplist=[]
			[newvplist.append(x) for x in full_sent[0]]
			newvplist.append(cc_node)
			full_sent[0]=Tree(full_sent[0].label(),newvplist)
	# print(full_sent)
	if conj_node!=None:
		if conj_subexists:
			# full_sent.append(conj_node)
			[full_sent.append(x) for x in conj_node]
		elif subexists:
			#add cc and conj_node to chidlren of vp at pos 1
			newvplist=[]
			[newvplist.append(x) for x in full_sent[1]]
			[newvplist.append(x) for x in conj_node]
			full_sent[1]=Tree(full_sent[1].label(),newvplist)

			# print(conj_node)
			# if toadd:
			# 	# conj_node.pretty_print()
			# 	conj_node=Tree('VP',[conj_node])
			# [tmp.append(x) for x in conj_node]
			# full_sent[1]=Tree(full_sent[1].label(),list(full_sent[1])+[conj_node])
			# full_sent[1]=tmp
		else:
			# full_sent[0]=Tree(full_sent[0].label(),list(full_sent[0])+[conj_node])
			newvplist=[]
			[newvplist.append(x) for x in full_sent[0]]
			[newvplist.append(x) for x in conj_node]
			full_sent[0]=Tree(full_sent[0].label(),newvplist)
			# tmp=list(full_sent[0])
			# if toadd:
			# 	# conj_node.pretty_print()
			# 	conj_node=Tree('VP',[conj_node])
			# [tmp.append(x) for x in conj_node]
			# full_sent[0]=Tree(full_sent[0].label(),tmp)
	# print(full_sent)
	# if conj_node==None
	if init:
		if subexists:
			if len(full_sent)>2:

				final = Tree('S', [Tree('S',[full_sent[0],full_sent[1]])]+full_sent[2:])
			elif len(full_sent)>1:
				if len(list(full_sent[1]))==1:
					full_sent[1]=list(full_sent[1])[0]
				final=Tree('S',full_sent)
			else:
				final=Tree('S',full_sent)
		else:
			# print('lo')
			final=Tree('S',full_sent)
		# [final.pretty_print() for sent in doc.sents]
		return final,subexists
	else:
		if subexists:
			if len(full_sent)>2:
				final = Tree('S', [Tree('S',[full_sent[0],full_sent[1]])]+full_sent[2:])
			else:
				final=Tree('S',[Tree('S',full_sent)])
		else:
			final=full_sent[0]
		# [final.pretty_print() for sent in doc.sents]
		return final,subexists
# print(make_S(doc.sents[0]))
print('Resulting CP Tree:')
[make_S(sent.root)[0].pretty_print() for sent in doc.sents]
# [print(make_S(sent.root)[0]) for sent in doc.sents]

# from nltk import Tree
# from nltk.draw.util import CanvasFrame
# from nltk.draw import TreeWidget

# cf = CanvasFrame()
# # t = Tree.fromstring('(S (NP this tree) (VP (V is) (AdjP pretty)))')
# tc = TreeWidget(cf.canvas(),[make_S(sent.root)[0] for sent in doc.sents][0])
# cf.add_widget(tc,10,10) # (10,10) offsets
# cf.print_to_file('tree.ps')
# cf.destroy()
# import os
# os.system('convert tree.ps output.png')
