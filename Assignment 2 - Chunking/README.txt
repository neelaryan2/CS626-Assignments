- To run the various shallow parsers, please (preferably) upload the notebooks to kaggle. 

BiLSTM
- For LSTM please change the accelarator to TPU (training would take 1-2 minutes with TPU).
- Preferable to run the code on kaggle as it already has all the packages compatible with the notebook.
- Running all the cells in the notebook will plot all the graphs like confusion matrix, classification report and tag scores.
- the conll data is already present on kaggle as well as the GloVe embeddings which can be readily attached to the notebook.
	- CONLL Corpora 	: https://www.kaggle.com/nltkdata/conll-corpora
	- GloVe Embeddings 	: https://www.kaggle.com/pkugoodspeed/nlpword2vecembeddingspretrained

CRF
- For CRF, CPU is sufficient.
- Preferable to run the code on kaggle as it already has all the packages compatible with the notebook.
- Running all the cells in the notebook will plot all the graphs like confusion matrix, classification report and tag scores.
	- CONLL Corpora		: https://www.kaggle.com/alirehan/conll-2k-chunking
	- GloVe Embeddings	: https://www.kaggle.com/rtatman/glove-global-vectors-for-word-representation

MEMM
- The complete code for SVM resides in the 'SVM_POS_tagger.ipynb' jupyter notebook.
- Running all the cells in the notebook will plot all the graphs like confusion matrix, classification report and tag scores.
- Preferable to run the code on kaggle as it already has all the packages compatible with the notebook.
- The notebook has no external dependencies except the python modules imported inside the code itself.
