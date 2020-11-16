Constituency to Dependency
- This notebook requires the following modules
	- spacy
	- nltk
	- benepar (cython and other dependencies)
	- tensorflow
- benepar does not work with tensorflow 2.0 out-of-the-box and requires some additional steps
	- currently code has already been included inside the notebook to solve the issue
	- if the automated fix does not work, then change the following inside <benepar_module_location>/base_parser.py
		- 'import tensorflow as tf'	-> 	'import tensorflow.compat.v1 as tf'
										'tf.disable_v2_behavior()'
	- now benepar should ideally work
- Preferable to run the code on kaggle as the notebook has already been tested on that platform
- The slowest part of the notebook will be processing all the imports and fetching the parses for the first time (overall takes 1 minute)
- Conversion from CP to DP takes almost no time at all
- output.txt contains the constituency and dependency parses of some sample sentences along with headified cp trees

Dependency to Constituency
- spacy==2.3.2
- nltk==3.5
