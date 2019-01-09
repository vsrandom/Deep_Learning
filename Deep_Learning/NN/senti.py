'''
Now we will be used our neural network on mnist dataset which was like a fixed pre package dataset , but now we want to use our Neural network on our own dataset, hence
we will be using +ve and -ve sentiment dataset
Now the first problem is we in our dataset each row is a text and even if we convert them to numbers still the length of each row is different and the neural network
model that we made needs to have same number length of each input..
'''
import nltk
from nltk.tokenize import word_tokenize # for a sentence it will create an array/list of words
import numpy as np
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer #stemming - it will remove all ing ed es etc from words

lemmatizer = WordNetLemmatizer()
hm_lines = 100000


def create_lexicon(pos,neg):
	lexicon = []
	with open(pos,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			all_words = word_tokenize(l)
			lexicon += list(all_words) ##long list of words of both positive and negative files

	with open(neg,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			all_words = word_tokenize(l)
			lexicon += list(all_words)

	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	w_counts = Counter(lexicon)
	l2 = []
	for w in w_counts:                             #w_counts={'the':5231,'and':7763..}
		#print(w_counts[w])
		if 1000 > w_counts[w] > 50: #now we don't want extremely common or rare words as it might blow up our model, we want our lexicon to be as
			l2.append(w)            #efficient as possible and lexicon will be our input and hence we wnt it to be short so we can have a decent
	print(len(l2)) #sized model of 2 or 3 layers with 1000 nodes something like that
	return l2





'''
featureset=
[
[[00100..],[1,0] or [0,1]]  where [1,0] is for positive and [0,1 is for negative file]
]
'''


def sample_handling(sample,lexicon,classification):

	featureset = []

	with open(sample,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon))
			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] += 1

			features = list(features)
			featureset.append([features,classification])

	return featureset

'''
Featureset is like this :
[10001001010111101001] [1,0]          [1,0]/[0,1] is classification !
[11001011101010110010] [0,1]
..
..
'''


def create_feature_sets_and_labels(pos,neg,test_size = 0.1):
	lexicon = create_lexicon(pos,neg)
	features = []
	features += sample_handling('pos.txt',lexicon,[1,0])
	features += sample_handling('neg.txt',lexicon,[0,1])
	random.shuffle(features) #some logic was there didn't got it
	features = np.array(features)

	testing_size = int(test_size*len(features))

	train_x = list(features[:,0][:-testing_size]) #only feature values not classification ie 90% of data
	train_y = list(features[:,1][:-testing_size]) #90% of classification value
	test_x = list(features[:,0][-testing_size:]) #last 10% of testing data
	test_y = list(features[:,1][-testing_size:]) #last 10% of classification values

	return train_x,train_y,test_x,test_y


if __name__ == '__main__':
	train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')
	# if you want to pickle this data:
	with open('sentiment_set.pickle','wb') as f:
		pickle.dump([train_x,train_y,test_x,test_y],f)
 #till now it was simple python code and hence in pickel file we stored our train_x,train_y,test_x,test_y data only
