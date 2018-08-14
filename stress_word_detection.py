import numpy as np
import pandas as pd
import librosa
import os
import pickle
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import random
np.random.seed(100)


MAX_RMSE_LENGTH = 200
DATA_PATH = '../data/'
OUTPUT_PATH = 'stressed_files.csv'

def main():
	X = np.array(np.zeros(MAX_RMSE_LENGTH))
	Y = []
	words = []
	directory_files = os.listdir(DATA_PATH)

	for filename in directory_files:
		y, sr = librosa.load(DATA_PATH + filename)

		rmse = librosa.feature.rmse(y)[0]
		length_difference = MAX_RMSE_LENGTH - rmse.shape[0]
		if length_difference > 0:
			rmse = np.append(rmse, np.zeros(length_difference))
		else:
			rmse = rmse[:MAX_RMSE_LENGTH]
		X = np.vstack((X, rmse))
		Y = np.append(Y, 0)
		words.append("")
		for w in re.findall(r"([a-zA-Z]+)_?", filename.split("-")[0]):
			if w.isupper():
				Y[-1] = 1
				words[-1] = w
		# print ("filename: {}\twords")

	X = X[1:]
	Y = np.array(Y)

	clf = RandomForestClassifier(n_estimators = 5, max_depth=7)
	clf.fit(X,Y)
	predictions = clf.predict(X)
	correct_count = 0
	for j in range(len(predictions)):
		if predictions[j] == Y[j]:
			correct_count += 1
		else:
			print "Misclassified the wav file: "+directory_files[j] +"\n"
			

	accuracy = float(correct_count)/float(len(predictions)) * 100
	print "Model Accuracy for detecting strees exits or not: "+ str(accuracy) 

	file_word = ""
	with open(OUTPUT_PATH,'w') as f:
		for index, postive in enumerate(predictions):
			if postive == 1 and words[index] != "":
				file_word += DATA_PATH + directory_files[index] + ',' + words[index] + '\n'
		f.write(file_word)

	pickle.dump(clf, open("model.pkl", "wb"))

if __name__ == '__main__':
	main()
