import librosa
import os
import numpy as np
import pickle

DATAPATH = "../data/"

OUTPUTPATH = "../output/"

# List of directories
_, directories, _ = next(os.walk(DATAPATH))

for dir_ in directories:
	directory_path = DATAPATH + dir_ + "/"
	_,_,list_of_files = next(os.walk(directory_path))
	outputFile = OUTPUTPATH + dir_ +".pkl"

	list_mfcc = []

	for file in list_of_files:
		x, sr = librosa.load(directory_path+file)
		mfcc = librosa.feature.mfcc(x,sr)
		#print ("mfcc", mfcc)
		#mfcc_array = np.pad(mfcc,((0,0),(0,(len(mfcc[0]) - 68 ))), mode='constant', constant_values=0)
		mfcc_array = np.pad(mfcc,((0,0),(0,300 - len(mfcc[0]))), mode='constant', constant_values=0)
		print(mfcc_array.shape)
		# mfcc_array = mfcc_array.flatten()
		# mfcc_array = mfcc_array.tolist()
		# if list_mfcc.shape[0] == 0:
		# 	list_mfcc = mfcc_array
		# else:
		# 	list_mfcc = np.vstack((list_mfcc, mfcc_array))
		list_mfcc.append(mfcc_array.tolist())

	print ("\n\n\n")
	list_mfcc = np.array(list_mfcc)
	#print (list_mfcc.shape, dir_)
	pickle.dump({dir_:list_mfcc}, open(outputFile, "wb"))
