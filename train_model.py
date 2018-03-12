# -*- coding: utf-8 -*-
#!/usr/bin/python
import codecs
import os
import math
import sys
from underthesea import word_sent
import pandas as pd 
import numpy as np 
from pyvi.pyvi import ViTokenizer
from sklearn import svm
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

model = svm.LinearSVC(random_state=0)
data = pd.read_csv("/home/trantri/Documents/subjects/nienluanchuyennganh_khmt/data.csv")
Temp = pd.read_csv("/home/trantri/Documents/subjects/nienluanchuyennganh_khmt/idf.csv")
attribute = []
idf = {}
for i in data:
	attribute.append(i)
attribute = attribute[:-1]
X = data[attribute]
Y = data['label']
for i in attribute:
	idf[i] = Temp[i]
train_data, test_data, label_data, label_test = train_test_split(X,Y,test_size=0.2,random_state=1)
sample_test = []
label_target = []

def Train(f):
	global model
	print("training model......")
	model.fit(train_data,label_data)
	joblib.dump(model, f)
	print("finish training!!!!")
	return

def Validation():
	global model
	global label_test
	la = []
	for lab in label_test:
		la.append(lab)
	print("predicting .....")
	pre = model.predict(test_data)
	correct = 0
	for i in range(len(pre)):
		if pre[i] == la[i]:
			correct += 1
	print("do chinh xac tren validation: {} %".format(correct/len(la) * 100))
	return

def Vectorization(document):
	tf = {}
	global temp
	vector = []
	for word in attribute:
		tf[word] = 0
	for word in document:
		if word in attribute:
			tf[word] += 1
	for word in attribute:
		tmp = tf[word] * idf[word]
		vector.append(np.asscalar(tmp.values))
	return vector

def Prepare_Test_Data(f):
	docs = os.listdir(f)
	for d in docs:
		file = f + "/" + d
		if d[0:2] == '1c':
			label_target.append(1)
		elif d[0:2] == '2t':
			label_target.append(2)
		elif d[0:2] == '3p':
			label_target.append(3)
		elif d[0:2] == '4g':
			label_target.append(4)
		elif d[0:2] == '5s':
			label_target.append(5)
		elif d[0:2] == '6k':
			label_target.append(6)
		elif d[0:2] == '7n':
			label_target.append(7)
		elif d[0:2] == '8g':
			label_target.append(8)
		elif d[0:2] == '9t':
			label_target.append(9)
		elif d[0:2] == '10':
			label_target.append(10)
		sample_test.append(Convert_Document(file))
	return

def Test():
	'''
	with open('/home/trantri/Documents/subjects/nienluanchuyennganh_khmt/test.csv','w') as file:
		for i in range(len(attribute)-1):
			file.write(attribute[i])
			file.write(',')
		file.write(attribute[len(attribute)-1])
		file.write('\n')

		for i in range(len(sample_test)-1):
			for j in range(len(sample_test[i])-1):
				file.write(str(sample_test[i][j]))
				file.write(',')
			file.write(str(sample_test[i][len(sample_test[i])-1]))
			file.write('\n')
		file.close()
	'''
	#Test =  pd.read_csv("/home/trantri/Documents/subjects/nienluanchuyennganh_khmt/test.csv")
	result_of_model = []
	count = 0
	result_of_model = model.predict(sample_test)
	for i in range(len(result_of_model)):
		if result_of_model[i] == label_target[i]:
			count += 1
	print("do chinh xac tren test: {} %".format(count/len(result_of_model) * 100))
	return

def Convert_Document(file):
	document = codecs.open(file,encoding="utf8", errors='ignore').read()
	document = ViTokenizer.tokenize(document)
	document = document.lower()
	table = str.maketrans("\"\'“”+=[]1234567890!?.,-/@#$%^&*()\\{}<>:;'", 41*" ")
	document = document.translate(table)
	document = document.split(" ")
	return Vectorization(document)
	
def Title_Of_Article(link_model,file):
	clf = joblib.load(link_model)
	Vector = Convert_Document(file)
	Tmp = clf.predict([Vector])
	if Tmp == 1:
		print("Cong Nghe Thong Tin")
	elif Tmp == 2:
		print("The Gioi")
	elif Tmp == 3:
		print("Phap Luat")
	elif Tmp == 4:
		print("Giao Duc")
	elif Tmp == 5:
		print("Suc Khoe")
	elif Tmp == 6:
		print("King Danh")
	elif Tmp == 7:
		print("Nau An")
	elif Tmp == 8:
		print("Giai Tri")
	elif Tmp == 9:
		print("Tinh Yeu")
	elif Tmp == 10:
		print("The Thao")
	return

if __name__ == '__main__':
	Train('/home/trantri/Documents/subjects/nienluanchuyennganh_khmt/MODEL1.pkl')
	Validation()
	Prepare_Test_Data("/home/trantri/Desktop/test")
	Test()
	f = "/home/trantri/Documents/subjects/nienluanchuyennganh_khmt/baibao.txt"
	link_model = "/home/trantri/Documents/subjects/nienluanchuyennganh_khmt/MODEL1.pkl"
	Title_Of_Article(link_model,f)
