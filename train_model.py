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
model = svm.SVC()

data = pd.read_csv("/home/trantri/Documents/subjects/nienluanchuyennganh_khmt/data.csv")

temp = pd.read_csv("/home/trantri/Documents/subjects/nienluanchuyennganh_khmt/idf.csv")
attribute = []
idf = {}
for i in data:
	attribute.append(i)
attribute = attribute[:-1]
X = data[attribute]
Y = data['label']

print(len(attribute))

for i in temp:
	print(i)
	
train_data, test_data, label_data, label_test = train_test_split(X,Y,test_size=0.2,random_state=1)

sample_test = []
label_target = []
def train():
	global model
	print("training model......")
	model.fit(train_data,label_data)
	print("finish training!!!!")
	return

def validation():
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

def prepare_test_data():
	result_of_model = []
	count = 0
	docs = os.listdir("/home/trantri/Desktop/test")
	for d in docs:
		file = "/home/trantri/Desktop/test/" + d
		print(file)
		if d[0:2] == '1c':
			label_target.append(1)
		elif d[0:2] == '2t':
			label_target.append(2)
		elif d[0:2] == '3p':
			label_target.append(3)
		elif d[0:2] == '4g':
			label_target.append(4)
		elif d[0:2] == '5s':
			labels_test.append(5)
		elif d[0:2] == '6k':
			label_target.append(6)
		elif d[0:2] == '7n':
			label_target.append(7)
		elif d[0:2] == '8g':
			label_target.append(8)
		elif d[0:2] == '9t':
			labels_test.append(9)
		elif d[0:2] == '10':
			label_target.append(10)
		result_of_model.append(test(file))
	for i in range(len(result_of_model)):
		if result_of_model[i] == label_target[i]:
			count += 1
	print("do chinh xac tren validation: {} %".format(count/len(result_of_model) * 100))

def vectorization(document):
	tf = {}
	global temp

	vector = []


	for word in attribute:
		tf[word] = 0

	for word in document:
		if word in attribute:
			tf[word] += 1

	for word in attribute:
		vector.append(tf[word] * idf[word])
	return vector
	
def test(file):
	document = codecs.open(file,encoding="utf8", errors='ignore').read()
	document = ViTokenizer.tokenize(document)
	document = document.lower()
	table = str.maketrans("\"\'“”+=[]1234567890!?.,-/@#$%^&*()\\{}<>:;'", 41*" ")
	document = document.translate(table)
	document = document.split(" ")
	instance = vectorization(document)
	result = model.predict(instance)
	return result

