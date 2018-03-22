# -*- coding: utf-8 -*-
#!/usr/bin/python
import codecs
import os
import math
import sys
import pandas as pd 
import numpy as np 
from pyvi.pyvi import ViTokenizer
from sklearn import svm
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

class text_classification:
	model = None
	train_data = None
	test_data = None
	label_data = None
	label_test = None
	attribute = []
	idf = {}
	def Load_Idf(self):
		feature= []
		Temp = pd.read_csv("/home/trantri/Documents/subjects/nienluanchuyennganh_khmt/idf.csv")
		for i in Temp:
			feature.append(i)
		for i in feature:
			self.idf[i] = Temp[i]
		return

	def Train_Data(self):
		global train_data,label_data,test_data,label_test,model
		feature = []
		Temp = pd.read_csv("/home/trantri/Documents/subjects/nienluanchuyennganh_khmt/idf.csv")
		for i in Temp:
			self.attribute.append(i)
		data = pd.read_csv("/home/trantri/Documents/subjects/nienluanchuyennganh_khmt/data.csv")
		for i in data:
			feature.append(i)
		feature = feature[:-1]
		X = data[feature]
		Y = data['label']
		train_data, test_data, label_data, label_test = train_test_split(X,Y,test_size=0.2,random_state=1)
		print("training model......")
		model = svm.LinearSVC(random_state=0)
		model.fit(train_data,label_data)
		joblib.dump(model,"/home/trantri/Documents/subjects/nienluanchuyennganh_khmt/MODEL2.pkl")
		print("finish training!!!!")
		return

	def Validation(self):
		global model
		global label_test
		global test_data
		label = []
		for i in label_test:
			label.append(i)
		print("predicting .....")
		validate = model.predict(test_data)
		count = 0
		for i in range(len(validate)):
			if validate[i] == label[i]:
				count += 1
		print("do chinh xac tren validation: {} %".format(count/len(label) * 100))
		return

	def Convert_Document(self,file):
		document = codecs.open(file,encoding="utf8", errors='ignore').read()
		document = ViTokenizer.tokenize(document)
		document = document.lower()
		table = str.maketrans("\"\'“”+=[]1234567890!?.,-/@#$%^&*()\\{}<>:;'", 41*" ")
		document = document.translate(table)
		document = document.split(" ")
		return self.Vectorization(document)

	def Vectorization(self,document):
		tf = {}
		global Temp
		vector = []
		for word in self.attribute:
			tf[word] = 0
		for word in document:
			if word in self.attribute:
				tf[word] += 1
		for word in self.attribute:
			tmp = tf[word] * self.idf[word]
			vector.append(np.asscalar(tmp.values))
		return vector

	def Predict(self,m,file):
		self.Load_Idf()
		clf = joblib.load(m)
		Vector = self.Convert_Document(file)
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

	def Test(self,f):
		sample_test = []
		label_target = []
		self.Load_Idf()
		global model
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
			sample_test.append(self.Convert_Document(file))
		print("testing model....")
		result = []
		count = 0
		result = model.predict(sample_test)
		for i in range(len(result)):
			if result[i] == label_target[i]:
				count += 1
		print("do chinh xac tren test: {} %".format(count/len(result) * 100))
		return


if __name__ == '__main__':
	classifier = text_classification()
	classifier.Train_Data()
	classifier.Validation()
	classifier.Test("/home/trantri/Documents/subjects/nienluanchuyennganh_khmt/test")
	classifier.Predict("/home/trantri/Documents/subjects/nienluanchuyennganh_khmt/MODEL.pkl","/home/trantri/Documents/subjects/nienluanchuyennganh_khmt/baibao.txt")