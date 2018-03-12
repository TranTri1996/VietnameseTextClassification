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

dictionary = {}
documents_string = []
documents_string_test = []
documents_number = []
features = []
nW = 0 # so luong cac tu khac nhau
time_appear = {}
count_docs = {}

idf = {}

samples = []
labels = []


def Convert_String_To_Number(document):
	doc_number = ""
	for i in range(len(document)-1):
		doc_number += str(dictionary[document[i]]) + " "
	doc_number += str(dictionary[document[len(document)-1]])
	return doc_number

def IDF(f):
	global count_docs
	global idf
	for word in features:
		count_docs[word] = 0

	for word in features:
		for doc in documents_string:
			if word in doc:
				count_docs[word] += 1
	for word in features:
		idf[word] = math.log(nW/count_docs[word])
	with open(f,'w') as file:
		for i in range(len(features)-1):
			file.write(features[i])
			file.write(',')
		file.write(features[len(features)-1])
		file.write('\n')
		for i in range(len(features)-1):
			file.write(str(idf[features[i]]))
			file.write(',')
		file.write(str(idf[features[len(features)-1]]))
		file.close()
	return

def TF(document):
	tf = {}
	for word in features:
		tf[word] = 0
	for word in document:
		if word in features:
			tf[word] += 1
	return tf

def TF_IDF():
	global idf
	global samples
	for doc in documents_string:
		tfidf = []
		tf = TF(doc)
		i = 0
		for word in features:
			tfidf.append(tf[word] * idf[word])
			i += 1
		samples.append(tfidf)
	return


#tien xu li
def Pretreatment(document): 
	document = ViTokenizer.tokenize(document)
	document = document.lower()
	table = str.maketrans("\"\'“”+=[]1234567890!?.,-/@#$%^&*()\\{}<>:;'", 41*" ")
	document = document.translate(table)
	document = document.split(" ")
	Add_Dictionary(document)
	documents_string.append(document)
	#documents_number.append(convert_string_to_number(document))
	return

def Add_Dictionary(document):
	global dictionary
	global nW
	global time_appear
	for word in document:
		if word not in dictionary:
			dictionary[word] = nW
			time_appear[word] = 1
			nW += 1
		else:
			time_appear[word] += 1
	return	
def Remove_Stopword():
	for word in dictionary:
		if time_appear[word] >=60 and time_appear[word] <=600:
			features.append(word)
	return

def Access_Data(f):
	docs = os.listdir(f)
	for d in docs:
		file = f + d
		print(file)
		if d[0:2] == '1c':
			labels.append(1)
		elif d[0:2] == '2t':
			labels.append(2)
		elif d[0:2] == '3p':
			labels.append(3)
		elif d[0:2] == '4g':
			labels.append(4)
		elif d[0:2] == '5s':
			labels.append(5)
		elif d[0:2] == '6k':
			labels.append(6)
		elif d[0:2] == '7n':
			labels.append(7)
		elif d[0:2] == '8g':
			labels.append(8)
		elif d[0:2] == '9t':
			labels.append(9)
		elif d[0:2] == '10':
			labels.append(10)
		content = codecs.open(file,encoding="utf8", errors='ignore').read()
		#tien xu li tung noi dung bai viet
		Pretreatment(content)
	return

#chia thu muc data thanh 2 thu muc la data va test
def Repare(f,data_link,test_link):
	dirs = os.listdir(f)
	for dir in dirs:
		forder = f + "/" + dir
		docs = os.listdir(forder)
		i = 0
		for doc in docs:
			file = forder + "/" + doc
			content = codecs.open(file,encoding="utf8", errors='ignore').read()
			filename = ""
			if i < 550:
				filename = data_link + dir + str(i) + ".txt"
			else:
				filename = test_link + dir + str(i) + ".txt"
			print(filename)
			file = open(filename,"w")
			file.write(content)
			i += 1
			file.close()
	return
def Make_File_Data(f):
	with open(f,'w') as file:
		for i in range(len(features)):
			file.write(features[i])
			file.write(',')
		file.write('label')

		file.write('\n')
		for i in range(len(samples)):
			for j in range(len(samples[i])):
				file.write(str(samples[i][j]))
				file.write(',')
			file.write(str(labels[i]))
			file.write('\n')
	file.close()
	return
if __name__ == '__main__':
	Repare("/home/trantri/Desktop/data_tuong_lam","/home/trantri/Desktop/data/","/home/trantri/Desktop/test/")
	Access_Data("/home/trantri/Desktop/data/")
	Remove_Stopword()
	print("size of features = {}".format(len(features)))
	print("number of different words = {}".format(nW))
	IDF('/home/trantri/Documents/subjects/nienluanchuyennganh_khmt/idf.csv')
	TF_IDF()
	Make_File_Data('/home/trantri/Documents/subjects/nienluanchuyennganh_khmt/data.csv')