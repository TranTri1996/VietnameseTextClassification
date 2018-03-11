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

sample_test = []
labels_test=[]

def convert_string_to_number(document):
	doc_number = ""
	for i in range(len(document)-1):
		doc_number += str(dictionary[document[i]]) + " "
	doc_number += str(dictionary[document[len(document)-1]])
	return doc_number

def IDF():
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
	with open('/home/trantri/Documents/subjects/nienluanchuyennganh_khmt/idf.csv','w') as file:
		for i in range(len(features)-1):
			file.write(features[i])
			file.write(',')
		file.write(features[len(features)-1])
		
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
def pretreatment(document): 
	document = ViTokenizer.tokenize(document)
	#dinh dang lai font chu thuong
	document = document.lower()
	#loai cac ki tu dac biet
	table = str.maketrans("\"\'“”+=[]1234567890!?.,-/@#$%^&*()\\{}<>:;'", 41*" ")
	document = document.translate(table)
	document = document.split(" ")
	add_dictionary(document)
	documents_string.append(document)
	#documents_number.append(convert_string_to_number(document))
	return

def add_dictionary(document):
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
def remove_stopword():
	for word in dictionary:
		if time_appear[word] >=60 and time_appear[word] <=600:
			features.append(word)
	return

def access_data():
	docs = os.listdir("/home/trantri/Desktop/data")
	for d in docs:
		file = "/home/trantri/Desktop/data/" + d
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
		pretreatment(content)
	return

#chia thu muc data thanh 2 thu muc la data va test
def repear():
	dirs = os.listdir("/home/trantri/Desktop/data_tuong_lam")
	for dir in dirs:
		forder = "/home/trantri/Desktop/data_tuong_lam" + "/" + dir
		docs = os.listdir(forder)
		i = 0
		for doc in docs:
			file = forder + "/" + doc
			content = codecs.open(file,encoding="utf8", errors='ignore').read()
			filename = ""
			if i < 550:
				filename = "/home/trantri/Desktop/data/" + dir + str(i) + ".txt"
			else:
				filename = "/home/trantri/Desktop/test/" + dir + str(i) + ".txt"
			print(filename)
			file = open(filename,"w")
			file.write(content)
			i += 1
			file.close()
	return
def make_file_data():
	with open('/home/trantri/Documents/subjects/nienluanchuyennganh_khmt/data.csv','w') as file:
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
	repear()
	access_data()
	remove_stopword()
	print("size of features = {}".format(len(features)))
	print("number of different words = {}".format(nW))
	IDF()
	TF_IDF()
	make_file_data()