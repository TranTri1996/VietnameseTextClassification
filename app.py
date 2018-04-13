import sys
from PyQt5.QtCore import pyqtSlot, QBasicTimer
from PyQt5.QtWidgets import QApplication, QDialog, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon, QPixmap
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
	attribute = None
	idf = None
	def Load_Idf(self):
		self.attribute = []
		self.idf = {}
		feature= []
		Temp = pd.read_csv("/home/trantri/Documents/subjects/nienluanchuyennganh_khmt/idf.csv")
		for i in Temp:
			feature.append(i)
			self.attribute.append(i)
		for i in feature:
			self.idf[i] = Temp[i]
		return

	def Convert_Document(self,contain):
		#document = codecs.open(file,encoding="utf8", errors='ignore').read()
		document = ViTokenizer.tokenize(contain)
		document = document.lower()
		table = str.maketrans("\"\'“”+=[]1234567890!?.,-/@#$%^&*()\\{}<>:;'", 41*" ")
		document = document.translate(table)
		document = document.split(" ")
		return self.Vectorization(document)

	def Vectorization(self,document):
		tf = {}
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

	def Predict(self,m,contain):
		Vector = []
		self.Load_Idf()
		clf = joblib.load(m)
		Vector = self.Convert_Document(contain)
		Tmp = clf.predict([Vector])
		result = ""
		if Tmp == 1:
			result = "Công Nghệ Thông Tin"
		elif Tmp == 2:
			result = "Thế Giới"
		elif Tmp == 3:
			result = "Pháp Luật"
		elif Tmp == 4:
			result = "Giáo Dục"
		elif Tmp == 5:
			result = "Sức Khỏe"
		elif Tmp == 6:
			result = "Kinh Doanh"
		elif Tmp == 7:
			result = "Nội Trợ"
		elif Tmp == 8:
			result = "Giải Trí"
		elif Tmp == 9:
			result = "Tình Yêu"
		elif Tmp == 10:
			result = "Thể Thao"
		return result
class DemoImpl(QDialog):

	def __init__(self, *args):
		super(DemoImpl, self).__init__(*args)
		loadUi('interface.ui', self)
		self.btn_predict.clicked.connect(self.btn_predict_clicked)
		self.btn_browser.clicked.connect(self.btn_browser_clicked)
		self.timer = QBasicTimer()
		self.step = 0
		self.progressBar.setValue(self.step)
		pixmap = QPixmap('logo.jpeg')
		self.image.setPixmap(pixmap)

	def timerEvent(self, e):     
		if self.step >= 100:      
			self.timer.stop()
			return
            
		self.step = self.step + 25
		self.progressBar.setValue(self.step)
	def btn_browser_clicked(self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		files, _ = QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()", "","All Files (*);;Text Files (*.txt)", options=options)
		f = files[0]
		self.lineEdit.setText(f)
		contain = codecs.open(f,encoding="utf8", errors='ignore').read()
		self.plain_text.clear()
		self.title.clear()
		self.step = 0
		self.progressBar.setValue(self.step)
		self.plain_text.appendPlainText(contain)
		
	def btn_predict_clicked(self):
		self.timer.start(100, self)
		contain = self.plain_text.toPlainText()
		classifier = text_classification()
		label = classifier.Predict("/home/trantri/Documents/subjects/nienluanchuyennganh_khmt/MODEL.pkl",contain)
		self.title.setText(label)
if __name__ == '__main__':
	app = QApplication(sys.argv)
	widget = DemoImpl()
	widget.show()
	sys.exit(app.exec_())