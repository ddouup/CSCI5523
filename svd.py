import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd

import sys

def SVD(df, dimension):
	data = df.iloc[:,1:].values
	labels = df.iloc[:,0]

	'''
	svd = TruncatedSVD(n_components=dimension)
	T = svd.fit_transform(data)

	output = pd.concat([labels, pd.DataFrame(T)], axis=1, ignore_index=True)
	print(output)
	name = 'data/'+str(dimension)+'d_mnist_test.csv'
	output.to_csv(name, header=False, index=False)
	'''

	U, Sigma, VT = randomized_svd(data, n_components=dimension)
	
	output_U = pd.concat([labels, pd.DataFrame(U)], axis=1, ignore_index=True)
	print(output_U.shape)
	name = 'data/'+str(dimension)+'d_U_mnist_test.csv'
	output_U.to_csv(name, header=False, index=False)

	temp = U*Sigma
	output_USigma = pd.concat([labels, pd.DataFrame(temp)], axis=1, ignore_index=True)
	print(output_USigma.shape)
	name = 'data/'+str(dimension)+'d_U*Sigma_mnist_test.csv'
	output_USigma.to_csv(name, header=False, index=False)


def main():
	df = pd.read_csv('data/mnist_test.csv', header=None)

	mylist = [5, 10, 20, 40]
	for i in mylist:
		SVD(df, i)
	
if __name__ == '__main__':
	main()