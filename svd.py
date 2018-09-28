import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD



def SVD(df, dimension):
	data = df.iloc[:,1:].values
	labels = df.iloc[:,0]

	svd = TruncatedSVD(n_components=dimension)
	T = svd.fit_transform(data)
	output = pd.DataFrame()
	name = dimension+'d_mnist_test'
	output.to_csv(name, header=False, index=False)

def main():
	df = pd.read_csv('data/test.csv', header=None)

	for i in [5, 10, 20, 40]
		SVD(df, i)
	
if __name__ == '__main__':
	main()