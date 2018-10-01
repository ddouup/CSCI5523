import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.measure

def main():
	df = pd.read_csv('data/mnist_test.csv', header=None)

	data = df.iloc[:,1:].values
	labels = df.iloc[:,0]

	output_data = np.zeros((10000, 49))
	for i in range(data.shape[0]):
		print('Processing No.',i,'image...')
		result = skimage.measure.block_reduce(data[i].reshape(28,28), (4,4), np.mean)
		output_data[i] = result.reshape(1,49)

	output = pd.concat([labels, pd.DataFrame(output_data)], axis=1, ignore_index=True)
	output.to_csv('data/7*7_mnist_test.csv', header=False, index=False)
	
if __name__ == '__main__':
	main()