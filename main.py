import pandas as pd
import numpy as np
import os, time
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from scipy.spatial.distance import pdist, squareform

def Euclidean_Distance(a,b):
	return np.linalg.norm(a-b)


def Cosine_Similarity(a,b):
	return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


def Jaccard_Similarity(a,b):
	return 1-np.dot(a,b)/(np.linalg.norm(a)**2+np.linalg.norm(b)**2-np.dot(a, b))


def countSameDigit(distance, df):
	for i in range(distance.shape[0]):
		distance[i][i] = 1

	array = np.argmin(distance, axis=1)

	#print(array)

	count = 0

	for i in range(array.size):
		print('Image'+str(i)+"'s digit:"+str(df.iloc[i,0]))
		print('Closet Image: '+str(array[i])+' Digit:'+str(df.iloc[array[i],0]))
		print()
		if df.iloc[i,0] == df.iloc[array[i],0]:
			print('Bingo!!!!!!!!!!!!!!!!!')
			print()
			count+=1
		else:
			print('Woopsssssssss')
			print()

	return count



def process(filename):
	start = time.time()

	output_file = 'result/'+os.path.splitext(os.path.basename(filename))[0]+'.txt'
	
	print("Processing data: ",filename, "...")
	print('Store to: ',output_file)

	text_file = open(output_file, "w")
	text_file.write('Processing data: '+filename+'...\n')

	df = pd.read_csv(filename, header=None)

	shape = df.shape
	text_file.write('Data Size: '+str(shape)+'\n\n')
	print('Data Size: ',shape)

	data = df.iloc[:,1:].values
	
	EuclideanDistance = np.ones((shape[0], shape[0]))
	CosineDistance = np.ones((shape[0], shape[0]))
	JaccardDistance = np.ones((shape[0], shape[0]))

	#EuclideanDistance = euclidean_distances(data, data)/10000
	#CosineDistance = 1-cosine_similarity(data, data)

	EuclideanDistance = squareform(pdist(data, 'euclidean'))/10000
	CosineDistance = squareform(pdist(data, 'cosine'))
	JaccardDistance = squareform(pdist(data, Jaccard_Similarity))

	'''
	print('Euclidean: ')
	print(EuclideanDistance)

	print('Cosine: ')
	print(CosineDistance)

	print('Jaccard: ')
	print(JaccardDistance)
	'''
	result_euclidean = countSameDigit(EuclideanDistance, df)
	result_cosine = countSameDigit(CosineDistance, df)
	result_jaccard = countSameDigit(JaccardDistance, df)

	text_file.write('Result using Euclidean Distance:'+str(result_euclidean)+'\n')
	text_file.write('Result using Cosine Similarity:'+str(result_cosine)+'\n')
	text_file.write('Result using Jaccard Similarity:'+str(result_jaccard)+'\n')

	end = time.time()
	text_file.write('\nTime consumed:'+str(end-start))
	text_file.close()


def main():
	process("data/test.csv")
	
	process("data/mnist_test.csv")

	mylist = [5, 10, 20, 40]
	for i in mylist:
		filename = 'data/'+str(i)+'d_U_mnist_test.csv'
		process(filename)
		filename = 'data/'+str(i)+'d_U*Sigma_mnist_test.csv'
		process(filename)

	process('data/7*7_mnist_test.csv')
	
	print('All finished')

if __name__ == '__main__':
	main()