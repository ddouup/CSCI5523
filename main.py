import pandas as pd
import numpy as np
import sys
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from scipy.spatial.distance import pdist, squareform

def Euclidean_Distance(a,b):
	return np.linalg.norm(a-b)


def Cosine_Similarity(a,b):
	return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


def Jaccard_Similarity(a,b):
	return np.dot(a,b)/(np.linalg.norm(a)**2+np.linalg.norm(b)**2-np.dot(a, b))


def findCloset(array):
	for i in range(array.shape[0]):
		array[i][i] = 1

	print(array)
	return np.argmin(array, axis=1)



def countSameDigit(array, data):
	count = 0

	for i in range(array.size):
		if data.iloc[i,0] == data.iloc[array[i],0]:
			#print()
			#print('Image',i,'digit:', data.iloc[i,0])
			#print('Image',array[i],'digit:', data.iloc[array[i],0])
			#print('Bingo!')
			count+=1

	return count



def main():
	df = pd.read_csv('data/mnist_test.csv', header=None)

	size = df.index.size
	print('Total data: ',size)

	data = df.iloc[:,1:].values
	
	EuclideanDistance = np.ones((size, size))
	CosineDistance = np.ones((size, size))
	JaccardDistance = np.ones((size, size))

	'''
	for i in range(size):
		for j in range(size):
			print('row: ',i,', columm: ',j)

			if i < j:
				EuclideanDistance[i][j] = Euclidean_Distance(data[i][1::], data[j][1::])/10000
				CosineDistance[i][j] = 1-Cosine_Similarity(data[i][1::], data[j][1::])
				JaccardDistance[i][j] = 1-Jaccard_Similarity(data[i][1::], data[j][1::])
			else:
				EuclideanDistance[i][j] = EuclideanDistance[j][i]
				CosineDistance[i][j] = CosineDistance[j][i]
				JaccardDistance[i][j] = JaccardDistance[j][i]
			
			print('Euclidean Distance: ', EuclideanDistance[i][j])
			print('Cosine Distance: ', CosineDistance[i][j])
			print('Jaccard Distance: ', JaccardDistance[i][j])
	'''

	#EuclideanDistance = euclidean_distances(data, data)/10000
	#CosineDistance = 1-cosine_similarity(data, data)

	EuclideanDistance = squareform(pdist(data, 'euclidean'))/10000
	CosineDistance = squareform(pdist(data, 'cosine'))
	JaccardDistance = squareform(pdist(data, 'jaccard'))


	print('Euclidean: ')
	print(EuclideanDistance)

	print('Cosine: ')
	print(CosineDistance)

	print('Jaccard: ')
	print(JaccardDistance)

	'''
	output = pd.DataFrame(EuclideanDistance)
	output.to_csv("a_Euclidean_Distance.csv", header=False, index=False)
	
	output = pd.DataFrame(CosineDistance)
	output.to_csv("a_Cosine_Distance.csv", header=False, index=False)

	output = pd.DataFrame(JaccardDistance)
	output.to_csv("a_Jaccard_Distance.csv", header=False, index=False)

	print('Distance saved.')
	'''

	EuclideanDistance_closet = findCloset(EuclideanDistance)
	CosineSimilarity_closet = findCloset(CosineDistance)
	JaccardSimilarity_closet = findCloset(JaccardDistance)

	print(EuclideanDistance_closet)
	print(CosineSimilarity_closet)
	print(JaccardSimilarity_closet)

	print('Result using Euclidean Distance:', countSameDigit(EuclideanDistance_closet, df))
	print('Result using Cosine Similarity:', countSameDigit(CosineSimilarity_closet, df))
	print('Result using Jaccard Similarity:', countSameDigit(JaccardSimilarity_closet, df))



if __name__ == '__main__':
	main()