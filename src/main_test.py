from main import run_experiment
from main import just_train_and_test
import dataSplit as dataSplit
import src.kernels_c as kernels_c
from wk_ngk_Kernels import wk,ngk

traindata = [('good good good good happy happy happy happy positive positive positive positive', 'right'),
				('right right right right happy happy happy happy positive positive positive positive', 'right'), 
				('wrong wrong wrong wrong sad sad sad sad negative negative negative negative', 'wrong'), 
				('good good good good happy happy happy happy content content content content', 'right'),
				('bad bad bad bad sad sad sad sad negative negative negative negative', 'wrong')]

testdata = [('good good good happy happy positive positive', 'right'),
			('bad bad bad bad bad bad', 'wrong'),
			('wrong bad bad sad sad negative', 'wrong'),
			]

train1 = [('good good awesome awesome', 'right'),
			('good awesome', 'right'),
			('bad answer not bad', 'wrong'),
			('bad test not bad', 'wrong')]

test1 = [('good awesome', 'right'),
		('bad bad', 'wrong')]

topic = 'right'

k=5
lambdaDecay=0.5

traindata = dataSplit.load_data('../data/datasets/train_short_new')
testdata = dataSplit.load_data('../data/datasets/test_short_new')

trainGram = dataSplit.load_data('../data/kernels/final/wk_gram_train'+'_k'+str(k)+'_l'+str(lambdaDecay*10))
testGram = dataSplit.load_data('../data/kernels/final/wk_gram_test'+'_k'+str(k)+'_l'+str(lambdaDecay*10))
#sampledata = dataSplit.load_data('../data/datasets/test_data_small')
#print(sampledata[1])

#VARIABLES for SSK

topic='corn'


#print(trainGram)
#print(testGram)

val=[]
i=0
result = just_train_and_test(trainGram, testGram.T, traindata, testdata, topic)
#print(train1)
#for topic in ['corn', 'earn', 'acq', 'crude']:
	#for k in [3, 7, 11]:
	#	for lambdaDecay in [0.01, 0.1, 0.3, 0.5, 0.9]:
#result = run_experiment(ngk, traindata, testdata, topic, 5, 0.5)

print(result)

