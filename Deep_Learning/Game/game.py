'''
In this case Neural Net generating signal either we want to move left or right, so when you have a cluster of weak signals putting them together produce a weak signal
which is stronger and more accurate than any of signal.
'''
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import gym
import numpy as np
import random #game is  our environment and we got a agent which will move randomly to collect data
import tflearn
from tflearn.layers.core import input_data,dropout,fully_connected #dropout?!!
from tflearn.layers.estimator import regression
from collections import Counter
from statistics import mean,median #to illustrate how well random did


LR=1e-3 #learning rate
env=gym.make('CartPole-v0') #environment
env.reset()
goal_steps=500 #something related to getting score on each frame if pole is balanced
score_requirements=50 #we want to learn from all random games that have score above 50
initial_games=10000 #later!!!

def some_random_games_first():
	for episodes in range(5):
		env.reset()
		for t in range(goal_steps):
			env.render() #want to see whats happening in game
			action=env.action_space.sample()#to take any random action
			observation,reward,done,info=env.step(action) #{pole position ,cart position etc},either 0 or 1,game over,other info
			if done:
				break

#some_random_games_first() #rendering 5 games

def initial_population():
	training_data=[] #observation and move made ,move will all be randoms and we are only going to append when the score is going to be above 50!!
	scores=[]
	accepted_scores=[]

	for _ in range(initial_games): #initial_games=1000
		score=0
		game_memory=[] #until the end of game we are not going to know if we won or not,hence we are going to store all such movements in game_memory
		prev_observation=[]
		#line 47-57 correspond to one game!!
		for _ in range(goal_steps):#goal_steps=500  ,not clear about goal_steps maybe thare are two frames for each game??!!!
			action=random.randrange(0,2) #either 0 or 1 ie move left or right !!
			observation,reward,done,info=env.step(action) #reward is either 0 or 1 implies weather the pole was standing or not !

			if len(prev_observation)>0:
				game_memory.append([prev_observation,action])

			prev_observation=observation
			score+=reward #reward is either going to be 0 or 1
			if done:
				break

		if score>=score_requirements: #we have a score after completing each game!!
			accepted_scores.append(score)
			for data in game_memory: #it is a list of lists where each element is basically a pair of a list containing 4 values and a binary value ie (0/1)
				if data[1]==1:
					output=[0,1]
				elif data[1]==0:
					output=[1,0]



				training_data.append([data[0],output])


		env.reset()
		scores.append(score)


	training_data_save=np.array(training_data)
	#print(training_data[:10])
	np.save('saved.npy',training_data_save) #stores training data in a file

	print('Average accepted score:',mean(accepted_scores))
	print('Median accepted score:',median(accepted_scores))
	print(Counter(accepted_scores)) #will print like 67:6 ie out the 10000 games games which scored above 50, which were equal to 67 were 6!!!

	return training_data


#initial_population()

#Now we created our training data and now we want to train our neural network model on basis of that data

def neural_network_model(input_size):
	network=input_data(shape=[None,input_size,1],name='input')#input siz=4 coming from observation

	network=fully_connected(network,128,activation='relu') # fully connected layer with input
	network=dropout(network,0.8) #0.8 is keep rate

	network=fully_connected(network,256,activation='relu')
	network=dropout(network,0.8)

	network=fully_connected(network,512,activation='relu')
	network=dropout(network,0.8)

	network=fully_connected(network,256,activation='relu')
	network=dropout(network,0.8)

	network=fully_connected(network,128,activation='relu')
	network=dropout(network,0.8)

	network=fully_connected(network,2,activation='softmax')#output layer is of size 2 ie to move left or right
	network=regression(network,optimizer='adam',learning_rate=LR,
						loss='categorical_crossentropy',name='targets')


	model=tflearn.DNN(network,tensorboard_dir='log')

	return model #now we have defined a model ,it is not trained


def train_model(training_data,model=False):
	X=np.array([i[0] for i in training_data] ).reshape(-1,len(training_data[0][0]), 1)#[[array([ 0.04124025,  0.17975061, -0.00536662, -0.33918143]), [1, 0]], [array([ 0.04483526, -0.01529457, -0.01215024, -0.04819565]), [1, 0]],
	y=[i[1] for i in training_data]

	if not model:
		model=neural_network_model(input_size=len(X[0]))

	model.fit({'input':X},{'targets':y},n_epoch=5,snapshot_step=500,show_metric=True,run_id='openaistuff')#now we don't wan't no of epoch's to be more ,in that case we will overfit and that will be a problem for coming testing data set
	return model



training_data=initial_population()
model=train_model(training_data)
#model.save('adasss.model')
#Now the model is trained and we will use it to play a game !!!

scores=[]
choices=[]

for each_game in range(10): #we want our model to play 10 games
	score=0
	game_memory=[]
	prev_obs=[]
	env.reset()
	for _ in range(goal_steps):
		env.render()
		if len(prev_obs)==0:
			action=random.randrange(0,2) #initially in the first frame we don't know what action/move to take it will be either 0 or 1 and our network output one_hot array ie [0,1]or [1,0]
		else:
			#print(np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))))
			action=np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])#now we are predicting on one frame and there are total goal_steps no of frames in each game and len(prev_obs)=4 and argmax basically return index of 1 in [1,0]or [0,1]!!!

		choices.append(action) #sometimes our network only predict few things which is not correct ,so we want to see the ration of action our network is predicting

		new_observation,reward,done,info=env.step(action)
		prev_obs=new_observation
		game_memory.append([new_observation,action])#Rienforcement Learning????
		score+=reward
		if done:
			break
	scores.append(score)

print('Average Score',sum(scores)/len(scores))
print('Choice1:{},Choice0:{}'.format(choices.count(1)/len(choices), #how many times 0 and 1 occur
		choices.count(0)/len(choices)))
