import numpy as np
import math

#from lib_robotis_hack import *

#class to establish a feature space
class manage_state:
	#make an i x j matrix thing
	def __init__(self, i, j = 1):
		self.state = np.zeros((i, j))
		
	def update_state(self, i, j = 0):
		#make a new state and flatten to use in learning
		self.state *= 0
		self.state[i, j] = 1
		self.state_vector = self.state.flatten()


class GTDGVF:
	#initialize learning parameters
	def __init__(self, gam, lam, alpha_W, alpha_omega, size):
		self.gam = gam
		self.lam = lam
		self.alpha_W = alpha_W
		self.alpha_omega = alpha_omega*self.alpha_W
		
		self.delta = 0.0
		
		self.W = np.zeros(size)
		self.omega = np.zeros(size)
		self.e = np.zeros(size)
		self.S_prime = np.zeros(size)
		
		#RUPEE stuffs
		self.alpha_h_rupee = 10*self.alpha_W
		self.rupee_beta_0 = (1-self.lam)*(self.alpha_W/30)
		self.rupee_tau = 0
		self.rupee_squiggle = np.zeros(size)
		self.h_rupee = np.zeros(size)
		
		#UDE stuffs
		self.alpha_h_ude = self.alpha_W/100
		self.ude_beta_0 = 10*self.alpha_W
		self.ude_tau = 0
		self.ude_squiggle = 0
		self.h_ude = np.zeros(size)
		
		self.predict = 0

	#update learning. allow for change in gamma is desired
	def update(self, cumulant, S, gam=None, rho=1):
		if gam is None:
			gam = self.gam
		
		self.predict = np.dot(self.W, self.S_prime)
		# self.predict = np.dot(self.W, S)
	
		self.delta = cumulant + gam*np.dot(self.W, S) - np.dot(self.W, self.S_prime)
		self.e = rho*(self.S_prime + gam*self.lam*self.e)
		self.W = self.W + self.alpha_W*(self.delta*self.e - gam*(1 - self.lam)*np.dot(self.e, self.omega)*S)
		self.omega = self.omega + self.alpha_omega*(self.delta*self.e - np.dot(self.S_prime, self.omega)*self.S_prime)
	
		self.S_prime = S

	def RUPEE(self):
		self.h_rupee = self.h_rupee + self.alpha_h_rupee*(self.delta*self.e - np.dot(self.h_rupee, self.S_prime)*self.S_prime)
		self.rupee_tau = (1 - self.rupee_beta_0)*self.rupee_tau + self.rupee_beta_0
		rupee_beta = self.rupee_beta_0/self.rupee_tau
		self.rupee_squiggle = (1 - rupee_beta)*self.rupee_squiggle + rupee_beta*self.delta*self.e
		self.rupee = np.sqrt(abs(np.dot(self.h_rupee, self.rupee_squiggle)))
	
	def UDE(self, var):
		self.h_ude = self.h_ude + self.alpha_h_ude*(self.delta*self.e - np.dot(self.h_ude, self.S_prime)*self.S_prime)
		self.ude_tau = (1 - self.ude_beta_0)*self.ude_tau + self.ude_beta_0
		ude_beta = self.ude_beta_0/self.ude_tau
		self.ude_squiggle = (1 - ude_beta)*self.ude_squiggle + ude_beta*self.delta
		self.ude = abs(self.ude_squiggle/(np.sqrt(var) + 1.0/100000.0))
	

class actor_critic:
	def __init__(self, gam = 0.9, lam = 0.4, alpha_w = 0.1, alpha_m = 0.1, alpha_s = 0.05, size = 10):
		self.gam = gam
		self.lam = lam
		self.alpha_W = alpha_w
		self.alpha_m = alpha_m
		self.alpha_s = alpha_s
		
		self.delta = 0.0
		self.predict = 0.0
		
		self.S_prime = np.zeros(size)
		self.W = np.ones(size)*0.5
		self.W_m = np.zeros(size)
		self.W_s = np.ones(size)
		self.e = np.zeros(size)
		self.e_m = np.zeros(size)
		self.e_s = np.zeros(size)
	
	def update(self, reward, a, m, sig, S, gam=None):
		if gam is None:
			gam = self.gam
		
		self.predict = np.dot(self.W, self.S_prime)
	
		self.delta = reward + gam*np.dot(self.W, S) - np.dot(self.W, self.S_prime)
		self.e = (self.S_prime + gam*self.lam*self.e)
		self.W = self.W + self.alpha_W*self.delta*self.e
		
		self.e_m = self.lam*self.gam*self.e_m + (a - m)*self.S_prime
		self.W_m = self.W_m + self.alpha_m*self.delta*self.e_m
		
		self.e_s = self.lam*self.gam*self.e_s + ((a - m)**2 - sig**2)*self.S_prime
		self.W_s = self.W_s + self.alpha_s*self.delta*self.e_s
	
		self.S_prime = S

class disc_actor_critic:
	def __init__(self, gam = 0.9, lam = 0.4, alpha_w = 0.1, alpha_a = 0.1, alpha_b = 0.1, alpha_c = 0.1, size = 10, numactions = 3):
		self.gam = gam
		self.lam = lam
		self.alpha_W = alpha_w
		self.alpha_a = alpha_a
		self.alpha_b = alpha_b
		self.alpha_c = alpha_c
		self.numactions = numactions
		
		self.delta = 0.0
		self.predict = 0.0
		
		self.S_prime = np.zeros(size)
		self.W = np.zeros(size)
		self.W_a = np.zeros(size)
		self.W_b = np.zeros(size)
		self.W_c = np.zeros(size)
		self.e = np.zeros(size)
		self.e_a = np.zeros(size)
		self.e_b = np.zeros(size)
		self.e_c = np.zeros(size)
		
		self.probs = np.array([0.3, 0.3, 0.4])
		self.act = 2
	
	def update(self, reward, S, gam=None):
		if gam is None:
			gam = self.gam
		
		self.predict = np.dot(self.W, self.S_prime)
	
		self.delta = reward + gam*np.dot(self.W, S) - np.dot(self.W, self.S_prime)
		self.e = (self.S_prime + gam*self.lam*self.e)
		self.W = self.W + self.alpha_W*self.delta*self.e
		
		action = np.zeros(self.numactions)
		action[self.act] = 1
		
		self.e_a = self.lam*self.gam*self.e_a + (action[0] - self.probs[0])*self.S_prime
		self.W_a = self.W_a + self.alpha_a*self.delta*self.e_a
		
		self.e_b = self.lam*self.gam*self.e_b + (action[1] - self.probs[1])*self.S_prime
		self.W_b = self.W_b + self.alpha_b*self.delta*self.e_b
		
		self.e_c = self.lam*self.gam*self.e_c + (action[2] - self.probs[2])*self.S_prime
		self.W_c = self.W_c + self.alpha_c*self.delta*self.e_c
	
		self.S_prime = S
		
	def softmax(self):
		values = [np.dot(self.W_a, self.S_prime), np.dot(self.W_b, self.S_prime), np.dot(self.W_c, self.S_prime)]
		self.probs = np.exp(values)/sum(np.exp(values))
		self.act = np.random.choice(self.numactions, 1, p=self.probs)[0]

class IncVariance:
	def __init__(self):
		self.sample_mean = 0
		self.n = 0
		self.ewmv = 0

	def update(self, x):
		old_mean = self.sample_mean

		self.n += 1
		self.sample_mean += (x - self.sample_mean) / self.n
		var_sample = (x - old_mean) * (x - self.sample_mean)
		self.ewmv += (var_sample - self.ewmv) / self.n
	
			
class Sarsa:
	def __init__(self, gamma=1.0, lamda=0.4,alpha=0.1,size=10,numactions=3):
		self.gamma = gamma
		self.alpha = alpha
		self.alpha_r = 0.01*alpha
		self.lamda = lamda
		self.xt = None
		self.xtp1= None
		self.w = None
		self.e = None
		self.Ravg = 0
		self.numactions = numactions
		self.reset(size,numactions)
		
	def get_action_egreedy(self,x,epsilon):
		v = numpy.zeros(self.numactions)
		# Compute action values for each action
		for i in range(0,self.numactions):	
			v[i] = numpy.dot(x, (self.w[:,i]).flatten())
		# Perform e-greedy selection
		if random.random() > epsilon:
			a = numpy.argmax(v)
		else:
			a = random.randint(0,self.numactions-1)
		return a, v

	def get_action_softmax(self,x):
		v = numpy.zeros(self.numactions)
		p = numpy.zeros(self.numactions)
		# Compute values and probabilities for each action	
		for i in range(0,self.numactions):	
			v[i] = numpy.exp(numpy.dot(x, (self.w[:,i]).flatten()))	
		for i in range(0,self.numactions):
			p[i] = v[i]/v.sum()
		#Wheel of fortune for action selection
		prob = random.random()
		psum=0
		for i in range(0,self.numactions):
			if  prob < p[i]+psum:
				a = i
				break
			else:
				psum += p[i]
		return a, p
		
	def reset(self,size,numactions):
		self.xt = numpy.array(numpy.zeros((size,numactions)))
		self.xtp1 = numpy.array(numpy.zeros((size,numactions)))
		self.w = numpy.array(numpy.zeros((size,numactions)))
		self.e = numpy.array(numpy.zeros((size,numactions)))
				
	def update(self,xt,xtp1,Rtp1,At,Atp1,gamma):
		self.gamma = gamma
		self.xt = self.xt*0
		self.xt[:,At] = xt
		self.xtp1 = self.xtp1*0
		self.xtp1[:,Atp1] = xtp1
		self.R = Rtp1
		self.delta = Rtp1 - self.Ravg + self.gamma*np.dot(self.xtp1.flatten(),self.w.flatten())-np.dot(self.xt.flatten(),self.w.flatten())
		self.Ravg = self.Ravg + self.delta*self.alpha_r
		self.e = self.gamma*self.lamda*self.e + self.xt
		self.w = self.w + self.alpha*self.delta*self.e
		return self.delta	