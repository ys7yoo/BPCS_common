# encoding: utf-8

# Copyright (c) 2011, François Sausset <sausset@gmail.com>
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The names of its contributors cannot be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE CONTRIBUTORS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import time
import random
import numpy as np
import scipy as sc
from scipy import sparse

class Tap():
	"""docstring for Tap"""
	def __init__(self, y, F, rho=None, mean=None, variance=None, noise=None):
		np.random.seed(int(time.time()))
		if not rho:
			self.rho_ = float(F.shape[0]) / F.shape[1]
			self.rhoLearning_ = True
		else:
			self.rho_ = rho
			self.rhoLearning_ = False
		if mean == None:
			self.mean_ = 0
			self.meanLearning_ = True
		else:
			self.mean_ = mean
			self.meanLearning_ = False
		if not variance:
			self.sigma2_ = np.sum(y ** 2) / (self.rho_ * np.sum(F.multiply(F).todense())) - self.mean_ ** 2
			self.sigma2Learning_ = True
		else:
			self.sigma2_ = variance
			self.sigma2Learning_ = False
		if not noise:
			self.noise_ = 1
			self.noiseLearning_ = True
		else:
			self.noise_ = noise
			self.noiseLearning_ = False
		self.tapAlpha_ = y
		self.tapGamma_ = 1 - np.random.rand(np.size(y))
		self.U_ = np.random.rand(F.shape[1]) + 1 / self.sigma2_
		self.V_ = 2 * np.random.rand(F.shape[1]) - 1 + self.mean_ / self.sigma2_
		self.F_ = F
		self.y_ = y
		self.F2_ = F.multiply(F)
	
	def thresholdA(self, x, y):
		"""docstring for thresholdA"""
		return self.rho_ * y / (x * \
			(self.rho_ + (1 - self.rho_) * np.sqrt(self.sigma2_ * x) * \
			np.exp(-y**2 / (2 * x) + self.mean_ ** 2 / (2 * self.sigma2_))))
	
	def thresholdADerivative(self, x, y):
		"""docstring for thresholdADerivative"""
		exp = (1 - self.rho_) * np.sqrt(self.sigma2_ * x) * \
			np.exp(-y**2 / (2 * x) + self.mean_ ** 2 / (2 * self.sigma2_))
		return self.thresholdA(x, y) / y + y**2 * exp * self.rho_ / (x * (self.rho_ + exp))**2
	
	def thresholdC(self, x, y):
		"""docstring for thresholdC"""
		exp = (1 - self.rho_) * np.sqrt(self.sigma2_ * x) * \
			np.exp(-y**2 / (2 * x) + self.mean_ ** 2 / (2 * self.sigma2_))
		return self.rho_ / x * ((1 + y ** 2 / x) * exp + self.rho_) / (self.rho_ + exp) ** 2
	
	def solution(self):
		"""docstring for solution"""
		return self.thresholdA(self.U_, self.V_)
	
	def iterate(self, damping, verbose):
		"""docstring for iterate"""
		oldX = self.thresholdA(self.U_, self.V_)
		
		# Partially TAP-ified equations.
		newV = ((self.y_ - self.tapAlpha_) / self.tapGamma_) * self.F_ + oldX * \
			((1. / self.tapGamma_) * self.F2_) + self.mean_ / self.sigma2_
		newU = (1. / self.tapGamma_) * self.F2_ + 1 / self.sigma2_
		self.V_ = damping * newV + (1 - damping) * self.V_
		self.U_ = damping * newU + (1 - damping) * self.U_
		newTapAlpha = self.F_ * self.thresholdA(self.U_, self.V_) - \
			(self.y_ - self.tapAlpha_) / self.tapGamma_ * \
			(self.F2_ * self.thresholdADerivative(self.U_, self.V_))
		newTapGamma = self.noise_ + self.F2_ * self.thresholdC(self.U_, self.V_)
		self.tapAlpha_ = damping * newTapAlpha + (1 - damping) * self.tapAlpha_
		self.tapGamma_ = damping * newTapGamma + (1 - damping) * self.tapGamma_
		
		# Prior parameters learning.
		if self.rhoLearning_ or self.meanLearning_ or self.sigma2Learning_:
			exp = np.exp(- self.V_ ** 2 / (2 * self.U_) + self.mean_ ** 2 / (2 * self.sigma2_))
			sqrt = np.sqrt(self.sigma2_ * self.U_)
			if self.meanLearning_ or self.sigma2Learning_:
				adjust = np.sum(1 / (sqrt * (1 - self.rho_) * exp + self.rho_))
		if self.meanLearning_:
			newMean = np.sum(self.thresholdA(self.U_, self.V_)) / (self.rho_ * adjust)
		if self.sigma2Learning_:
			newSigma2 = np.sum(self.thresholdC(self.U_, self.V_) + \
				self.thresholdA(self.U_, self.V_) ** 2) / (self.rho_ * adjust) - self.mean_ ** 2
			if newSigma2 < 0:
				newSigma2 = 1e-10
		if self.rhoLearning_:
			newRho = np.sum(self.U_/ self.V_ * self.thresholdA(self.U_, self.V_)) / \
				np.sum(exp / ((1 - self.rho_) * exp + self.rho_ / sqrt))
			if newRho < 0:
				newRho = 1e-10
			elif newRho > 1:
				newRho = 1
		if self.noiseLearning_:
			newNoise = np.sum((self.y_ - self.tapAlpha_) ** 2 / (1 + self.tapGamma_ / self.noise_) ** 2) / \
				np.sum(1. / (1 + self.tapGamma_ / self.noise_))
			if newNoise < 0:
				newNoise = 1e-30
		
		if self.rhoLearning_:
			self.rho_ = damping * newRho + (1 - damping) * self.rho_
		if self.meanLearning_:
			self.mean_ = damping * newMean + (1 - damping) * self.mean_
		if self.sigma2Learning_:
			self.sigma2_ = damping * newSigma2 + (1 - damping) * self.sigma2_
		if self.noiseLearning_:
			self.noise_ = damping * newNoise + (1 - damping) * self.noise_
		
		if verbose:
			print("Learned sparsity = " + repr(self.rho_) + \
				" ; learned mean = " + repr(self.mean_) + \
				" ; learned variance = " + repr(self.sigma2_) + \
				" ; learned noise = " + repr(self.noise_))
		
		return np.mean(np.abs(self.thresholdA(self.U_, self.V_) - oldX))
	
	def run(self, maxIter=5000, convergence=1e-7, damping=0.5, verbose=False):
		"""docstring for BEP"""
		it = 0
		conv = convergence + 1
		while it < maxIter and conv > convergence:
			conv = self.iterate(damping, verbose)
			it += 1
			if verbose:
				print("Iteration number: " + repr(it))
		
		return (it, conv)
	
