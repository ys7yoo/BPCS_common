#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2011, François Sausset <sausset@gmail.com>
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must stroduce the above copyright
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

import sys
import getopt
import time
import numpy as np
import scipy as sc
from scipy import sparse
from math import *
import pywt
import matplotlib.pylab as pl
import phantom
import BP

help_message = '''
Usage: comparison [options] alpha
- needed argument: alpha (measurement ratio)
- options:
	-v -> verbose mode
	--n=? -> phantom size (power of 2)
	--dmp=? -> damping parameter
	--iter=? -> maximum number of iterations
	--L=? -> number of blocks in the seeded case
	--J1=? -> first coupling parameter in the seeded case
	--J2=? -> second coupling parameter in the seeded case
	--alpha0=? -> first block measurement ratio in the coupled case
	--file=? -> path of a grayscale PNG image file with a square size (power of 2)
'''

class Usage(Exception):
	def __init__(self, msg):
		self.msg = msg


def generatePhantom(phantomSize=32, filePath=""):
	"""docstring for generatePhantom"""
	if filePath:
		P = pl.imread(filePath)
		phantomSize = P.shape[0]
	else:
		P = phantom.phantom(n=phantomSize)
	coeffs = pywt.wavedec2(P, 'haar')
	coeffs = np.array(coeffs)
	iter = int(log(phantomSize,2))
	x_0 = []
	for	i in range(0,iter + 1):
		x_0 = np.concatenate((x_0, np.array(coeffs[i]).ravel()))
	
	return {'image': P, 'coeffs': x_0}


def decodeSolution(solution):
	"""docstring for decodeSolution"""
	iter = int(log(np.size(solution), 2)) / 2
	solutionCoeffs = [np.reshape(solution[0], (1,1))]
	for i in range(0, iter):
		coeff1 = np.reshape(solution[4 ** i:2 * 4 ** i], (2 ** i,2 ** i))
		coeff2 = np.reshape(solution[2 * 4 ** i:3 * 4 ** i], (2 ** i,2 ** i))
		coeff3 = np.reshape(solution[3 * 4 ** i:4 * 4 ** i], (2 ** i,2 ** i))
		solutionCoeffs.append((coeff1, coeff2, coeff3))
	return pywt.waverec2(solutionCoeffs, 'haar')


def main(argv=None):
	# Default values
	verbose = False
	phantomSize = 64
	damping = 0.5
	maxIter = 1000
	L = 10
	J1 = 40
	J2 = 0.2
	alpha0 = 0.9
	filePath = ""
	
	# Arguments parsing.
	if argv is None:
		argv = sys.argv
	try:
		try:
			opts, args = getopt.getopt(argv[1:], "ho:v", ["help", "n=", "dmp=", "iter=", "L=", "J1=", "J2=", "alpha0=", "file="])
		except getopt.error, msg:
			raise Usage(msg)
		
		# Options processing.
		for option, value in opts:
			if option == "-v":
				verbose = True
			if option in ("-h", "--help"):
				raise Usage(help_message)
			if option == "--n":
				phantomSize = int(value)
			if option == "--dmp":
				damping = float(value)
			if option == "--iter":
				maxIter = int(value)
			if option == "--L":
				L = int(value)
			if option == "--J1":
				J1 = float(value)
			if option == "--J2":
				J2 = float(value)
			if option == "--alpha0":
				alpha0 = float(value)
			if option == "--file":
				filePath = value
		
		# Arguments processing.
		if not args:
			raise Usage("\n \t Missing arguments!")
		else:
			alpha = float(args[0])
	
	except Usage, err:
		print >> sys.stderr, sys.argv[0].split("/")[-1] + ": " + str(err.msg)
		print >> sys.stderr, "\t for help use --help"
		return 2
		
	np.random.seed(int(time.time()))
	P = generatePhantom(phantomSize, filePath)
	if filePath:
		phantomSize = P['image'].shape[0]
		origin = 'upper'
	else:
		origin = 'lower'
	x_0 = P['coeffs']
	size = np.size(x_0)
	
	print("The sparsity of the signal is " + str(float((x_0 != 0).sum(0)) / size))
	print("The chosen measurement ratio is " + str(alpha))
	
	#  Plot original picture.
	pl.figure(figsize=(3.8,1.25), dpi=phantomSize, facecolor='w')
	pl.figimage(P['image'], xo=int(0.2 * P['image'].shape[0]), cmap='gray', origin=origin)
	pl.figtext(0.18, 0.84, 'Original', ha='center', va='bottom', size=14)
	pl.draw()
	
	subSize = size // L
	measurements = int(alpha * size)
	
	# Basic case.
	F = np.random.randn(measurements, size) / np.sqrt(size)
	y = np.dot(F, x_0)
	
	# BP reconstruction
	F = sc.sparse.csr_matrix(F)
	y = F * x_0
	print("EM-BP recovery with learning (Be patient...)")
	decode = BP.Tap(y, F, noise=1e-30)
	iterations, meanDiff = decode.run(maxIter=maxIter, damping=damping, verbose=verbose)
	if verbose:
		print(iterations, meanDiff)
	print("Learned sparsity = " + str(decode.rho_))
	print("Learned mean = " + str(decode.mean_))
	print("Learned variance = " + str(decode.sigma2_))
	print("	DONE.")
		
	solutionBP = decodeSolution(decode.solution())
	
	pl.figimage(solutionBP, xo=int(1.4 * solutionBP.shape[0]), cmap='gray', origin=origin)
	pl.figtext(0.5, 0.84, 'Regular BP', ha='center', va='bottom', size=14)
	pl.draw()
		
	# Seeded case.
	F = sc.sparse.lil_matrix((measurements, size))
	print("Seeded BP recovery with learning (Be patient...)")
	alphas = np.empty(L)
	alphas[0] = alpha0
	alphas[1:] = (alpha * L - alphas[0]) / (L - 1)
	meas = int(alphas[0] * subSize)
	sigma = sqrt(subSize)
	F[:meas, :subSize] = np.random.randn(meas, subSize) / sigma
	if L != 1:
		F[:meas, subSize:2 * subSize] = sqrt(J2) * np.random.randn(meas, subSize) / sigma
		for	l in range(1, L-1):
			beginMeas = meas
			endMeas = beginMeas + int(alphas[l] * subSize)
			i1, i2, i3, i4 = (l - 1) * subSize, l * subSize, (l + 1) * subSize, (l + 2) * subSize
			F[beginMeas:endMeas, i1:i2] = sqrt(J1) * np.random.randn(endMeas - beginMeas, subSize) / sigma
			F[beginMeas:endMeas, i2:i3] = np.random.randn(endMeas - beginMeas, subSize) / sigma
			F[beginMeas:endMeas, i3:i4] = sqrt(J2) * np.random.randn(endMeas - beginMeas, subSize) / sigma
			meas = endMeas
		F[meas:, (L - 2) * subSize:(L - 1) * subSize] = sqrt(J1) * np.random.randn(measurements - meas, subSize) / sigma
		F[meas:, (L - 1) * subSize:] = np.random.randn(measurements - meas, size + (1 - L) * subSize) / sigma
	# Permuting the columns of F to probe wavelets coefficients randomly.
	# To improve by avoiding to have an intermediate dense array.
	F = sc.sparse.csr_matrix(np.random.permutation(F.transpose().todense()).T)
	F.eliminate_zeros()
	
	y = F * x_0
	
	decode = BP.Tap(y, F, noise=1e-30)
	iterations, meanDiff = decode.run(maxIter=maxIter, damping=damping, verbose=verbose)
	if verbose:
		print("Convergence after " + str(iterations) + \
			" iterations and a mean update of the decoded signal of " + str(meanDiff))
	print("Learned sparsity = " + str(decode.rho_))
	print("Learned mean = " + str(decode.mean_))
	print("Learned variance = " + str(decode.sigma2_))
	print("	DONE.")
	
	solutionSeededBP = decodeSolution(decode.solution())
	
	pl.figimage(solutionSeededBP, xo=int(2.6 * solutionSeededBP.shape[0]), cmap='gray', origin=origin)
	pl.figtext(0.82, 0.84, 'Seeded BP', ha='center', va='bottom', size=14)
	pl.savefig('comparison.png', dpi=phantomSize)
	pl.show()

if __name__ == '__main__':
	sys.exit(main())

