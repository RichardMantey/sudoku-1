# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
# from hyperparams import Hyperparams as hp

# fpath = hp.train_fpath if type=="train" else hp.test_fpath
#     lines = open(fpath, 'r').read().splitlines()[1:]
#     nsamples = len(lines)
    
#     X = np.zeros((nsamples, 9*9), np.float32)  
#     Y = np.zeros((nsamples, 9*9), np.int32) 
    
#     for i, line in enumerate(lines):
#         quiz, solution = line.split(",")
#         for j, (q, s) in enumerate(zip(quiz, solution)):
#             X[i, j], Y[i, j] = q, s

filename = 'data/sudoku.npz'
data = np.load(filename)
quizzes = data['quizzes']
solutions = data['solutions']
quizzes_flat = quizzes.reshape((1000000, -1))
solutions_flat = solutions.reshape((1000000, -1))
rows = quizzes_flat.shape[0]
cols = quizzes_flat.shape[1]

file = open("data/sudoku.csv", "w")
for i in range(rows):
	quiz_str = str(quizzes_flat[i,])[1:-1].replace(" ", "").replace('\n', "")
	soln_str = str(solutions_flat[i,])[1:-1].replace(" ", "").replace('\n', "")
	value = quiz_str+','+soln_str+"\n"

	file.write(value)
file.close()

print "Done"

# print quizzes.shape
# print quizzes_flat.shape
# print solutions.shape
# print np.array2string(quizzes_flat[0], separator='')[1:-1]