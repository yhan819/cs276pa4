import sys
import math
import re
import numpy as np
from sklearn import linear_model, svm, preprocessing



if __name__ == '__main__':
  sys.stderr.write('# Input arguments: %s\n' % str(sys.argv))
  r1 = sys.argv[1]
  r2 = sys.argv[2] 
  
  score1 = []
  score2 = []
  f = open(r1, 'r')
  f2 = open(r2,'r')
  for line in f:
    score1.append((line, line.split()[-1]))
  for line in f2:
    score2.append((line, line.split()[-1]))

  diff = []
  for i in range(0,len(score1)):
    diff.append((score1[i][0], score2[i][0], math.fabs(float(score1[i][1]) - float(score2[i][1]))))
  result = sorted(diff, key=lambda x: x[2], reverse=True)

  counter = 0
  for r in result:
    if counter < 15:
      print r
      counter += 1
