# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: 
# https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import math
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter)
# will list the files in the input directory

# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# sample = pd.read_csv("../input/sample_submission.csv", sep='\t')
# print('Loading Dataset...')
# train = pd.read_csv("train.tsv", sep='\t')
# test = pd.read_csv("test.tsv", sep='\t')
# print('Dataset loaded!...')

# Any results you write to the current directory are saved as output.


class Evaluator(object):
  """
  Object used to evaluate the test or validation set. 
  """

  def __init__(self, y_test):
    self.y_test = y_test

  def rmsle(self, y_pred):
    assert len(y_pred) == len(self.y_test)
    return np.sqrt(
        np.mean(np.power(np.log1p(y_pred) - np.log1p(self.y_test), 2)))

  def rmsle_loop(self, y_pred):
    assert len(y_pred) == len(self.y_test)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(self.y_test[i] + 1))
                    **2.0 for i, pred in enumerate(self.y_test)]
    return (sum(terms_to_sum) * (1.0 / len(self.y_test)))**0.5

