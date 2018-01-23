# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: 
# https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import math,re
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

NUM_BRANDS = 4000
NUM_CATEGORIES = 1000
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 50000


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


def handle_missing_inplace(dataset):
  dataset['general_cat'].fillna(value='missing', inplace=True)
  dataset['subcat_1'].fillna(value='missing', inplace=True)
  dataset['subcat_2'].fillna(value='missing', inplace=True)
  dataset['brand_name'].fillna(value='missing', inplace=True)
  dataset['item_description'].fillna(value='missing', inplace=True)


def split_cat(text):
  try:
    return text.split("/")
  except:
    return ("No Label", "No Label", "No Label")


def get_qntys(data):
  qnty_matches = []
  qnty_re = [r'(\d+) ?x [^\d]', r'(\d+) ?pairs?']
  for r in qnty_re:
    qnty_matches.append(
        data.name.str.extract(r, flags=re.IGNORECASE, expand=False).dropna()
        .astype(int))
  return qnty_matches

def get_q(data):
  """
	Gets two columns ('name' & 'item_description') Pandas Dataset and then
	returns the expected quantity.
	"""
  des_n = re.findall(r'^([Ss]ize )\d+( size)', data['item_description'])
  name_n = re.findall(r'\d+', data['name'])
  # import ipdb; ipdb.set_trace()
  if len(des_n) == 1 and len(name_n) == 1:
    if int(des_n[0]) == int(name_n[0]):
      ret_n = int(des_n[0])
    else:
      ret_n = 1
    return ret_n
	# pass
  # return pd.concat(qnty_matches).reset_index().drop_duplicates(
  #     subset='index', keep='last').set_index('index')


def cutting(dataset):
  pop_brand = dataset['brand_name'].value_counts().loc[
      lambda x: x.index != 'missing'].index[:NUM_BRANDS]
  dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
  pop_category1 = dataset['general_cat'].value_counts().loc[
      lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
  pop_category2 = dataset['subcat_1'].value_counts().loc[
      lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
  pop_category3 = dataset['subcat_2'].value_counts().loc[
      lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
  dataset.loc[~dataset['general_cat'].isin(pop_category1),
              'general_cat'] = 'missing'
  dataset.loc[~dataset['subcat_1'].isin(pop_category2), 'subcat_1'] = 'missing'
  dataset.loc[~dataset['subcat_2'].isin(pop_category3), 'subcat_2'] = 'missing'


def to_categorical(dataset):
  dataset['general_cat'] = dataset['general_cat'].astype('category')
  dataset['subcat_1'] = dataset['subcat_1'].astype('category')
  dataset['subcat_2'] = dataset['subcat_2'].astype('category')
  dataset['item_condition_id'] = dataset['item_condition_id'].astype(
      'category')
