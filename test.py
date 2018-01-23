import unittest
import numpy as np
from kernel import Evaluator, get_q
import gc
import pandas as pd


class TestUM(unittest.TestCase):
  def setUp(self):
    pass

  def test_rmsle_same(self):
    self.assertEqual(
        Evaluator(np.array([1, 2, 3])).rmsle(np.array([1, 2, 3])), 0)

  def test_rmsle_diff(self):
    self.assertEqual(
        Evaluator(np.array([1.1, 2.2, 3])).rmsle(np.array([1, 2, 3.1])),
        0.048837915417030281)

  def test_rmsle_zeros(self):
    self.assertEqual(Evaluator(np.zeros(5)).rmsle(np.zeros(5)), 0.0)

  def test_rmsle_loop(self):
    self.assertEqual(
        Evaluator(np.array([1, 2, 3])).rmsle_loop(np.array([1, 2, 3])), 0)
    self.assertEqual(
        Evaluator(np.array([1, 2, 3])).rmsle_loop(np.array([1, 2, 3.1])),
        0.014256286526046115)


#     self.assertEqual(Evaluator(np.zeros(5)).rmsle_loop(np.zeros(5)), 0.0)


class TestUM(unittest.TestCase):
  def setUp(self):
    pass

  def test_get_q(self):
    data_1 = {
        "item_description": "hey just trying 23 things",
        "name": "23 different items"
    }
    data_2 = {
        "item_description": "3 chain versace bracelet",
        "name": "Vintage Versace 3 chain bracelet"
    }

    data_3 = {
        "item_description": "38 irmas/perfect t solids mix",  #
        "name": "Bundle for kels1669"
    }

    data_4 = {
        "item_description": "Foundation, eyeshadow,concealer and much more \
        80+ pieces of make up",
        "name": "Make up"
    }

    data_5 = {
        "item_description": "Girls Nike Shox size 2c EEUC, no noticeable \
        flaws",
        "name": "Nike Shox size 2c"  #559
    }

    data_6 = {
        "item_description": "NWT 32D Victoria Secret Bling Lined Demi Bra",
        "name": "NWT 32D Victoria Secret Bling Bra"  #152
    }

    data_7 = {
        "item_description":
        "'MOOD COLOR CHANGE ~ GEL NAIL POLISH NEW! ONE BOTTLE\
         ~ #04 LOVELY SHADE NO LIGHT NEEDED REMINDER THAT COLORS CAN VARY SINCE \
         ITS MOOD CHANGING. ALSO CHANGES IN WARM OR COOL SETTINGS. \
         SUPER COOL!!! BRAND NEW!!!'",
        "name": "MOOD COLOR CHANGE ~ #04 NAIL POLISH NEW!"  #1134591
    }

    data_8 = {
        "item_description": "Alienware x51 r2 Blue ray disc drive",
        "name":
        'Taken from a working desktop, replaced slot with 2.5" SSD'  #322667
    }

    data_8 = {
        "item_description":
        """Brand new with box Adjustable ankle strap. Silver\
         42mm - This bracelet is designed by ANCOOL, a new and unique look on\
          your watch. - Excellent stainless steel made, soft to wear indoor\
           & outdoor. - The design is an idea about Traditional Chinese Word\
            "Happy", it means a double blessing to your family. Parameter: -\
             Bracelet length: 160mm 42mm fits wrist 6.50"-7.87" (165mm-200mm)\
              Package include: - 1 * metal strap - 1 * link removal tool - 1\
               * screwdriver""",
        "name": "Apple Watch Stainless Steel Brace 42mm"  #923655
    }

    data_9 = {
        "item_description": "Victoria secret 34 c corest top Will bundle to\
         save on shipping If you have any questions please feel free to ask",
        "name": "Victoria secret 34 c corest top"  #39
    }


    # self.assertEqual(get_q(data_1), 23)
    # self.assertEqual(get_q(data_2), 3)
    # self.assertEqual(get_q(data_3), 1)

  def test_get_q_df(self):
    print('Loading Dataset...')
    train = pd.read_table("train.tsv")
    test = pd.read_table("test.tsv")
    print('Dataset loaded!...')

    dftt = train[(train.price < 1.0)]
    train = train.drop(train[(train.price < 1.0)].index)
    print(train.shape)
    del dftt['price']
    nrow_train = train.shape[0]  #-dftt.shape[0]
    # nrow_test = train.shape[0] + dftt.shape[0]
    print(train.shape, dftt.shape, test.shape)
    # merge: pd.DataFrame = pd.concat([train, dftt])
    merge = pd.concat([train, dftt])
    print(merge.shape)
    merge['item_description'] = merge['item_description'].fillna('None')
    merge['name'] = merge['name'].fillna('None')

    del train
    del test
    gc.collect()

    data = merge[['item_description', 'name']]

    nums = data.apply(get_q, axis=1)

    self.assertEqual(nums.loc[723675], 3)

  # def test_rmsle_same(self):
  #   self.assertEqual(
  #       Evaluator(np.array([1, 2, 3])).rmsle(np.array([1, 2, 3])), 0)

  # def test_rmsle_diff(self):
  #   self.assertEqual(
  #       Evaluator(np.array([1.1, 2.2, 3])).rmsle(np.array([1, 2, 3.1])),
  #       0.048837915417030378)


if __name__ == '__main__':
  unittest.main()
