from __future__ import division
import unittest
from features.syntactic import Constituents
from dump_vectors import BookDataWrapper
import numpy as np


__author__='suraj'

class TestConstituents(unittest.TestCase):
    def setUp(self):
        self.phrasal = Constituents(PHR=True)
        self.clausal = Constituents(CLS=True)
        self.phr_cls = Constituents(PHR=True, CLS=True)

        self.parse_trees = [

            """
(ROOT
  (FRAG
    (NP
      (NP (JJ voluptuous) (NNS curves))
      (SBAR
        (S
          (NP
            (NP (DT another) (NN world))
            (CC and)
            (NP (DT another) (NN life)))
          (VP (VBD existed)
            (ADVP (RB exclusively))
            (PP (IN for)
              (NP
                (NP (PRP her))
                (, ,)
                (NP (DT a) (NN world))))))))))
            """,
            """
            (ROOT
  (S
    (PP (IN On)
      (NP
        (NP (PRP$ her) (NNS knees))
        (, ,)
        (ADJP (JJ numb)
          (PP (TO to)
            (NP
              (NP (DT the) (NN discomfort))
              (PP (IN of)
                (NP
                  (NP (JJ crushed) (NNS stones))
                  (PP (IN in)
                    (NP (PRP$ her)
                      (NX (NNS shins)))))))))))
    (, ,)
    (NP (NNP Damascena))
    (VP (VBD leaned)
      (PP (IN into)
        (NP
          (NP (DT the) (JJ grayish) (JJ pink) (NN bud))
          (, ,)
          (VP (VBN overcome)
            (PP (IN by)
              (NP (PRP$ its) (JJ delicate) (NN scent)))))))
    (. .)))
            """,

            """
            (ROOT
  (S
    (S
      (NP (PRP She))
      (VP
        (VP (VBD ran)
          (PP (TO to)
            (NP (DT the) (NN backyard))))
        (, ,) (RB not)
        (VP (VBD moved)
          (PP
            (PP (IN by)
              (NP
                (NP (DT the) (NN brilliance))
                (PP (IN of)
                  (NP (DT the) (NN dewdrop)))))
            (, ,)
            (CC but)
            (PP (IN by)
              (NP (PRP$ its) (JJ very) (NN setting)))))))
    (: :)
    (S
      (NP (SYM a))
      (VP (VBD rose)
        (NP
          (NP
            (NP (NN bud) (VBG growing))
            (PP (IN in)
              (NP (JJ loamy) (NN soil))))
          (CC and)
          (NP
            (NP (DT a) (NN ring))
            (PP (IN of)
              (NP (NNS weeds)))))))
    (. .)))
            """,

        ]

        # Damascena: The Tale Of Roses And Rumi

        self.book = BookDataWrapper(book_id='3ZH6oAEACAAJ', isbn_10='0982279744', content='')
        self.book._parse_tree = self.parse_trees

        self.empty_book = BookDataWrapper(book_id='', isbn_10='', content='')
        self.empty_book._parse_tree = []

    def test_syntactic_tag_count(self):
        # test the counting of the tags phrasal
        parse_tree = [tree.replace('\n', ' ') for tree in
                      self.parse_trees]  # as done before coputing the count in the class
        self.assertEqual(self.phrasal._count_constituents(['ADJP'], parse_tree)[0], 1, 'ADJP Count Mismatch')
        self.assertEqual(self.phrasal._count_constituents(['ADVP'], parse_tree)[0], 1, 'ADVP Count Mismatch')
        self.assertEqual(self.phrasal._count_constituents(['CONJP'], parse_tree)[0], 0, 'CONJP Count Mismatch')
        self.assertEqual(self.phrasal._count_constituents(['FRAG'], parse_tree)[0], 1, 'FRAG Count Mismatch')
        self.assertEqual(self.phrasal._count_constituents(['LST'], parse_tree)[0], 0, 'LST Count Mismatch')
        self.assertEqual(self.phrasal._count_constituents(['NAC'], parse_tree)[0], 0, 'NAC Count Mismatch')
        self.assertEqual(self.phrasal._count_constituents(['NP'], parse_tree)[0], 33, 'NP Count Mismatch')
        self.assertEqual(self.phrasal._count_constituents(['NX'], parse_tree)[0], 1, 'NX Count Mismatch')
        self.assertEqual(self.phrasal._count_constituents(['PP'], parse_tree)[0], 14, 'PP Count Mismatch')
        self.assertEqual(self.phrasal._count_constituents(['PRN'], parse_tree)[0], 0, 'PRN Count Mismatch')
        self.assertEqual(self.phrasal._count_constituents(['PRT'], parse_tree)[0], 0, 'PRT Count Mismatch')
        self.assertEqual(self.phrasal._count_constituents(['QP'], parse_tree)[0], 0, 'QP Count Mismatch')
        self.assertEqual(self.phrasal._count_constituents(['RRC'], parse_tree)[0], 0, 'RRC Count Mismatch')
        self.assertEqual(self.phrasal._count_constituents(['UCP'], parse_tree)[0], 0, 'UCP Count Mismatch')
        self.assertEqual(self.phrasal._count_constituents(['VP'], parse_tree)[0], 7, 'VP Count Mismatch')
        self.assertEqual(self.phrasal._count_constituents(['WHADJP'], parse_tree)[0], 0, 'WHADJP Count Mismatch')
        self.assertEqual(self.phrasal._count_constituents(['WHAVP'], parse_tree)[0], 0, 'WHAVP Count Mismatch')
        self.assertEqual(self.phrasal._count_constituents(['WHNP'], parse_tree)[0], 0, 'WHNP Count Mismatch')
        self.assertEqual(self.phrasal._count_constituents(['WHPP'], parse_tree)[0], 0, 'WHPP Count Mismatch')
        self.assertEqual(self.phrasal._count_constituents(['X'], parse_tree)[0], 0, 'X Count Mismatch')

        # test the counting of the clausal tags
        self.assertEqual(self.clausal._count_constituents(['SBAR'], parse_tree)[0], 1, 'SBAR Count Mismatch')
        self.assertEqual(self.clausal._count_constituents(['SQ'], parse_tree)[0], 0, 'SQ Count Mismatch')
        self.assertEqual(self.clausal._count_constituents(['SBARQ'], parse_tree)[0], 0, 'SBARQ Count Mismatch')
        self.assertEqual(self.clausal._count_constituents(['SINV'], parse_tree)[0], 0, 'SINV Count Mismatch')
        self.assertEqual(self.clausal._count_constituents(['S'], parse_tree)[0], 5, 'S Count Mismatch')

        # test with empty parse tree
        self.assertEqual(self.clausal._count_constituents(['S'], [])[0], 0, 'S Count Mismatch for empty parse tree')

    def test_get_feature_names(self):
        # test the feature names
        self.assertEqual(set(self.phrasal.get_feature_names().tolist()), set(
            ['ADJP', 'ADVP', 'CONJP', 'FRAG', 'LST', 'NAC', 'NP', 'NX', 'PP', 'PRN', 'PRT', 'QP', 'RRC', 'UCP', 'VP',
             'WHADJP', 'WHAVP', 'WHNP', 'WHPP', 'X']))

        self.assertEqual(set(self.clausal.get_feature_names().tolist()), set(['SBAR', 'SQ', 'SBARQ', 'SINV', 'S']))

        self.assertEqual(set(self.phr_cls.get_feature_names().tolist()), set(
            ['ADJP', 'ADVP', 'CONJP', 'FRAG', 'LST', 'NAC', 'NP', 'NX', 'PP', 'PRN', 'PRT', 'QP', 'RRC', 'UCP', 'VP',
             'WHADJP', 'WHAVP', 'WHNP', 'WHPP', 'X'] + ['SBAR', 'SQ', 'SBARQ', 'SINV', 'S']))

    def test_transform(self):
        phrasal_features_count = 20
        clausal_features_count = 5
        book_lst = [self.book, self.empty_book]
        X_phrasal_transformed = self.phrasal.transform(book_lst)
        # test the shape of transformed numpy array
        self.assertEqual(X_phrasal_transformed.shape, (len(book_lst), phrasal_features_count),
                         "Transformed features shape mis match")

        # test the feature values
        total = 1 + 1 + 0 + 1 + 0 + 0 + 33 + 1 + 14 + 0 + 0 + 0 + 0 + 0 + 7 + 0 + 0 + 0 + 0 + 0

        feature_values = np.array([[1 / total, 1 / total, 0 / total, 1 / total, 0 / total, 0 / total, 33 / total,
                                    1 / total, 14 / total, 0 / total, 0 / total, 0 / total, 0 / total, 0 / total,
                                    7 / total, 0 / total, 0 / total, 0 / total, 0 / total, 0 / total],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0]
                                   ])

        self.assertEqual(feature_values.all(), X_phrasal_transformed.all(), "Feature values mistmatch")




if __name__ == "__main__":
    unittest.main()
