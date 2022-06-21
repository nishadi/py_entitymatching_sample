import csv
import logging
import os
import resource
import sys
import time
import numpy as np

import fcntl
import py_entitymatching as em


def _extract_feature_vectors(path_A, path_B, path_pairs_file,
                             train_feature_subset=None):
  start_time = time.time()
  # Read input tables
  A = em.read_csv_metadata(path_A, key='id')
  B = em.read_csv_metadata(path_B, key='id')

  logging.info('Number of tuples in A: ' + str(len(A)))
  logging.info('Number of tuples in B: ' + str(len(B)))
  logging.info('Number of tuples in A X B (i.e the cartesian product): ' + str(
    len(A) * len(B)))

  # Read the candidate set
  pairs = em.read_csv_metadata(path_pairs_file, key='_id', ltable=A,
                               rtable=B,
                               fk_ltable='table1.id',
                               fk_rtable='table2.id')

  if train_feature_subset is None:
    # Get features
    feature_table = em.get_features_for_matching(A, B,
                                                 validate_inferred_attr_types=False)

    # Remove ID based features
    logging.info('All features {}'.format(feature_table['feature_name']))
    train_feature_subset = feature_table[4:]
    logging.info('Selected features {}'.format(train_feature_subset))

  # Generate features
  feature_vectors = em.extract_feature_vecs(pairs,
                                            feature_table=train_feature_subset,
                                            attrs_after='label')

  # Impute feature vectors with the mean of the column values.
  feature_vectors = em.impute_table(feature_vectors,
                                    exclude_attrs=['_id', 'table1.id',
                                                   'table2.id', 'label'],
                                    strategy='mean', missing_val=np.NaN)
  t = time.time() - start_time
  return feature_vectors, t, train_feature_subset


def mg_predict(train, test,  t_train, t_test, dataset_name):
  '''
  Predict results using four different classifiers of Magellan
  and average the results

  :param train: Train data frame
  :param test: Test data frame
  :return:
  '''
  # Create a set of ML-matchers
  dt = em.DTMatcher(name='DecisionTree')
  svm = em.SVMMatcher(name='SVM')
  rf = em.RFMatcher(name='RF')
  lg = em.LogRegMatcher(name='LogReg')

  # Train and eval on different classifiers
  for clf, clf_name in [(dt, 'DecisionTree'), (svm, 'SVM'), (rf, 'RF'),
                        (lg, 'LogReg')]:
    start_time = time.time()
    clf.fit(table=train,
            exclude_attrs=['_id', 'table1.id', 'table2.id', 'label'],
            target_attr='label')
    t_train += time.time() - start_time
    train_max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Predict M
    start_time = time.time()
    predictions = clf.predict(table=test,
                              exclude_attrs=['_id', 'table1.id', 'table2.id',
                                             'label'],
                              append=True, target_attr='predicted',
                              inplace=False)

    # Evaluate the result
    eval_result = em.eval_matches(predictions, 'label', 'predicted')
    t_test += time.time() - start_time
    test_max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    em.print_eval_summary(eval_result)

    p = eval_result['precision']
    r = eval_result['recall']
    if (p + r - p * r) == 0:
      f_star = 0
    else:
      f_star = p * r / (p + r - p * r)
    logging.info('---{} p{} r{} fst{}'.format(clf_name, p, r, f_star))

    p = round(p * 100, 2)
    r = round(r * 100, 2)
    f_star = round(f_star * 100, 2)

    result_file = '/home/remote/u6852937/projects/results.csv'
    file_exists = os.path.isfile(result_file)
    with open(result_file, 'a') as results_file:
      heading_list = ['method', 'dataset_name', 'train_time', 'test_time',
                      'train_max_mem', 'test_max_mem', 'TP', 'FP', 'FN',
                      'TN', 'Pre', 'Re', 'F1', 'Fstar']
      writer = csv.DictWriter(results_file, fieldnames=heading_list)

      if not file_exists:
        writer.writeheader()

      fcntl.flock(results_file, fcntl.LOCK_EX)
      result_dict = {
        'method': 'magellan' + clf_name,
        'dataset_name': dataset_name,
        'train_time': round(t_train, 2),
        'test_time': round(t_test, 2),
        'train_max_mem': train_max_mem,
        'test_max_mem': test_max_mem,
        'TP': eval_result['pred_pos_num'] - eval_result['false_pos_num'],
        'FP': eval_result['false_pos_num'],
        'FN': eval_result['false_neg_num'],
        'TN': eval_result['pred_neg_num'] - eval_result['false_neg_num'],
        'Pre': ('{prec:.2f}').format(prec=p),
        'Re': ('{rec:.2f}').format(rec=r),
        'F1': ('{f1:.2f}').format(f1=round(eval_result['f1'] * 100, 2)),
        'Fstar': ('{fstar:.2f}').format(fstar=f_star)
      }
      writer.writerow(result_dict)
      fcntl.flock(results_file, fcntl.LOCK_UN)


if __name__ == '__main__':
  # Reading input
  path = sys.argv[1]
  dataset_name = sys.argv[2]

  path_A = path + 'tableA.csv'
  path_B = path + 'tableB.csv'
  path_train = path + 'train.csv'
  path_test = path + 'test.csv'

  # Generate training feature vectors using full data set
  train, t_train, train_feature_subset = _extract_feature_vectors(path_A,
                                                                  path_B,
                                                             path_train)

  # Generate testing feature vectors using only the specified link
  test, t_test, _ = _extract_feature_vectors(path_A, path_B, path_test,
                                         train_feature_subset)

  mg_predict(train, test, t_train, t_test, dataset_name)
