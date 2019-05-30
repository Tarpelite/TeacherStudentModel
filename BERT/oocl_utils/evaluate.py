from __future__ import print_function

import os
import sys
import argparse
from collections import Counter, defaultdict

AUS_PRIMARY_LABEL = ['Booking', 'CargoRelease', 'ContainerHandling', 'Customs', 'Documentation', 'General',
                     'Invoice/Payment', 'None', 'Rate',
                     'Report', 'Unclassified']
AUS_SECONDARY_LABEL = ['Booking', 'Booking-Request', 'Booking-DGN', 'Booking-Confirmation', 'Booking-Cancel',
                       'Booking-Amendment', 'Booking-FirmUp',
                       'Booking-Request-Attachment', 'CargoRelease', 'CargoRelease-PinRelease',
                       'CargoRelease-ReleaseInstruction', 'ContainerHandling',
                       'ContainerHandling-EmptyRelease', 'ContainerHandling-Reuse',
                       'ContainerHandling-EmptyRestitution', 'ContainerHandling-DND',
                       'Customs', 'Documentation-ConsignmentNote-Attachment', 'Documentation-SI',
                       'Documentation-BL-Release', 'Documentation-BL-Confirmation',
                       'Documentation-BL-Amendment', 'Documentation-BL', 'Documentation-SI-Attachment',
                       'Documentation-ArrivalNotice',
                       'General-Service/Vessel', 'Invoice/Payment', 'Invoice/Payment-Payment',
                       'Invoice/Payment-Invoice', 'None', 'Rate', 'Report', 'Unclassified']

UKD_PRIMARY_LABEL = ['None', 'Report', 'Booking', 'CargoRelease', 'ContainerHandling', 'Customs', 'Documentation',
                     'General', 'Invoice/Payment',
                     'Rate', 'Transportation', 'UCR', 'Unclassified']
UKD_SECONDARY_LABEL = ['None', 'Report', 'Booking', 'Booking-Request', 'Booking-DGN', 'Booking-Confirmation',
                       'Booking-Cancel', 'Booking-Amendment',
                       'Booking-VGM', 'Booking-Request-Attachment', 'CargoRelease-PinRelease',
                       'CargoRelease-ReleaseInstruction', 'CargoRelease-PinExtend',
                       'ContainerHandling', 'ContainerHandling-BookinRequest', 'ContainerHandling-EmptyRestitution',
                       'ContainerHandling-DND', 'Customs',
                       'Customs-SAD', 'Documentation-SI', 'Documentation-BL-Release', 'Documentation-BL-Confirmation',
                       'Documentation-BL-Amendment',
                       'Documentation-BL', 'Documentation-SI-Attachment', 'General-Service/Vessel', 'Invoice/Payment',
                       'Invoice/Payment-Payment',
                       'Invoice/Payment-Invoice', 'Rate', 'Transportation', 'Transportation-DeliveryNote',
                       'Transportation-Amendment', 'Transportation-Request',
                       'UCR', 'Unclassified']

SECONDARY_TO_PRIMARY = lambda x: ';'.join([i.split('-')[0] for i in x.split(';')])
LABEL_TO_IDX = lambda x, d: set([d[i.strip()] for i in x.split(';')])


def multilabel_for_category_evaluate(true_labels_idx, pred_labels_idx, labels, labels_to_idx, kind, verbose):
    """
    Compute precision, recall, f1-score of each category.
    """
    label_2_prf = {}
    flat_true_idx = []
    for idx in true_labels_idx:
        flat_true_idx.extend(idx)
    true_counter = Counter(flat_true_idx)
    flat_pred_idx = []
    for idx in pred_labels_idx:
        flat_pred_idx.extend(idx)
    true_counter = Counter(flat_true_idx)
    pred_counter = Counter(flat_pred_idx)
    tp_counter = defaultdict(int)
    for true_label, pred_label in zip(true_labels_idx, pred_labels_idx):
        for tp_idx in (true_label & pred_label):
            tp_counter[tp_idx] += 1
    for label in labels:
        idx = labels_to_idx[label]
        label_2_prf[label] = [0 if tp_counter[idx] == 0 else tp_counter[idx] / pred_counter[idx],
                              0 if tp_counter[idx] == 0 else tp_counter[idx] / true_counter[idx],
                              0,
                              true_counter[idx]]
    for label in label_2_prf:
        if label_2_prf[label][0] and label_2_prf[label][1]:
            label_2_prf[label][2] = float("{0:.2f}".format(2 / (1 / label_2_prf[label][0] + 1 / label_2_prf[label][1])))
        label_2_prf[label][0] = float("{0:.2f}".format(label_2_prf[label][0]))
        label_2_prf[label][1] = float("{0:.2f}".format(label_2_prf[label][1]))
    if verbose:
        for label in labels:
            print(label, label_2_prf[label][0], label_2_prf[label][1], label_2_prf[label][2], label_2_prf[label][3])
    return label_2_prf


def multilabel_classification_evaluate(true_labels_idx, pred_labels_idx, kind, verbose):
    """
    ref: https://en.wikipedia.org/wiki/Multi-label_classification#Statistics_and_evaluation_metrics
    Accuracy:  for each instance is defined as the proportion of the prediction correct labels to the total number (predicted and actual) of labels, averaged over all instances.

    Precision: the proportion of the prediction correct labels to the total number of predicted labels, averaged over all instances. (Samples)
    Recall: the proportion of the prediction correct labels to the total number of actual labels, averaged over all instances. (Samples)
    F1-score: harmonic mean of precision and recall, averaged over all instances. (Samples)

    Precision: p_micro = \sum_i |l_i \cap p_i| / (\sum_i |l_i \cap p_i| + \sum_i |p_i \subset l_i|) (micro)
    Recall: r_micro = \sum_i |l_i \cap p_i| / (\sum_i |l_i \cap p_i| + \sum_i |l_i \subset p_i|) (micro)
    F1-score: 2 / (1/p_micro + 1/r_micro) (micro)
    where l_i is a label set of i-th document and p_i is a predicted tag set of i-th document
    """
    accuracys = []
    precisions = []
    recalls = []
    f1s = []

    l_cap_p = 0
    p_subset_l = 0
    l_subset_p = 0

    for true_idx, pred_idx in zip(true_labels_idx, pred_labels_idx):
        accuracys.append(len(true_idx & pred_idx) / len(true_idx | pred_idx))
        precisions.append(len(true_idx & pred_idx) / len(pred_idx))
        recalls.append(len(true_idx & pred_idx) / len(true_idx))
        f1s.append(2 * len(true_idx & pred_idx) / (len(true_idx) + len(pred_idx)))

        l_cap_p += len(true_idx & pred_idx)
        p_subset_l += len(pred_idx - true_idx)
        l_subset_p += len(true_idx - pred_idx)

    avg = lambda x: sum(x) / len(x)

    a, p, r, f = avg(accuracys), avg(precisions), avg(recalls), avg(f1s)
    p_micro, r_micro, f_micro = l_cap_p / (l_cap_p + p_subset_l), l_cap_p / (l_cap_p + l_subset_p), 2 * l_cap_p / (
                2 * l_cap_p + l_subset_p + p_subset_l)
    if verbose:
        print(kind)
        print('Accuracy:', a)
        print('Precision (samples):', p)
        print('Recall (samples):', r)
        print('F1-score (samples):', f)
        print('Precision (micro):', p_micro)
        print('Recall (micro):', r_micro)
        print('F1-score (micro):', f_micro)
    return a, p, r, f, p_micro, r_micro, f_micro


def evaluation_report(true_file, pred_file, data_name, output_file, report_path, everbose=False):
    print(data_name)
    assert data_name in ['AUS', 'UKD']

    true_labels = open(true_file, 'r', encoding='utf-8').readlines()
    true_labels = [true_label.strip() for true_label in true_labels]
    pred_labels = open(pred_file, 'r', encoding='utf-8').readlines()
    pred_labels = [pred_label.strip() for pred_label in pred_labels]
    print(true_labels)
    print(pred_labels)

    assert len(true_labels) == len(pred_labels), "the number of prediction label != the number of true label"

    # label to idx
    if data_name == 'AUS':
        PRIMARY_LABEL = AUS_PRIMARY_LABEL
        SECONDARY_LABEL = AUS_SECONDARY_LABEL
    if data_name == 'UKD':
        PRIMARY_LABEL = UKD_PRIMARY_LABEL
        SECONDARY_LABEL = UKD_SECONDARY_LABEL

    PRIMARY_LABEL_IDX = {k: v for v, k in enumerate(PRIMARY_LABEL)}
    SECONDARY_LABEL_IDX = {k: v for v, k in enumerate(SECONDARY_LABEL)}
    secondary_true_labels_idx = [LABEL_TO_IDX(true_label, SECONDARY_LABEL_IDX) for true_label in true_labels]
    secondary_pred_labels_idx = [LABEL_TO_IDX(pred_label, SECONDARY_LABEL_IDX) for pred_label in pred_labels]

    primary_true_labels_idx = [LABEL_TO_IDX(SECONDARY_TO_PRIMARY(true_label), PRIMARY_LABEL_IDX) for true_label in
                               true_labels]
    primary_pred_labels_idx = [LABEL_TO_IDX(SECONDARY_TO_PRIMARY(pred_label), PRIMARY_LABEL_IDX) for pred_label in
                               pred_labels]

    p_a, p_p, p_r, p_f, p_p_micro, p_r_micro, p_f_micro = multilabel_classification_evaluate(primary_true_labels_idx,
                                                                                             primary_pred_labels_idx,
                                                                                             'Primary', verbose)
    s_a, s_p, s_r, s_f, s_p_micro, s_r_micro, s_f_micro = multilabel_classification_evaluate(secondary_true_labels_idx,
                                                                                             secondary_pred_labels_idx,
                                                                                             'Secondary', verbose)

    p_label_2_prf = multilabel_for_category_evaluate(primary_true_labels_idx, primary_pred_labels_idx, PRIMARY_LABEL,
                                                     PRIMARY_LABEL_IDX, 'Primary', verbose)
    s_label_2_prf = multilabel_for_category_evaluate(secondary_true_labels_idx, secondary_pred_labels_idx,
                                                     SECONDARY_LABEL, SECONDARY_LABEL_IDX, 'Secondary', verbose)

    if output_file:
        print("Kind, accuracy\n")
        print("Primary", p_a)
        print("Secondary", s_a)
        wr = open(report_path, 'w+', encoding='utf-8')
        wr.write('kind,accuracy\n')
        wr.write('{0},{1}\n'.format('Primary', p_a))
        wr.write('{0},{1}\n'.format('Secondary', s_a))
        wr.write('\n')
        wr.write('kind,precision (samples),recall (samples),f1-score (samples)\n')
        wr.write('{0},{1},{2},{3}\n'.format('Primary', p_p, p_r, p_f))
        wr.write('{0},{1},{2},{3}\n'.format('Secondary', s_p, s_r, s_f))
        wr.write('\n')
        wr.write('kind,precision (micro),recall (micro),f1-score (micro)\n')
        wr.write('{0},{1},{2},{3}\n'.format('Primary', p_p_micro, p_r_micro, p_f_micro))
        wr.write('{0},{1},{2},{3}\n'.format('Secondary', s_p_micro, s_r_micro, s_f_micro))
        wr.write('\n')
        wr.write('Primary Categories\n')
        wr.write('class,precision,recall,f1-score,true number\n')
        for label in PRIMARY_LABEL:
            wr.write('{0},{1},{2},{3},{4}\n'.format(label, p_label_2_prf[label][0], p_label_2_prf[label][1],
                                                    p_label_2_prf[label][2], p_label_2_prf[label][3]))
        wr.write('\n')
        wr.write('Secondary Categories\n')
        wr.write('class,precision,recall,f1-score,true number\n')
        for label in SECONDARY_LABEL:
            wr.write('{0},{1},{2},{3},{4}\n'.format(label, s_label_2_prf[label][0], s_label_2_prf[label][1],
                                                    s_label_2_prf[label][2], s_label_2_prf[label][3]))
        wr.write('\n')
        wr.close()


def main():
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("true_file",
                        help="ground true (correct) labels file")
    parser.add_argument("pred_file",
                        help="predicted labels file")
    parser.add_argument("data_name",
                        help="which dataset, AUS or UKD")
    parser.add_argument("--output_file",
                        help="save the evaulation report at output file")
    parser.add_argument("--verbose", default=False,
                        help="print evaluation reports and accuracy")
    args = parser.parse_args()

    evaluation_report(args.true_file, args.pred_file, args.data_name, args.output_file, args.verbose)


if __name__ == '__main__':
    main()