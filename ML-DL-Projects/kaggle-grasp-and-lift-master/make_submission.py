#!/usr/bin/env python

from os.path import join


def make_submission():
    subjects = range(1, 13)
    submission_file = join('data',
                           'submissions',
                           'convnet_fixed.csv')
    header = ['id', 'HandStart', 'FirstDigitTouch', 'BothStartLoadPhase',
              'LiftOff', 'Replace', 'BothReleased']
    print('generating submission file: %s' % (submission_file))
    with open(submission_file, 'w') as ofile:
        ofile.write('%s\n' % ','.join(header))
        for subj_id in subjects:
            pred_file = join('data',
                             'predictions',
                             'subj%d_fixed.csv' % (
                                 subj_id))
            print('  reading probabilities from %s' % (pred_file))
            with open(pred_file, 'r') as ifile:
                for line in ifile:
                    ofile.write(line)


def main():
    make_submission()

if __name__ == '__main__':
    main()
