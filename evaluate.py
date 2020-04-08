import argparse
from rdkit import Chem
import pandas as pd

from data_utils import canonicalize_smiles, disable_rdkit_log

disable_rdkit_log()


def match_smiles_set(otuput_set, target_set):
    if len(otuput_set) != len(target_set):
        return False

    for smiles in target_set:
        if smiles not in otuput_set:
            return False
    return True


def get_rank(row, base, max_rank):
    target_smiset = row['target'].split('.') # already canonicalized
    for i in range(1, max_rank+1):
        output_smiset = row['{}{}'.format(base, i)].split('.')
        if match_smiles_set(output_smiset, target_smiset):
            return i
    return 0


def main(opt):
    with open(opt.targets, 'r') as f:
        targets = [''.join(line.strip().split(' ')) for line in f.readlines()]

    targets = targets[:]
    outputs = [[] for i in range(opt.beam_size)]

    test_df = pd.DataFrame(targets)
    test_df.columns = ['target']
    total = len(test_df)

    with open(opt.outputs, 'r') as f:
        for i, line in enumerate(f.readlines()):
            outputs[i % opt.beam_size].append(''.join(line.strip().split(' ')))
    
    for i, preds in enumerate(outputs):
        test_df['pred_{}'.format(i + 1)] = preds
        test_df['pred_can_{}'.format(i + 1)] = \
            test_df['pred_{}'.format(i + 1)].apply(
                lambda x: canonicalize_smiles(x))

    test_df['rank'] = test_df.apply(
        lambda row: get_rank(row, 'pred_can_', opt.beam_size), axis=1)
    correct = 0
    invalid_smiles = 0

    logs = []
    line_accuracy = ''
    line_invalid = ''
    for i in range(1, opt.beam_size+1):
        correct += (test_df['rank'] == i).sum()
        invalid_smiles += (test_df['pred_can_{}'.format(i)] == '').sum()

        accuracy = correct/total * 100
        invalid_ratio = invalid_smiles/(total*i) * 100
        print('Top-{}: {:.1f}% || Invalid SMILES {:.2f}%'
                .format(i, accuracy, invalid_ratio))
        logs.append('%d,%g,%g\n' % (i, accuracy, invalid_ratio))
        line_accuracy += '%g,' % accuracy
        line_invalid += '%g,' % invalid_ratio

    if opt.log_file:
        with open(opt.log_file, 'w') as f:
            f.writelines(logs)
    with open('results.csv', 'a+') as f:
        line = opt.outputs + ',' + line_accuracy + line_invalid + '\n'
        f.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='evaluate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-beam_size', type=int, default=10,
                       help='Beam size')
    parser.add_argument('-outputs', type=str, required=True,
                       help="Path to file containing the translation outputs")
    parser.add_argument('-targets', type=str, required=True,
                       help="Path to file containing targets")
    parser.add_argument('-log_file', type=str, default=None,
                       help="Path to file containing scoring results")

    opt = parser.parse_args()
    main(opt)
