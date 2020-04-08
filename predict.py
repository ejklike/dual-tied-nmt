import shlex
import torch

import onmt.opts
from onmt.utils.parse import ArgumentParser
from onmt.translate.translator import build_translator

from data_utils import canonicalize_smiles, smi_tokenizer, disable_rdkit_logging


def get_transformer_result(smiles_list_to_translate, modelpath, 
                           beam_size=10, n_best=10, max_length=300, 
                           verbose=False):
    assert n_best <= beam_size
    
    disable_rdkit_logging()

    print("User Inputs: %d" % len(smiles_list_to_translate))
    
    texts = [smi_tokenizer(canonicalize_smiles(smiles)) 
             for smiles in smiles_list_to_translate]

    empty_indices = [i for i, x in enumerate(texts) if x == ""]
    texts_to_translate = [x for x in texts if x != ""]

    print("Translation Inputs: %d" % len(texts_to_translate))

    scores = []
    predictions = []
    if len(texts_to_translate) > 0:
        # args
        args_string = ("-model %s -beam_size %s -n_best %s -save_output 0 "
                       "-max_length %s %s -replace_unk -gpu 0 -src 0 "
                       % (modelpath, beam_size, n_best, max_length, 
                          '-verbose' if verbose else ''))

        # parse args
        parser = ArgumentParser(description='translate.py')
        onmt.opts.translate_opts(parser)
        opt = parser.parse_args(shlex.split(args_string))

        # tarnslation
        translator = build_translator(opt, report_score=False)
        scores, predictions = translator.translate(
            texts_to_translate,
            batch_size=len(texts_to_translate))

    # NOTE: translator returns lists of `n_best` list
    def flatten_list(_list): return sum(_list, [])
    tiled_texts = [t for t in texts_to_translate
                    for _ in range(n_best)]
    results = flatten_list(predictions)

    def maybe_item(x): return x.item() if type(x) is torch.Tensor else x
    scores = [maybe_item(score_tensor)
              for score_tensor in flatten_list(scores)]

    # build back results with empty texts
    for i in empty_indices:
        j = i * n_best
        results = (results[:j] +
                    [("", None)] * n_best + results[j:])
        scores = scores[:j] + [0] * n_best + scores[j:]

    results = [items.replace(' ', '') for items in results]
    print("Translation Results: %d\n" % len(results))
    print()

    return results, scores


if __name__ == '__main__':
    beam_size = 10
    n_best = 10
    max_length = 300

    smiles_list_to_translate = [
        'c1(-c2ccccc2)ccccc1',
        'COC(=O)CCC(=O)c1ccc(O)cc1O', 
        'COC(=O)CCC(=O)c1ccc(OC2CCCCO2)cc1OC']
    modelpath = '/home/sr6/ejej.kim/dev/retrosynthesis/onmt1.0/experiments/2020_03_4rxns/model/2020_03_4rxns_nlatent_1_step_500000.pt'
    modelpath = '/home/sr6/ejej.kim/dev/retrosynthesis/onmt1.0/experiments/2020_03_4rxns/checkpoints/nlatent_1/2020_03_4rxns_nlatent_1_step_460000.pt'
    results, scores = get_transformer_result(smiles_list_to_translate, 
                                             modelpath, 
                                             beam_size=beam_size, 
                                             n_best=n_best, 
                                             max_length=max_length, 
                                             verbose=False)

    for input_idx, input_smi in enumerate(smiles_list_to_translate):
        print('Input #%d / %d' % (input_idx, len(smiles_list_to_translate)))
        print(input_smi)
        for beam_idx in range(beam_size):
            i = input_idx * beam_size + beam_idx
            print('Beam %d (score=%.4f)\t: %s' 
                  % (beam_idx, scores[i], results[i]))
        print()
