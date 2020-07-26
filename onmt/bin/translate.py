#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import codecs

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser

from data_utils import (canonicalize_translation_output, 
                        tile_lines_n_times,
                        linecount,
                        combine_translation)


def translate(opt):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    translator = build_translator(opt, logger=logger,
                                  report_score=True, 
                                  log_score=True)

    desired_output_length = linecount(opt.src) * opt.beam_size

    logger.info("=== FWD TRANSLATION: RETROSYNTHESIS ===")
    src_path = opt.src
    tgt_path = None #opt.tgt
    out_path = opt.output + '/fwd_out.txt'

    if linecount(out_path) == desired_output_length:
        logger.info("Already translated. Pass.")
    else:
        # data preparation
        src_shards = split_corpus(src_path, opt.shard_size)
        tgt_shards = split_corpus(tgt_path, opt.shard_size)
        shard_pairs = zip(src_shards, tgt_shards)
        # translate
        translator.out_file = codecs.open(out_path, 'w+', 'utf-8')
        for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
            logger.info("Translating shard %d." % i)
            translator.translate(
                direction='x2y',
                src=src_shard,
                tgt=tgt_shard,
                src_dir=opt.src_dir,
                batch_size=opt.batch_size,
                batch_type=opt.batch_type,
                attn_debug=opt.attn_debug,
                align_debug=opt.align_debug
                )
    # canonicalize
    out_can_path = opt.output + '/fwd_out_can.txt'
    canonicalize_translation_output(out_path, out_can_path, aware_latent=True)


    logger.info("=== BWD TRANSLATION: REACTION PREDICTION ===")
    translator.beam_size = 1
    translator.n_best = 1

    # tiled src file
    opt.tiled_src = opt.output + '/tiled_src.txt'
    tile_lines_n_times(opt.src, opt.tiled_src, n=opt.beam_size)

    src_path = opt.output + '/fwd_out_can.txt'
    tgt_path = opt.tiled_src
    out_path = opt.output + '/bwd_out.txt'
    
    if linecount(out_path) == desired_output_length:
        logger.info("Already translated. Pass.")
    else:
        # data preparation
        src_shards = split_corpus(src_path, opt.shard_size)
        tgt_shards = split_corpus(tgt_path, opt.shard_size)
        shard_pairs = zip(src_shards, tgt_shards)
        # translate
        translator.out_file = codecs.open(out_path, 'w+', 'utf-8')
        for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
            logger.info("Translating shard %d." % i)
            translator.translate(
                direction='y2x',
                src=src_shard,
                tgt=tgt_shard,
                src_dir=opt.src_dir,
                batch_size=opt.batch_size,
                batch_type=opt.batch_type,
                attn_debug=opt.attn_debug,
                align_debug=opt.align_debug
                )

    # combine K outputs
    logger.info("=== Combine TWO-WAY score outputs ===")
    input_path = opt.output + '/fwd_out.txt'
    bwd_input_path = opt.output + '/bwd_out.txt'
    output_path = opt.output + '/pred.txt'
    combine_translation(input_path, bwd_input_path, 
                        opt.num_experts, opt.beam_size, 
                        output_path=output_path)

def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    translate(opt)


if __name__ == "__main__":
    main()
