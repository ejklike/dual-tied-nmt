#!/usr/bin/env bash

ONMT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ ! -z $1 ]]; then
    STEP=$1
fi

#======= EXPERIMENT INPUT ======
GPUARG="" # default
GPUARG="0"

BEAM=10

BATCH_SIZE=4096
VALID_STEP=10000
REPORT_EVERY=1000

# BATCH_SIZE=128
# VALID_STEP=10
# REPORT_EVERY=10

# update these variables
NAME="l3_tied"
USER_ARGS="-num_experts 3 -tied"

NAME="l5_tied"
USER_ARGS="-num_experts 5 -tied"

#======= EXPERIMENT SETUP ======

OUT="onmt-runs/$NAME"

DATA="$ONMT/data/retro_transformer_data/USPTO-50k_no_rxn"
TRAIN_SRC=$DATA/src-train.txt
VALID_SRC=$DATA/src-val.txt
TRAIN_TGT=$DATA/tgt-train.txt
VALID_TGT=$DATA/tgt-val.txt

# TEST_SRC=$DATA/src-test.10.txt
# TEST_TGT=$DATA/tgt-test.10.txt

TEST_SRC=$DATA/src-test.txt
TEST_TGT=$DATA/tgt-test.txt

DATA_PREFIX=data/USPTO-50k_processed

#====== EXPERIMENT BEGIN ======

# Check if input exists
for f in $TRAIN_SRC $TRAIN_TGT $VALID_SRC $VALID_TGT $TEST_SRC $TEST_TGT; do
    if [[ ! -f "$f" ]]; then
        echo "Input File $f doesnt exist. Please fix the paths"
        exit 1
    fi
done

if [[ -z $STEP ]]; then
    echo "Output dir = $OUT"
    [ -d $OUT ] || mkdir -p $OUT
    [ -d $OUT/models ] || mkdir $OUT/models
    [ -d $OUT/test ] || mkdir -p  $OUT/test

    # echo "Step 1: Preprocess"
    # python $ONMT/preprocess.py \
    #     -train_src $TRAIN_SRC \
    #     -train_tgt $TRAIN_TGT \
    #     -valid_src $VALID_SRC \
    #     -valid_tgt $VALID_TGT \
    #     -save_data $DATA_PREFIX \
    #     -share_vocab -overwrite

    echo "Step 2: Train"
    GPU_OPTS=""
    # if [[ ! -z $GPUARG ]]; then
    #     GPU_OPTS="-world_size 1 -gpu_ranks 0 -accum_count 4" # $GPUARG"
    #     GPU_OPTS="-world_size 1 -gpu_ranks 0 -accum_count 4" # $GPUARG"
    # fi
    GPU_OPTS="-world_size 2 -gpu_ranks 0 1 -accum_count 2" # $GPUARG"
    # GPU_OPTS="-world_size 1 -gpu_ranks 0 -accum_count 4" # $GPUARG"
    CMD="python $ONMT/train.py -data $DATA_PREFIX \
        -save_model $OUT/models/$NAME $GPU_OPTS -train_steps 500000 \
        -save_checkpoint_steps 10000 -keep_checkpoint 50 \
        -valid_step $VALID_STEP -report_every $REPORT_EVERY \
        -batch_size $BATCH_SIZE -batch_type tokens -normalization tokens \
        -max_grad_norm 0 -optim adam -adam_beta1 0.9 -adam_beta2 0.998 \
        -decay_method noam -warmup_steps 8000 \
        -param_init 0  -param_init_glorot \
        -learning_rate 2 -label_smoothing 0.0 \
        -layers 4 -rnn_size 256 -word_vec_size 256 \
        -heads 8 -transformer_ff 2048 \
        -encoder_type transformer -decoder_type transformer \
        -dropout 0.1 -position_encoding -share_embeddings \
        -global_attention general -global_attention_function softmax -self_attn_type scaled-dot $USER_ARGS \
        2>&1 | tee -a $OUT/train_$NAME.log"
    echo "Training command :: $CMD"
    eval "$CMD"
fi

if [[ ! -z $STEP ]]; then
    GPU_OPTS=""
    if [ ! -z $GPUARG ]; then
        GPU_OPTS="-gpu 0"
    fi
    model=$OUT/models/${NAME}_step_${STEP}.pt
    TRANSLATE_OUT=$OUT/test/step_${STEP}
    echo "Output dir = $TRANSLATE_OUT"
    [ -d $TRANSLATE_OUT ] || mkdir -p $TRANSLATE_OUT
    
    echo "Step 3a: Translate Test"
    python $ONMT/translate.py -model $model \
        -src $TEST_SRC -tgt $TEST_TGT \
        -output $TRANSLATE_OUT \
        -beam_size $BEAM -n_best $BEAM \
        -batch_size 128 \
        -replace_unk $GPU_OPTS \
        2>&1 | tee -a $OUT/test/test_$NAME.log

    echo "Step 3b: Evaluate Test"
    echo "Evaluate FWD"
    python $ONMT/evaluate.py -beam_size $BEAM \
    -output $TRANSLATE_OUT/pred.txt -target $TEST_TGT \
    -log_file $TRANSLATE_OUT/pred.txt.score
    echo "Evaluate FWD_CYCLE"
    python $ONMT/evaluate.py -beam_size $BEAM \
    -output $TRANSLATE_OUT/pred_cycle.txt -target $TEST_TGT \
    -log_file $TRANSLATE_OUT/pred_cycle.txt.score
fi
#-verbose 

#===== EXPERIMENT END ======
