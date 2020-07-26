#!/usr/bin/env bash

ONMT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#======= USER INPUT CONFIGURATION ======
if [ $# -eq 0 ]; then
    echo "No arguments supplied."
    exit
fi
USE_BSUB=$1
NGPU=$2
JOBTYPE=$3
INHOUSE=$4

#======= EXPERIMENT INPUT ======
BEAM=10
TEST_BATCH_SIZE=384
N_MODELS=1

BATCH_SIZE=4096
VALID_STEP=1000
REPORT_EVERY=1000

# TRAIN_FROM_STEP=3000

HYP_OPTS="\
-layers 6 -rnn_size 256 -word_vec_size 256 -heads 8 -transformer_ff 2048 \
-share_embeddings -dropout 0.3 -max_relative_positions 4"
# HYP2_OPTS="\
# -layers 6 -rnn_size 512 -word_vec_size 512 -heads 8 -transformer_ff 2048 \
# -tied -share_embeddings -dropout 0.1"
HYP3_OPTS="$HYP2_OPTS -label_smoothing 0.1"
HYP4_OPTS="$HYP2_OPTS -share_decoder_embeddings"

OPT1_OPTS="\
-max_grad_norm 0 -optim adam -adam_beta1 0.9 -adam_beta2 0.998 \
-decay_method noam -warmup_steps 8000 -learning_rate 2 \
-param_init 0 -param_init_glorot"

# update these variables

# ====== BASE HYP2 ====== #

NAME="HYP_OPTS_l3"
USER_ARGS="-method oneway_latent -num_experts 3 $HYP_OPTS $OPT1_OPTS"

#======= EXPERIMENT SETUP ======

DATA="$ONMT/data/retro_transformer_data/USPTO-50k_no_rxn"
DATA_PREFIX="$ONMT/data/USPTO-50k_processed"

TRAIN_SRC=$DATA/src-train.txt
VALID_SRC=$DATA/src-val.txt
TRAIN_TGT=$DATA/tgt-train.txt
VALID_TGT=$DATA/tgt-val.txt

# TEST_SRC=$DATA/src-test.10.txt
# TEST_TGT=$DATA/tgt-test.10.txt

TEST_SRC=$DATA/src-test.txt
TEST_TGT=$DATA/tgt-test.txt

if [[ $JOBTYPE == 'demo' ]]; then
    BATCH_SIZE=128
    VALID_STEP=10
    REPORT_EVERY=10
    NAME="test_$NAME"
    if [[ -z $STEP ]]; then
        JOBTYPE='train'
    else
        JOBTYPE='test'
    fi
fi

#====== EXPERIMENT BEGIN ======

# Check if input exists
for f in $TRAIN_SRC $TRAIN_TGT $VALID_SRC $VALID_TGT $TEST_SRC $TEST_TGT; do
    if [[ ! -f "$f" ]]; then
        echo "Input File $f doesnt exist. Please fix the paths"
        exit 1
    fi
done

echo "Step 1: Preprocess"
files=$(shopt -s nullglob dotglob; echo $DATA_PREFIX*)
if (( ${#files} ))
then
    echo "Preprocessed *.pt files already exist. Pass."
    echo "$files"
else
python $ONMT/preprocess.py \
    -train_src $TRAIN_SRC \
    -train_tgt $TRAIN_TGT \
    -valid_src $VALID_SRC \
    -valid_tgt $VALID_TGT \
    -save_data $DATA_PREFIX \
    -share_vocab -overwrite
fi
echo ""

OUT="onmt-runs/$NAME"
if [[ $JOBTYPE == 'train' ]]; then
    echo "Step 2: Train"
    echo "Output dir = $OUT"
    [ -d $OUT ] || mkdir -p $OUT
    [ -d $OUT/models ] || mkdir $OUT/models
    [ -d $OUT/test ] || mkdir -p  $OUT/test

    GPU_OPTS=""
    if [ $NGPU -eq 1 ]; then
        GPU_OPTS="-gpu_ranks 0 -world_size 1 -accum_count 4"
    elif [ $NGPU -eq 2 ]; then
        GPU_OPTS="-gpu_ranks 0 1 -world_size 2 -accum_count 2"
    elif [ $NGPU -eq 3 ]; then
        GPU_OPTS="-gpu_ranks 0 1 2 -world_size 3 -accum_count 1"
    elif [ $NGPU -eq 4 ]; then
        GPU_OPTS="-gpu_ranks 0 1 2 3 -world_size 4 -accum_count 1"
    fi

    TRAIN_FROM_OPTS=""
    if [[ ! -z "$TRAIN_FROM_STEP" ]]; then
        TRAIN_FROM_OPTS="-train_from \
            $OUT/models/${NAME}_step_$TRAIN_FROM_STEP.pt"
    fi

    CMD="python $ONMT/train.py -data $DATA_PREFIX \
        -save_model $OUT/models/$NAME -save_dir $OUT -train_steps 500000 \
        -save_checkpoint_steps $VALID_STEP -keep_checkpoint 50 \
        -valid_step $VALID_STEP -report_every $REPORT_EVERY \
        -batch_size $BATCH_SIZE -batch_type tokens -normalization tokens \
        -encoder_type transformer -decoder_type transformer \
        -global_attention general -global_attention_function softmax \
        -self_attn_type scaled-dot -position_encoding \
        $USER_ARGS $GPU_OPTS $TRAIN_FROM_OPTS \
        2>&1 | tee -a $OUT/train_$NAME.log"
    echo "$CMD"
    eval "$CMD"
fi


STEP=3000
NUM_EXPERTS=3

TRANSLATE_OUT=$OUT/test/step_${STEP}

MODEL=$OUT/models/${NAME}_step_${STEP}.pt
if [[ $N_MODELS -gt 1 ]]; then
    echo "Step 3a: Average Models"
    models="$MODEL"
    MODEL="$OUT/models/${NAME}_step_${STEP}_avg${N_MODELS}.pt"
    TRANSLATE_OUT=$OUT/test/step_${STEP}_avg${N_MODELS}

    if [[ ! -f "$MODEL" ]]; then
        step="$STEP"
        for i in `seq 1 9`; do
        step="$(( $STEP - $VALID_STEP * i ))"
            models="$models $OUT/models/${NAME}_step_${step}.pt"
        done

        CMD="python average_models.py -models $models -output $MODEL"
        echo "$CMD"
        eval "$CMD"
    else
        echo "Averaged pt file already exists. Pass."
        echo "(Check: $MODEL)"
    fi
    echo ""
fi


if [[ ( $JOBTYPE == 'test' ) && ( ! -z $STEP ) ]]; then
    GPU_OPTS=""
    if [ $NGPU -gt 0 ]; then
        GPU_OPTS="-gpu 0"
    fi

    [ -d $TRANSLATE_OUT ] || mkdir -p $TRANSLATE_OUT

    echo "Step 3b: Translate Test"
    echo "Output dir = $TRANSLATE_OUT"
    echo ""
    python $ONMT/translate.py -model $MODEL \
        -src $TEST_SRC -tgt $TEST_TGT \
        -output $TRANSLATE_OUT \
        -beam_size $BEAM -n_best $BEAM \
        -num_experts $NUM_EXPERTS \
        -batch_size $TEST_BATCH_SIZE \
        -replace_unk $GPU_OPTS \
        2>&1 | tee -a $TRANSLATE_OUT/test_$NAME.log
    echo "Check Output dir = $TRANSLATE_OUT"
fi

if [[ ( $JOBTYPE == 'test' || $JOBTYPE == 'eval' ) && ( ! -z $STEP ) ]]; then
    echo "Step 3c: Evaluate Test"
    echo "Output dir = $TRANSLATE_OUT"
    echo ""
    echo "Evaluate FWD"
    python $ONMT/evaluate.py -beam_size $BEAM \
        -output $TRANSLATE_OUT/fwd_out_can.txt -target $TEST_TGT \
        -log_file $TRANSLATE_OUT/fwd_out_can.txt.score
    echo ""
    echo "Evaluate FWD_CYCLE"
    python $ONMT/evaluate.py -beam_size $BEAM \
        -output $TRANSLATE_OUT/pred.txt -target $TEST_TGT \
        -log_file $TRANSLATE_OUT/pred.txt.score
    echo ""
    echo "Check Output dir = $TRANSLATE_OUT"
fi

#===== EXPERIMENT END ======
