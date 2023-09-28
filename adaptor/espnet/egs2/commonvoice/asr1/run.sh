#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_7k" # or train_1k or train_1k_en
valid_set="dev_20"

asr_config=conf/train_adaptor_conformer_postdec-aed.yaml

./asr.sh \
    --audio_format flac.ark \
    --ngpu 4 \
    --use_lm false \
    --use_lid true \
    --token_type hugging_face \
    --hugging_face_model_name_or_path facebook/mbart-large-50-many-to-many-mmt \
    --max_wav_duration 30 \
    --asr_config "${asr_config}" \
    --feats-normalize utterance_mvn \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${valid_set}" "$@"
