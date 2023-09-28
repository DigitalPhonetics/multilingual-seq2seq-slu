#!/usr/bin/env bash

# Copyright 2022 University of Stuttgart (Pavel Denisov)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=1       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0

 . utils/parse_options.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir -p ${COMMONVOICE}
if [ -z "${COMMONVOICE}" ]; then
    log "Fill the value of 'COMMONVOICE' of db.sh"
    exit 1
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log "data preparation started"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage1: Preparing data for commonvoice"

    for lang in ar az bn cs de en es et fa "fi" fr gl hi id it ja ka kk lt lv mn mr nl pl pt ro ru sv-SE sw ta th tr uk ur vi zh-CN zh-HK zh-TW; do
      train_set=train_"$(echo "${lang}" | tr - _)"
      train_dev=dev_"$(echo "${lang}" | tr - _)"
      test_set=test_"$(echo "${lang}" | tr - _)"

      ### Task dependent. You have to make data the following preparation part by yourself.
      for part in "validated" "test" "dev"; do
          # use underscore-separated names in data directories.
          local/data_prep.pl "${COMMONVOICE}/cv-corpus-12.0-2022-12-07/${lang}" ${part} data/"$(echo "${part}_${lang}" | tr - _)"
      done

      # remove test&dev data from validated sentences
      utils/copy_data_dir.sh data/"$(echo "validated_${lang}" | tr - _)" data/${train_set}
      utils/filter_scp.pl --exclude data/${train_dev}/wav.scp data/${train_set}/wav.scp > data/${train_set}/temp_wav.scp
      utils/filter_scp.pl --exclude data/${test_set}/wav.scp data/${train_set}/temp_wav.scp > data/${train_set}/wav.scp
      utils/fix_data_dir.sh data/${train_set}
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage2: Preparing data for wenetspeech"
    local/wenetspeech_data.sh --stage 1
    mv data/train_m data/train_wns-m
    mv data/dev data/dev_wns
    rm data/{train_wns-m,dev_wns}/utt2dur
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage3: Combine data"
    utils/combine_data.sh data/train data/train_*
    utils/combine_data.sh data/dev data/dev_*

    mkdir -p data/.backup
    mv -u data/train_* data/dev_* data/.backup/
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage4: Filter data"

    mkdir -p downloads

    for subset in "train_1k" "train_1k_en" "train_7k" "dev_20"; do
      wget -c -O downloads/${subset}.lst "https://zenodo.org/record/8384623/files/${subset}.lst?download=1"
      utils/copy_data_dir.sh data/${subset%%_*} data/${subset}
      utils/filter_scp.pl downloads/${subset}.lst data/${subset%%_*}/wav.scp > data/${subset}/wav.scp
      utils/fix_data_dir.sh data/${subset}
      local/generate_utt2lid.py data/${subset}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
