#!/bin/bash

python ../finetune.py \
	--model_name_or_path=facebook/mbart-large-50-many-to-many-mmt \
	--data_dir=data \
	--output_dir=output \
	--max_source_length=128 \
	--max_target_length=128 \
	--val_max_target_length=128 \
	--test_max_target_length=128 \
	--do_train --do_predict \
	--num_train_epochs=30 \
	--auto_lr_find --auto_scale_batch_size \
	--val_metric=loss --freeze_encoder --freeze_embeds \
	--n_val=-1 --n_train=-1 --n_test=-1 \
	--tgt_lang it_IT --task translation --src_lang it_IT \
	--gpus=1
