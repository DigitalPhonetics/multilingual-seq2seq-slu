# [Leveraging Multilingual Self-Supervised Pretrained Models for Sequence-to-Sequence End-to-End Spoken Language Understanding](https://arxiv.org/abs/2310.06103)

## NLU

### Training

1. Install `"transformers<=4.19.0"`.
2. Go to dataset directory: `cd nlu/slurp/`.
3. Prepare data: `./prepare_data.py /path/to/slurp data`.
4. Train a model: `./finetune.sh` (tested on A6000 GPU).
5. Run evaluation: `./evaluate.py data/test.target output/test_generations.txt`.

Alternatively, you can use the pretrained models hosted on Hugging Face Hub.

### Pretrained Models

| Dataset | Directory | Pretrained model |
|--|--|--|
|SLURP  | `nlu/slurp` | [akreal/mbart-large-50-finetuned-slurp](https://huggingface.co/akreal/mbart-large-50-finetuned-slurp)|
|SLUE |`nlu/slue` | [akreal/mbart-large-50-finetuned-slue](https://huggingface.co/akreal/mbart-large-50-finetuned-slue) |
|CATSLU | `nlu/catslu`| [akreal/mbart-large-50-finetuned-catslu](https://huggingface.co/akreal/mbart-large-50-finetuned-catslu) |
| MEDIA | `nlu/media`| [akreal/mbart-large-50-finetuned-media](https://huggingface.co/akreal/mbart-large-50-finetuned-media) |
| PortMEDIA-Dom| `nlu/portmedia_dom`| [akreal/mbart-large-50-finetuned-portmedia-dom](https://huggingface.co/akreal/mbart-large-50-finetuned-portmedia-dom) |
| PortMEDIA-Lang| `nlu/portmedia_lang`| [akreal/mbart-large-50-finetuned-portmedia-lang](https://huggingface.co/akreal/mbart-large-50-finetuned-portmedia-lang) |


## SLU

### Training

1. Install the regular ESPnet version.
2. Copy the model configuration file from this repository:
`cp slu/slurp/train_asr_conformer_xlsr_mbart.yaml /path/to/espnet/egs2/slurp_entity/asr1/conf/`
3. Run the recipe:
`./run.sh --asr_config conf/train_asr_conformer_xlsr_mbart.yaml`.
4. If you want to use pretrained Adaptor, download it from the link in the next section and run the recipe with it:
`./run.sh --asr_config conf/train_asr_conformer_xlsr_mbart.yaml --pretrained_model downloads/conformer08x08h_d1024_xlsr_ts_lr5e-5_attcela_7kh_ave.pth:::decoder --asr_tag postdec-aed_7kh`.

### Pretrained Models

The following are SLU models trained with the Adaptor that is pretrained on 7k hours with PostDec-AED loss.
| Dataset | Recipe | Pretrained model |
|--|--|--|
|SLURP|`slurp_entity`|[Link](https://doi.org/10.5281/zenodo.8380247)|
|SLUE|`slue-voxpopuli`|[Link](https://doi.org/10.5281/zenodo.8380193)|
|CATSLU|`catslu_entity`|[Link](https://doi.org/10.5281/zenodo.8379839)|
|MEDIA|`media`|[Link](https://doi.org/10.5281/zenodo.8379989)|
|PortMEDIA-Dom|`portmedia_dom`|[Link](https://doi.org/10.5281/zenodo.8379924)|
|PortMEDIA-Lang|`portmedia_lang`|[Link](https://doi.org/10.5281/zenodo.8379957)|

Cross-lingual PortMEDIA-Lang SLU model finetuned from the MEDIA SLU model: [Link](https://doi.org/10.5281/zenodo.8379982).

## Adaptor

### Training

1. Install the custom version of ESPnet:
`git clone --branch v.202207 --depth 1 git@github.com:espnet/espnet.git /path/to/espnet-adaptor-pretrain`
2. Copy the modifications:
`rsync -avh adaptor/espnet/ /path/to/espnet-adaptor-pretrain/`
3. Follow ESPnet installation instructions.
4. Run the recipe:
`cd /path/to/espnet-adaptor-pretrain/egs2/commonvoice/asr1; ./run.sh`

### Pretrained Models

| Loss | Configuration | Pretrained model |
|--|--|--|
| PreEnc MC | `conf/train_adaptor_conformer_preenc-mc.yaml` | [Link](https://zenodo.org/record/8361271/files/conformer08x08h_d1024_xlsr_ts_lr5e-5_pm.pth?download=1) |
| PreEnc CTC | `conf/train_adaptor_conformer_preenc-ctc.yaml` | [Link](https://zenodo.org/record/8361271/files/conformer08x08h_d1024_xlsr_ts_lr5e-5_ctc.pth?download=1) |
| PostEnc MC | `conf/train_adaptor_conformer_postenc-mc.yaml` | [Link](https://zenodo.org/record/8361271/files/conformer08x08h_d1024_xlsr_ts_lr5e-5_fpe.pth?download=1) |
| PostDec MC | `conf/train_adaptor_conformer_postdec-mc.yaml` | [Link](https://zenodo.org/record/8361271/files/conformer08x08h_d1024_xlsr_ts_lr5e-5_pdla.pth?download=1) |
| PostDec AED | `conf/train_adaptor_conformer_postdec-aed.yaml` | [Link](https://zenodo.org/record/8361271/files/conformer08x08h_d1024_xlsr_ts_lr5e-5_attcela.pth?download=1) |
| PreEnc CTC + PostDec AED | `conf/train_adaptor_conformer_preenc-ctc_postdec-aed.yaml` | [Link](https://zenodo.org/record/8361271/files/conformer08x08h_d1024_xlsr_ts_lr5e-5_ctc-attcela.pth?download=1) |
| PreEnc CTC + PostDec AED + PostEnc MC | `conf/train_adaptor_conformer_preenc-ctc_postenc-mc_postdec-aed.yaml` | [Link](https://zenodo.org/record/8361271/files/conformer08x08h_d1024_xlsr_ts_lr5e-5_mnsal2-ctc-attce.pth?download=1) |
| PostEnc MC (1K hours English data) |  | [Link](https://zenodo.org/record/8361271/files/conformer08x08h_d1024_xlsr_ts_lr5e-5_english_fpe.pth?download=1) |
| PostDec AED (7K hours data) |  | [Link](https://zenodo.org/record/8386716/files/conformer08x08h_d1024_xlsr_ts_lr5e-5_attcela_7kh_ave.pth?download=1) |

## Citation
```
@article{denisov2023leveraging,
  title={Leveraging Multilingual Self-Supervised Pretrained Models for Sequence-to-Sequence End-to-End Spoken Language Understanding},
  author={Denisov, Pavel and Vu, Ngoc Thang},
  journal={arXiv preprint arXiv:2310.06103},
  year={2023}
}
```
