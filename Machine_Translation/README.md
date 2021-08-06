## Setup

```sh
pip install --editable ./
```

## Requirement 

- PyTorch version >= 1.5.0
- Python version >= 3.6

## Preprocessing 

### IWSLT14 De-En

#### Download and prepare the data
```sh
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..
```

#### Preprocess/binarize the data
```sh
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```

### WMT17 En-De

#### Download and prepare the data
```sh
cd examples/translation/
bash prepare-wmt14en2de.sh
cd ../..
```

#### Preprocess/binarize the data
```sh
TEXT=examples/translation/wmt17_en_de
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt17_en_de --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20

```

### WMT En-Fr

#### Download and prepare the data
```sh
cd examples/translation/
bash prepare-wmt14en2fr.sh
cd ../..
```

#### Preprocess/binarize the data
```sh
TEXT=examples/translation/wmt14_en_fr
fairseq-preprocess \
    --source-lang en --target-lang fr \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt14_en_fr --thresholdtgt 0 --thresholdsrc 0 \
    --workers 60
```

## Run

### Train a De-En translation model with iwslt dataset 

```sh
mkdir -p checkpoints/transformer_iwslt_de_en
CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
--save-dir checkpoints/IWSLTDE_EN \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
```

### Train a En-De translation model with WMT dataset 

```sh
mkdir -p checkpoints/transformer_wmt_en_de
CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/wmt17_en_de \
    --arch transformer_wmt_en_de --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
--save-dir checkpoints/transformer_wmt_en_de \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
```


### Train a En-Fr translation model with WMT dataset 
```sh
mkdir -p checkpoints/transformer_wmt_en_fr

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/wmt14_en_fr  \
    --arch transformer_vaswani_wmt_en_fr_big --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
--save-dir checkpoints/transformer_wmt_en_fr \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
```

