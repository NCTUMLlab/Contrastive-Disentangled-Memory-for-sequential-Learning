## Python Requirement

```sh
torch
chainer==6.0.0 cupy==6.0.0    
torchaudio                    
torch_optimizer    
```

## Setup

```sh
cd tools
make
```

## Download Aishell dadaset and Preprocessing

```sh
cd egs/aishell/asr
./run.sh --stage -1  
```

## Train

```sh
cd egs/aishell/asr
./run.sh --stage 4 
```

## Test

```sh
cd egs/aishell/asr
./run.sh --stage 5
```
