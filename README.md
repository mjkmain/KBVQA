# KBVQA
Knowledge-Based Visual Question Answering

## Directory structure
- 학습을 위한 디렉토리 구조는 다음과 같습니다.

       ┣━━━ data
       ┃      ┗━━━ images
       ┃              ┗━━━ image1
       ┃              ┗━━━ ...
       ┣━━━ preprocessed_data
       ┃      ┗━━━ vqa_data.csv   
       ┗━━━ VQA_py
              ┗━━━ main.py
              ┗━━━ model.py
              ┗━━━ train.py
              ┗━━━ util.py
              ┗━━━ focal.py
              ┗━━━ vqa_dataset.py
       
## Train

- with out Transformer Layer
- Classifier에서 트랜스포머 레이어 사용하지 않고 MLP로만 학습을 진행합니다.
```bash
python main.py
```

- Using Transformer Layer
- Classifier에서 트랜스포머 레이어를 사용해서 학습을 진행합니다.
```bash
python main.py --use_transformer_layer
```

## Args
- n_epoch (type=int, required=False, default=50)
- lr (type=float, required=False, default=3e-5)
- weight_decay (type=float, required=False, default=0.001)
- batch_size (type=int, required=False, default=512)

- use_focal (action='store_true')
- use_weight (action='store_true')
- focal_gamma (type=float, required=False, default=2.0)

- use_transformer_layer (action='store_true')
- train_data (default='all', choices=['ans1', 'ans2', 'all'])
- max_token (default=512)
