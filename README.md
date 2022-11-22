# KBVQA
Knowledge-Based Visual Question Answering

## Directory structure

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
```bash
python main.py
```

- Using Transformer Layer
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
