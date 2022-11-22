import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torchvision.transforms as transforms
from vqa_dataset import VQADataset
from torch.utils.data import DataLoader

def get_args():
    parser = argparse.ArgumentParser()

    # Train Config
    parser.add_argument('--n_epoch', type=int, required=False, default=50)
    parser.add_argument('--lr', type=float, required=False, default=3e-5)
    parser.add_argument('--weight_decay', type=float, required=False, default=0.001)
    parser.add_argument('--batch_size', type=int, required=False, default=512)

    # Loss Config
    parser.add_argument('--focal_gamma', type=float, required=False, default=2.0)
    parser.add_argument('--use_weight', action='store_true')
    parser.add_argument('--use_focal', action='store_true')


    # Model Config
    parser.add_argument('--use_transformer_layer', action='store_true', default=False)

    # Dataset Config
    parser.add_argument('--train_data', default='all', choices=['ans1', 'ans2', 'all'])
    
    # Tokenizer Config
    parser.add_argument('--max_token', type=int, required=False, default=50)
    
    config = parser.parse_args()
    return config

def get_answerlist():
    data = pd.read_csv("../preprocessed_data/vqa_data.csv")
    data = data[['img_path', 'question', 'answer']]
    data = data.dropna()
    answer_list = data['answer'].value_counts().reset_index()
    answer_list.columns=['answer', 'count']
    answer_list['weight'] = 1 - answer_list['count']/answer_list['count'].sum()

    return answer_list



def get_dataloader(config):

    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
    answer_list = get_answerlist()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    data = pd.read_csv("../preprocessed_data/vqa_data.csv")
    data = data[['img_path', 'question', 'answer']]

    even = range(0, len(data), 2)
    odd = range(1, len(data), 2)

    ans1_data = data.loc[even]
    ans1_data = ans1_data.dropna()

    ans2_data = data.loc[odd]
    ans2_data = ans2_data.dropna()

    ans1_train_data, ans1_valid_data = train_test_split(ans1_data, test_size=0.2)
    ans2_train_data, ans2_valid_data = train_test_split(ans2_data, test_size=0.2)


    ans1_valid_data = ans1_valid_data.reset_index(drop=True)
    ans2_valid_data = ans2_valid_data.reset_index(drop=True)

    data = data.dropna()
    
    
    '''
    get train dataset
    '''
    if config.train_data == 'ans1':
        train_data = ans1_train_data.reset_index(drop=True)

    if config.train_data == 'ans2':
        train_data = ans2_train_data.reset_index(drop=True)

    if config.train_data == 'all':
        train_data = pd.concat([ans1_train_data, ans2_train_data]).reset_index(drop=True)



    train_dataset = VQADataset(tokenizer, train_data, answer_list, config.max_token, transform) 
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=10, shuffle=True, pin_memory=True)

    a1_valid_dataset = VQADataset(tokenizer, ans1_valid_data, answer_list, config.max_token, transform) 
    a1_valid_loader = DataLoader(dataset=a1_valid_dataset, batch_size=config.batch_size, num_workers=10, shuffle=True, pin_memory=True)

    a2_valid_dataset = VQADataset(tokenizer, ans2_valid_data, answer_list, config.max_token, transform) 
    a2_valid_loader = DataLoader(dataset=a2_valid_dataset, batch_size=config.batch_size, num_workers=10, shuffle=True, pin_memory=True)

    return train_loader, a1_valid_loader, a2_valid_loader