import torch
from tqdm import tqdm
import numpy as np
from copy import deepcopy


total_train_loss = []
total_train_acc = []
total_train_f1 = []

total_valid_loss = []
total_valid_acc = []
total_valid_f1 = []

def train_fn(model, train_loader, a1_valid_loader, a2_valid_loader, criterion, optimizer, device, n_epoch):
    a1_valid_acc = 0
    a2_valid_acc = 0
    
    best_acc = 0

    for epoch in range(n_epoch):
        if (epoch+1)%5 == 0:
            optimizer.param_groups[0]['lr'] *= 0.8
             
        train_count_correct = 0
        train_total_num = 0
        a1_valid_total_num = 0
        a2_valid_total_num = 0
        a1_valid_count_correct = 0
        a2_valid_count_correct = 0
        train_f1 = 0
        
        train_loss = 0
        a1_valid_loss = 0
        a2_valid_loss = 0

        model.train()
        for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), leave=False): 
            optimizer.zero_grad()
            imgs = batch['image'].to(device)  
            q_bert_ids = batch['ids'].to(device) 
            q_bert_mask = batch['mask'].to(device) 
            
            answers = batch['answer'].to(device) 
            answers = answers.squeeze()


            outputs = model(q_bert_ids, q_bert_mask, imgs) 

            loss = criterion(outputs, answers)

            train_loss += float(loss)
            loss.backward(loss)
            optimizer.step()
            

            '''
            acc
            '''
            predicted = torch.argmax(outputs, dim=1)
            count_correct = np.count_nonzero((np.array(predicted.cpu())==np.array(answers.cpu())) == True)      
            train_count_correct += count_correct
            train_total_num += answers.size(0)
            
        train_loss /= len(train_loader)
        train_f1 /= len(train_loader)

        train_acc = train_count_correct/train_total_num

        '''
        answer1 validation
        '''
        model.eval()
        for idx, batch in tqdm(enumerate(a1_valid_loader), total=len(a1_valid_loader), leave=False):
            with torch.no_grad():
                imgs = batch['image'].to(device)
                q_bert_ids = batch['ids'].to(device)
                q_bert_mask = batch['mask'].to(device)
                answers = batch['answer'].to(device) #응답
                answers = answers.squeeze()
                outputs = model(q_bert_ids, q_bert_mask, imgs)
                
                loss = criterion(outputs, answers)

                a1_valid_loss += float(loss)
                

                predicted = torch.argmax(outputs, dim=1)
                count_correct = np.count_nonzero((np.array(predicted.cpu())==np.array(answers.cpu())) == True)      
                a1_valid_count_correct += count_correct
                a1_valid_total_num += answers.size(0)
            
        a1_valid_loss /= len(a1_valid_loader)
        a1_valid_acc = a1_valid_count_correct/a1_valid_total_num

        '''
        answer2 validation
        '''
        model.eval()
        for idx, batch in tqdm(enumerate(a2_valid_loader), total=len(a2_valid_loader), leave=False):
            with torch.no_grad():
                imgs = batch['image'].to(device)
                q_bert_ids = batch['ids'].to(device)
                q_bert_mask = batch['mask'].to(device)
                answers = batch['answer'].to(device) #응답
                answers = answers.squeeze()
                outputs = model(q_bert_ids, q_bert_mask, imgs)
                loss = criterion(outputs, answers)
                a2_valid_loss += float(loss)

                predicted = torch.argmax(outputs, dim=1)
                count_correct = np.count_nonzero((np.array(predicted.cpu())==np.array(answers.cpu())) == True)      
                a2_valid_count_correct += count_correct
                a2_valid_total_num += answers.size(0)
            
        a2_valid_loss /= len(a2_valid_loader)
        a2_valid_acc = a2_valid_count_correct/a2_valid_total_num

        valid_acc = (a1_valid_count_correct + a2_valid_count_correct) / (a1_valid_total_num + a2_valid_total_num)
        
        if valid_acc > best_acc:
            best_model = deepcopy(model)
            best_acc = valid_acc
        
        total_train_acc.append(train_acc)
        total_train_f1.append(train_f1)
        total_train_loss.append(train_loss)
        total_valid_acc.append(valid_acc)


        print(f"[{epoch+1}/{n_epoch}] LR : {optimizer.param_groups[0]['lr']:.2e} [TRAIN LOSS: {train_loss:.4f}] [TRAIN ACC: {train_acc:.4f}] [VALID ACC: {valid_acc:.4f}] [a1 VALID ACC: {a1_valid_acc:.4f}] [a2 VALID ACC: {a2_valid_acc:.4f}]")
    last_model = deepcopy(model)
    best_path = f'./ft_models/best_acc{best_acc*100:.2f}.pt'
    torch.save(best_model, best_path)
    # torch.save(last_model, last_path)
    return best_path, best_acc