from load_data import get_cifar100_train

from AlexNet import AlexNet
from tqdm import tqdm 
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataloader, valid_dataloader = get_cifar100_train(
    batch_size = 40,
    val_split = 0.2
)


alex_net = AlexNet(num_classes = 100, dropout = 0.5)
alex_net = alex_net.to(device) 

EPOCHS = 120

optimizer = torch.optim.SGD(
    alex_net.parameters(),
    lr=0.01,       
    momentum=0.9,   
    weight_decay=5e-4  
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5) 

criterion = torch.nn.CrossEntropyLoss()


for ep in range(EPOCHS):
    
    train_progress_bar = tqdm(train_dataloader, desc = f"Epoch #{ep + 1} (Training):", unit = "batch(es)", leave = False)
    
    
    total_train_loss = 0
    count = 0
    
    correct_train = 0
    total_train = 0
    
    
    for src, tgt in train_progress_bar:
        alex_net.train()
        
        src = src.to(device)
        tgt = tgt.to(device) 
        
        outputs = alex_net(src)
        
        loss = criterion(outputs, tgt)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_progress_bar.set_postfix(loss = loss.item())
    
        total_train_loss += loss.item()
        count += 1
        
        _, predicted = outputs.max(1)
        correct_train += (predicted == tgt).sum().item()
        total_train += tgt.size(0)
    
    train_accuracy = correct_train / total_train * 100
    print(f"\nEPOCH #{ep + 1} SUMMARY :\n\tLearning Rate : {scheduler.get_last_lr()[0]}\n\tAvg training loss : {total_train_loss/count}\n\tTraining accuracy : {round(train_accuracy, 2)} %")
    
    alex_net.eval()
    
    valid_progress_bar = tqdm(valid_dataloader, desc = f"Epoch #{ep + 1} (Validation):", unit = "batch(es)", leave = False)
    total_valid_loss = 0
    correct_valid = 0
    count_valid = 0

    with torch.no_grad():
        for src, tgt in valid_progress_bar:
            src = src.to(device)
            tgt = tgt.to(device)
            
            # Forward pass
            outputs = alex_net(src)
            loss = criterion(outputs, tgt)
            
            # Calculate validation loss and accuracy
            total_valid_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_valid += (predicted == tgt).sum().item()
            count_valid += tgt.size(0)
    
    avg_valid_loss = total_valid_loss / len(valid_dataloader)
    
    
    valid_accuracy = correct_valid / count_valid * 100
    print(f"\n\tAvg validation loss : {avg_valid_loss}\n\tValidation accuracy : {round(valid_accuracy, 2)} %")
    
    scheduler.step(avg_valid_loss) 
        
        
        
alex_net.save("alex_net_cifar100.pt")
        
    