from AlexNet import AlexNet


from load_data import get_cifar100_test
import torch 
from tqdm import tqdm 

device = "cuda" if torch.cuda.is_available() else "cpu"

dataloader = get_cifar100_test(
    batch_size = 40
)


alex_net = AlexNet(num_classes = 100, dropout = 0.2)
alex_net.load("alex_net_cifar100.pt")
alex_net = alex_net.to(device) 


alex_net.eval()
    
test_progress_bar = tqdm(dataloader, desc = f"(Testing):", unit = "batch(es)", leave = False)
total_valid_loss = 0
correct_valid = 0
count_valid = 0

criterion = torch.nn.CrossEntropyLoss()

with torch.no_grad():
    for src, tgt in test_progress_bar:
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
    
avg_valid_loss = total_valid_loss / len(dataloader)
valid_accuracy = correct_valid / count_valid * 100
print(f"\n\tAvg test loss : {avg_valid_loss}\n\tTesting accuracy : {round(valid_accuracy, 2)} %")
