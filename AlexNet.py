import torch 
import torch.nn as nn 


class AlexNet(nn.Module):
    
    def __init__(self, 
                 num_classes = 100,
                 dropout = 0.1):
        super().__init__()
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            nn.Conv2d(64, 192, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            nn.Conv2d(192, 384, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            
            nn.Conv2d(384, 256, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )
        
        self.classifier_layer = nn.Sequential(
            nn.Dropout(p = dropout),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace = True),
            
            nn.Dropout(p = dropout),
            nn.Linear(4096, 4096), 
            nn.ReLU(inplace = True),
            
            nn.Linear(4096, num_classes)
        )
        
        
        print(f"Total number of parameters : {sum(p.numel() for p in self.parameters())}")
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        
        return self.classifier_layer(x) 

    def save(self, filename):
        torch.save(self.state_dict(), filename)
    
    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        

    