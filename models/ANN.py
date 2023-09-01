import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, in_features=4, h1 =8,h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2, out_features)
    
    # 입력 -> 은닉층 1 -> 은닉층 2 -> 출력
        
    #순전파
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
    
        return x
	
def ANN(args):
    model = Model(in_features= args.in_features, h1=args.h1, h2=args.h2, out_features=args.out_features)
    return model 