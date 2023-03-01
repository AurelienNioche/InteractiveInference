import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.zeros(1).to(mps_device)
    print (x)
    print(x.device)
else:
    print ("MPS device not found.")