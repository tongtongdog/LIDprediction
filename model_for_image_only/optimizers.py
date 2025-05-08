import torch
# from lion_pytorch import Lion

def create_optim(name, model, args):
    if name == 'adam':
        optimizer    = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4, amsgrad=False)

    elif name == 'adamw':
        optimizer    = torch.optim.AdamW(params=model.parameters(), lr=args.min_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05, amsgrad=False) 

    elif name == 'sgd':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    # elif name == 'lion':
    #     optimizer = Lion(params=model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-3)

    else :
        raise KeyError("Wrong optim name `{}`".format(name))        

    return optimizer

