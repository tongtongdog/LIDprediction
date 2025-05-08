from arch.smart_net.model import Up_SMART_Net, Up_SMART_Net_Dual_CLS_SEG, Up_SMART_Net_Dual_CLS_REC, Up_SMART_Net_Single_CLS, Up_SMART_Net_Dual_SEG_REC


# Create Model
def create_model(stream, name):
    if stream == 'Upstream':
        
        if name == 'Up_SMART_Net':
            model = Up_SMART_Net()     

        ## Dual    
        elif name == 'Up_SMART_Net_Dual_CLS_SEG':
            model = Up_SMART_Net_Dual_CLS_SEG()

        elif name == 'Up_SMART_Net_Dual_CLS_REC':
            model = Up_SMART_Net_Dual_CLS_REC()

        elif name == 'Up_SMART_Net_Dual_SEG_REC':
            model = Up_SMART_Net_Dual_SEG_REC()       

        ## Single
        elif name == 'Up_SMART_Net_Single_CLS':
            model = Up_SMART_Net_Single_CLS()

        else :
            raise KeyError("Wrong model name `{}`".format(name))        

    else :
        raise KeyError("Wrong stream name `{}`".format(stream))
        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Learnable Params:', n_parameters)   

    return model