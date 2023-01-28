from icecream import ic

def freeze(model, idx_to_unfreeze_from_last, device):
    full_layers = list(model.named_children())
    list_layers = full_layers[-idx_to_unfreeze_from_last:]

    for param in model.parameters():
        param.requires_grad = False

    for layer in list_layers:
        for param in layer[1].parameters():
            param.requires_grad = True

    for name, layer in full_layers:
        ic(name)
        for param in layer.parameters():
            print(param.requires_grad)
    
    # summary(model, [(3, 224, 224), (820, )], device=device)