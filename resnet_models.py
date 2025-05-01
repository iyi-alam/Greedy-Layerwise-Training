import torch
import torch.nn as nn


class skipblock(nn.Module):
    def __init__(self, in_features, kernel_size):
        super().__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.Conv2d(in_features, in_features, kernel_size = kernel_size, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(in_features),
            nn.Conv2d(in_features, in_features, kernel_size = kernel_size, padding = 1),
            nn.ReLU(),
        )
        self.layer_ids = [0 for _ in self.layers]

    def forward(self, x: torch.tensor):
        x = x + self.layers(x)
        return x
    

class downsampling(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_features, out_features, 1),
            nn.MaxPool2d(2,2)
        )
        self.layer_ids = [1 for _ in self.layers]

    def forward(self, x):
        return self.layers(x)
    

class greedy_resnet(nn.Module):
    def __init__(self, image_channels, in_features, out_features, initial_linear_input_size, num_classes):
        super().__init__()
        skip = skipblock(in_features, 3)
        #downsample = downsampling(in_features, out_features)
        layerlist = [*skip.layers] #+ [*downsample.layers]
        layerids= [*skip.layer_ids]
        self.skip_length = len(skip.layers)
        self.flops = 0

        self.conv0 = nn.Conv2d(image_channels, in_features, kernel_size = 3, padding = 1)

        self.layers = nn.ModuleList(layerlist)
        self.layer_ids = layerids+[2] # last element is just added to avoid invalid
        # index due to accessing self.layer_ids[i+1] in forward function

        # Create a layer ID to identify the layers of skip block
        
        self.greedy_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(1,3),
        )
        self.output_layers = nn.Linear(initial_linear_input_size, num_classes)

    def forward(self, x):
        y = self.conv0(x)
        saved_input = y
        skip_counter = 1
        for i, layer in enumerate(self.layers):
            y = layer(y)
            #print('Received output from :', layer)

            if ((self.layer_ids[i] == 0 and (self.layer_ids[i+1] == 1 or \
                self.layer_ids[i+1] == 2)) or (skip_counter != 0 and \
                (skip_counter % self.skip_length == 0))):
                
                # Next layer is downsampling layer so add the skip connection
                y = y + saved_input
                saved_input = y
                skip_counter = 0
                #print('Added Skip Connection')

            if (self.layer_ids[i] == 1 and (self.layer_ids[i+1] == 0 or \
                self.layer_ids[i+1] == 2)):
                # Downsampling layer ended and skip layer begun, so save the input
                #print('Saved last input')
                saved_input = y
                skip_counter = -3

            skip_counter += 1
            #print('\n')

        y = self.output_layers(self.greedy_layers(y))
        return y

    def compute_flops(self, H, W):

        # Flops of initial conv layers
        self.flops += H*W*self.conv0.in_channels*self.conv0.out_channels \
                        *(self.conv0.kernel_size[0]**2)
        self.flops /= 1e9
        
        # Flops through intermediate layers
        for i,layer in enumerate(self.layers):
            if isinstance(layer, nn.Conv2d):
                layer_flops = H*W*layer.in_channels*layer.out_channels \
                            *(layer.kernel_size[0]**2)
                self.flops += layer_flops/1e9

            elif isinstance(layer, nn.BatchNorm2d):
                self.flops += layer.num_features*H*W / 1e9
            elif isinstance(layer, nn.MaxPool2d):
                H = H/2
                W = W/2
            else:
                continue

        # Flops through linear layer
        self.flops += (self.output_layers.in_features*self.output_layers.out_features)/1e9
        
        return self.flops

def add_layers(model, in_features, out_features, layer_id, num_classes, device):
    if layer_id ==0:
        # Add skip layer only
        skip = skipblock(in_features, 3)
        layerlist = [*skip.layers]
        layerids = [*skip.layer_ids]
        
    else:
        downsample = downsampling(in_features, out_features)
        layerlist = [*downsample.layers]
        layerids = [*downsample.layer_ids]
        
    model.layers.extend(layerlist)
    model.layer_ids = model.layer_ids[:-1] + layerids + [2]

    # Reject the last linear layer and add new layer
    model.output_layers = nn.Linear(out_features, num_classes)
    model = model.to(device)


def update_optimizer(optimizer, model, lr=0.001, momentum = 0.9):
    # Get the old optimizer's settings (momentum, weight decay, etc.)
    optimizer_state = optimizer.state_dict()

    # Create a new optimizer with updated model parameters
    new_optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum = momentum)

    return new_optimizer