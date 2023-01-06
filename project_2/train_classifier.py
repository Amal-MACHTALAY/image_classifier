"""
@author: amal machtalay
"""

from torch import nn
import torch

# Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
class Classifier(nn.Module):
    
    def __init__(self, input_size, output_size, num_hiddens, p_dropout= 0.1):
        super().__init__()
        
        # Input layer
        self.input_layers = nn.Linear(input_size, num_hiddens[0])
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(n, m) for n, m in zip(num_hiddens[:-1], num_hiddens[1:])])
        
        # Output layer
        self.output_layers = nn.Linear(num_hiddens[-1], output_size)

        # Dropout function 
        self.dropout = nn.Dropout(p=p_dropout) # p : drop probability
        
        # ReLU activation function
        self.activation = nn.ReLU()

    def forward(self, x):
        # make sure that input tensor is flattened
        x = x.view(x.shape[0], -1)
        # Add input layer
        x = self.input_layers(x)
        x = self.dropout(self.activation(x)) # Add activation and Dropout
        # Add hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.dropout(self.activation(x)) # Add activation and Dropout
        # Add output layer
        x = self.output_layers(x) 
        # Add Softmax function : convert to probability
        px = nn.functional.log_softmax(x,dim=1) 

        return px
    
# Train the classifier layers using backpropagation 
def run_Classifier(model, loss_func, optimizer, loaders, device, train_mode):
    
    run_loss = 0
    accuracy = 0
    #torch.set_grad_enabled(True)  # Context-manager
    
    # Looping through images, get a batch size of images on each loop
    for i, data in enumerate(loaders, 0):
        
        # get the [inputs, labels] and convert them into CUDA tensors
        inputs, labels = data[0].to(device), data[1].to(device)

        # forward + backward + optimize
        outputs = model.forward(inputs) # Forward Pass
        loss = loss_func(outputs, labels)
        if train_mode==True:
            optimizer.zero_grad() # clear gradient : avoid accumulation
            loss.backward() # calculate gradient
            optimizer.step() # update weights
            
        # running loss
        run_loss += loss.item() # the mean loss of "epoch"
        
        p_outputs = torch.exp(outputs)
        _, predicted = p_outputs.max(dim=1) # search for the highest proba
        accuracy += (predicted == labels.data).type(torch.FloatTensor).mean()

    run_loss = run_loss/len(loaders)
    accuracy = accuracy/len(loaders)
    #percent_accuracy = 100*accuracy

    return run_loss, accuracy#, percent_accuracy

def train_test_Classifier(num_epochs, model, loss_func, optimizer, train_loaders, valid_loaders, device):
    
    train_mode=True
    valid_mode=False
    
    # convert the parameters of model to CUDA tensors
    model.to(device)

    # Looping through epochs, each epoch is a full pass through the network
    for epoch in range(num_epochs):

        # Switch on the train mode
        model.train()

        # Train model
        train_loss,_ = run_Classifier(model, loss_func, optimizer, train_loaders, device, train_mode)

        # Switch off the train mode during validation
        model.eval() # turn off  some specific layers/parts of the model (ex. Dropouts Layers, BatchNorm Layers)  
        with torch.no_grad(): # Turn off gradients
            valid_loss, valid_accuracy = run_Classifier(model, loss_func, optimizer, valid_loaders, device, valid_mode)

        print(f"Epoch {epoch+1}/{num_epochs}... "
              f"Train loss: {train_loss:.3f}... "
              f"Valid loss: {valid_loss:.3f}... "
              f"Valid accuracy: {valid_accuracy:.3f}")
        
        train_loss = 0
        valid_loss = 0
        valid_accuracy = 0
        
        # Switch back to the train mode
        model.train()
            
    return 0
