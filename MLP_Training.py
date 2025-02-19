
import torch
import torch.nn as nn
import torch.nn.functional as F

class mlp_training():
    def __init__(self,train_data,train_label,validation_data,validation_label,mini_batch_size):
        self.train_data=train_data
        self.train_label=train_label
        self.validation_data=validation_data
        self.validation_label=validation_label
        self.mini_batch_size=mini_batch_size

    def mlp_architecture(self):
        # for reproducibility
        g=torch.Generator().manual_seed(2147483647) 
        #look up list
        #larger embedding
        #cramming 27 characters into 10 dimentional space from each word
        #each row represents one character
        self.C=torch.rand((27,10),generator=g)
        #Layer1 weights and biases
        #slightly shorter hidden layer
        self.W1=torch.rand((30,200),generator=g)
        self.B1=torch.rand(200,generator=g)
        #Layer2 weights and biases
        self.W2=torch.rand((200,27),generator=g)
        self.B2=torch.rand(27,generator=g)
        self.parameters=[self.C,self.W1,self.B1,self.W2,self.B2]

        print("number of parameters: ",sum(p.nelement() for p in self.parameters))
        #to take derivative
        for p in self.parameters:
            p.requires_grad=True

    def forward_pass(self):
        #mini batch creation
        #32 random numbers from the range of x index
        self.batch_idx=torch.randint(0,self.train_data.shape[0],(self.mini_batch_size,))

        #embbedding layer
        #all word possibilities of 3 characters
        #we use batch_idx to only grab 32 random rows for a batch
        #embedding will be 32 ,not the whole data
        emb=self.C[self.train_data[self.batch_idx]] #(all possible 3 characters,3,10)

        #layer1 and activation
        #view is another Pytorch's manipulation function which is more space effcient
        # it can change the arrangment of tensors to become compatible with the next layer input
        #new arrangement should be 3 characters x 10 embeddings=30

        #all possible 3 charactersx30 @ 30x200
        #The tanh function (hyperbolic tangent) squashes values to the range [-1, 1], adding non-linearity.
        #It is a non-linear function that transforms input values into a range between -1 and 1.
        h=torch.tanh(emb.view(len(emb),30)@self.W1+self.B1)  #(all,200)

        #layer2
        logits=h@self.W2+self.B2 #(32x27)

        return logits

    def training(self,epoch,learning_rate):
        #epochs
        #also we change the learning rate with i
        for i in range(epoch):
            #forward pass
            logits=self.forward_pass() #(32x27)


            #those calculcations are the same as the cross enthropy loss
            #always use cross enthropy instead of the manual calculations due to the efficiency
            #we use batch_idx in Y for batches
            loss=F.cross_entropy(logits, self.train_label[self.batch_idx])


            #backward pass

            #set the gradients to zero
            for p in self.parameters:
                p.grad=None

            #populate those gradient
            loss.backward()
            
            #update
            #when encountering platue, we can train in a shorter steps by learning decay (10x lower) to imporve the loss
            for p in self.parameters:
                #p.data refers to the actual data (values) of the parameter tensor p. This is where the weights and biases are stored.
                #p.grad contains the gradients of the loss with respect to the parameter p
                #The negative sign (-) indicates that we are moving in the opposite direction of the gradient, which is the direction of steepest descent

                #choosing a suitable learning rate is so important
                #we have tested different learning rates
                p.data+=-learning_rate*p.grad
            print(f"epoch:{i+1},loss:{loss.item()}")

    def inference(self):
        #validation set loss
        val_batch_idx=torch.randint(0,self.validation_data.shape[0],(self.mini_batch_size,))
        emb=self.C[self.validation_data[val_batch_idx]] #(32,3,10)
        h=torch.tanh(emb.view(len(emb),30)@self.W1+self.B1)  
        logits=h@self.W2+self.B2 #(all,27)
        val_loss=F.cross_entropy(logits,self.validation_label[val_batch_idx])
        print("final validation loss: ",val_loss.item())
