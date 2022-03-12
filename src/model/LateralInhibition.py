import torch as torch
import torch.nn as nn

from model.ApproxHeaviside import ApproxHeaviside

class LateralInhibition(nn.Module):
    def __init__(self, num,sigma):
        super().__init__()

        self.sigma=sigma
        ApproxHeaviside.sigma=sigma

        self.num_units=num
        myw=torch.empty(num,num)
        nn.init.normal_(myw,std=0.02)
        self.w = nn.Parameter(myw)
        self.b = nn.Parameter(torch.zeros(num))
        #self.relu=nn.ReLU()

        #print("BUILD:")
        #print("   w=",self.w)
        #print("   b=",self.b)
        #print(self.zerodiag)
        #exit(-1)

    def forward(self, inputs):
        #print("CALL:")
        #print(inputs)
        #print("reshape")
        shape=list(inputs.size())
        #print(shape)
        rinputs=torch.reshape(inputs,(shape[0]*shape[1],1,self.num_units))
        #print("   rinputs=",rinputs)

        w1=self.w.clone()
        w1=w1.fill_diagonal_(0)
        #w1=self.w # aici e gresit => tb prin get diag => neg => add


        #logger.info("   w1=",w1)

        #w1=tf.linalg.set_diag(self.w, self.zerodiag)
        inhib=torch.matmul(rinputs,w1) # aici tb w1
        #print("   inhib=",inhib)

        #print("   rinputs.size=",rinputs.shape)
        #print("   w1.size=",w1.shape)
        #print("   inhib.size=",inhib.shape)



        inhib=inhib+self.b
        #print("   inhib+b=",inhib)
        #inhib=self.relu(inhib)
        inhib=ApproxHeaviside.apply(inhib) #,self.values)

        #print("   inhib(relu)=",inhib)
        inhib=torch.reshape(inhib,[shape[0]*shape[1],self.num_units])
        #print("   inhib=",inhib)
        w_inhib=torch.diag_embed(inhib)
        #print("   w_inhib=",w_inhib)
        r=torch.matmul(rinputs, w_inhib)
        #print("   r=",r)
        #r=torch.reshape(r,[shape[0],self.num_units])
        r=torch.reshape(r,shape)
        #logger.info("   r(reshape)=",r)
        #exit(-1)
        return r
