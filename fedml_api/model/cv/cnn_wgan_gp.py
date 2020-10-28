import torch 
import torch.nn as nn
import torch.autograd as autograd

def calc_gradient_penalty(netD, real_data, fake_data, device, _lambda):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data).to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * _lambda
    return gradient_penalty

class Discriminator(nn.Module):
    def __init__(self, model_dim):
        super(Discriminator, self).__init__()
        self.DIM = model_dim

        main = nn.Sequential(
            nn.Conv2d(1, self.DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(self.DIM, 2*self.DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(2*self.DIM, 4*self.DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
            
        )
        self.main = main
        self.output = nn.Sequential(
            nn.Linear(4*4*4*self.DIM, 1),
            nn.Sigmoid()
            )

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4*4*4*self.DIM)
        out = self.output(out)
        return out

class Generator(nn.Module):
    def __init__(self, model_dim, output_dim):
        super(Generator, self).__init__()
        self.DIM = model_dim
        self.output_dim = output_dim

        preprocess = nn.Sequential(
            nn.Linear(128, 4*4*4*self.DIM),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4*self.DIM, 2*self.DIM, 5),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2*self.DIM, self.DIM, 5),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(self.DIM, 1, 8, stride=2)

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4*self.DIM, 4, 4)
        output = self.block1(output)
        output = output[:, :, :7, :7]
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        return output.view(-1, self.output_dim)




    


