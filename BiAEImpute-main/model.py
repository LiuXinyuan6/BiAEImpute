import torch.nn as nn
import torch
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

latent_size = 128



class Row_Encoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(Row_Encoder, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size

        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256,latent_size),
            nn.ReLU()
        )


    def forward(self, x):
        encoded = self.model(x)
        return encoded


class Row_Decoder(nn.Module):
    def __init__(self, latent_size, input_size):
        super(Row_Decoder, self).__init__()
        self.latent_size = latent_size
        self.input_size = input_size

        self.model = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, input_size),
            nn.ReLU()
        )

    def forward(self, z):
        x = self.model(z)
        return x


class Col_Encoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(Col_Encoder, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size

        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256,latent_size),
            nn.ReLU()
        )


    def forward(self, x):
        # x=x.squeeze(dim=0)
        encoded = self.model(x)
        return encoded


class Col_Decoder(nn.Module):
    def __init__(self, latent_size, input_size):
        super(Col_Decoder, self).__init__()
        self.latent_size = latent_size
        self.input_size = input_size

        self.model = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, input_size),
            #nn.Tanh(),
            nn.ReLU()
        )

    def forward(self, z):
        x = self.model(z)
        return x



if __name__ == '__main__':

    # encoder = Row_Encoder(input_size,latent_size)
    # print(encoder)
    # decoder = Row_Decoder(latent_size,input_size)
    # print(decoder)

    encoder = Col_Encoder(input_size,latent_size)
    print(encoder)
    decoder = Col_Decoder(latent_size,input_size)
    print(decoder)
