import dataloaders.mp3_dataloader as mp3_dataloader
import librosa
import torch
import models.autoencoder as autoencoder
import torch.optim as optim
import torch.nn as nn

sample_path = 'data/'
sample_rate = 16000
sample_length = 1024*10
n_epochs = 100
save_path = 'checkpoints/VAE.pth'

def main():
    ## check if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = mp3_dataloader.MP3Dataset(sample_path, sample_length=sample_length, SR=sample_rate)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)

    model = autoencoder.VAE_Audio(input_size=sample_length, latent_size=512).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for i, sample in enumerate(dataloader):
            optimizer.zero_grad()
            x = sample.to(device)
            x_hat, mu, logvar = model(x)
            loss = autoencoder.loss_function(x, x_hat, mu, logvar)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'epoch {epoch} step {i} loss {loss.item()}')
        torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    main()