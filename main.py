# https://github.com/EmilienDupont/wgan-gp/blob/master/main.py
import torch
import torch.optim as optim
from src.datasetmgr import get_dataloader
from src.models import UNet, ZhuNet
from .trainer import Trainer

data_loader = get_dataloader(data_dirs=[''], batch_size=64)
generator = UNet(3, 2)
discriminator = ZhuNet()

# Initialize optimizers
lr = 1e-4
betas = (.9, .99)
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

# Train model
epochs = 200
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer, use_cuda=torch.cuda.is_available())
trainer.train(data_loader, epochs, save_training_gif=False)

# Save models
# name = 'mnist_model'
# torch.save(trainer.G.state_dict(), './gen_' + name + '.pt')
# torch.save(trainer.D.state_dict(), './dis_' + name + '.pt')