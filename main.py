# https://github.com/EmilienDupont/wgan-gp/blob/master/main.py
# %%
import torch
import torch.optim as optim
from src.datasetmgr import get_dataloader
from src.models import UNet, ZhuNet
from src.config import getConfig
from trainer import Trainer
# %%
args = getConfig('/home/kevin2li/ut-gan/src/config/default.yml')
data_loader = get_dataloader(data_dirs=args['data_dirs'], batch_size=args['batch_size'])
generator = UNet(1, 1)
discriminator = ZhuNet()

# %%
G_optimizer = optim.Adam(generator.parameters(), lr=args['lr'], betas=args['betas'])
D_optimizer = optim.Adam(discriminator.parameters(), lr=args['lr'], betas=args['betas'])

# Train model
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer, use_cuda=torch.cuda.is_available())
trainer.train(data_loader, args['max_epoch'], save_training_gif=False)

# Save models
# name = 'mnist_model'
# torch.save(trainer.G.state_dict(), './gen_' + name + '.pt')
# torch.save(trainer.D.state_dict(), './dis_' + name + '.pt')