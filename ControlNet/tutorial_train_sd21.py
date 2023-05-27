from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint



# Configs
resume_path = './models/checkpoint-epoch=01.ckpt'
batch_size = 10
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = True


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath='./models',
    filename='checkpoint-{epoch:02d}',
    save_last = True 
)

trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, checkpoint_callback], \
 max_epochs=2, default_root_dir="./models")


# Train!
trainer.fit(model, dataloader)

# Save the model
model.save_checkpoint('saved_model.ckpt')
