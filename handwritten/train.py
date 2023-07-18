import pytorch_lightning as pl
from litdata import LitData
from litmodel import LitModel


BATCH_SIZE = 128
LATENT_DIM = 64
LR = 0.00001
MAX_EPOCHES = 200

if __name__ == '__main__':

    lit_data = LitData(BATCH_SIZE)
    lit_model = LitModel(lit_data.dims, LATENT_DIM, LR)

    trainer = pl.Trainer(max_epochs=MAX_EPOCHES)
    trainer.fit(lit_model, lit_data)
