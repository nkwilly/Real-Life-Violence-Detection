import datetime
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from base.base_trainer import BaseTrain

class ModelTrainer(BaseTrain):
    def __init__(self, model, data_train, data_validate, config):
        super(ModelTrainer, self).__init__(model, data_train, data_validate, config)
        self.callbacks = []
        self.log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.init_callbacks()

    def init_callbacks(self):
        # Callback 1: Sauvegarde du meilleur modèle au format HDF5 (.h5)
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(
                    self.config.callbacks.checkpoint_dir,
                    f"{self.config.exp.name}-best-model.h5"  # Extension .h5 explicite
                ),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=True,
                save_weights_only=False,  # Sauvegarde le modèle entier (architecture + poids)
                verbose=1
            )
        )

        # Callback 2: Arrêt anticipé si la perte de validation stagne
        self.callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=self.config.callbacks.ESPatience,
                restore_best_weights=True
            )
        )

        # Callback 3: Réduction du taux d'apprentissage sur plateau
        self.callbacks.append(
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=self.config.callbacks.lrSPatience,
                min_lr=self.config.callbacks.lrSmin_lr
            )
        )

        # Callback 4: TensorBoard pour la visualisation
        self.callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=self.log_dir,
                histogram_freq=1
            )
        )

    def train(self):
        print(f"train_data = {self.train_data}")
        print(f"validation_data = {self.val_data}")
        print(f"epochs = {self.config.trainer.EPOCHS}")
        print(f"callbacks = {self.callbacks}")
        print(f"config = {self.config}")
        history = self.model.fit(
            self.train_data,
            validation_data=self.val_data,
            epochs=self.config.trainer.EPOCHS,
            callbacks=self.callbacks,
            verbose=1
        )

if __name__ == "__main__":
    pass