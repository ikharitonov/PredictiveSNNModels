from nnconfigs.Config import BaseConfig
from pathlib import Path

class SNNCustomConfig(BaseConfig):
    def __init__(self, cli_args=None, model_name=None, dataset_name=None, configuration_name=None, configuration_file=None, continue_training=False):

        self.base_path = Path.home() / 'RANCZLAB-NAS/iakov/produced/mnist_sequence_checkpoints'
        self.default_config_path = Path.home() / 'code/SensorymotorPredictiveModel/mnist_digits_snn/default_configs/snn_config.txt'
        
        super().__init__(cli_args, model_name, dataset_name, configuration_name, configuration_file, continue_training)

        data_path = Path.home() / 'RANCZLAB-NAS/iakov/produced/'
        self.dirs = {
            "training": data_path / f'{self.dataset_name}.npy',
            "training_1to1_targets": data_path / f'{self.dataset_name}_1to1_targets.npy',
            "training_predictive_targets": data_path / f'{self.dataset_name}_predictive_targets.npy',
        }
        self.weights_save_dir = self.base_path / self.training_name
        self.pretrained_weights_path = Path.home() / 'RANCZLAB-NAS/iakov/produced/classification_task_weights/mnist_classification_weights_matrix.npy'