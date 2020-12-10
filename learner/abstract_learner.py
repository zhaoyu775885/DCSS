import torch
from abc import ABC
from abc import abstractmethod
from utils.writer import Writer
import os


class AbstractLearner(ABC):
    def __init__(self, dataset, net, device, args):
        self.dataset = dataset
        self.net = net
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.args = args

        self.forward = self.net.to(self.device)
        self.loss_fn = self._setup_loss_fn()

        # logs save in text and visualize in tensorboard
        self.recoder = Writer(self.args.log_dir)

    def _build_dataloader(self, batch_size, is_train, valid=False, search=False):
        return self.dataset.build_dataloader(batch_size, is_train, valid, search)

    @abstractmethod
    def _setup_loss_fn(self):
        pass

    @abstractmethod
    def _setup_optimizer(self):
        pass

    @abstractmethod
    def train(self, **kwargs):
        pass

    @abstractmethod
    def test(self, **kwargs):
        pass

    def save_model(self, model_path):
        # todo: supplement the epoch info in the model_name
        # todo: use the checkpoint file to save the model queue
        dirname = os.path.dirname(model_path)
        with open(os.path.join(dirname, 'checkpoint'), 'w') as fH:
            fH.write(os.path.basename(model_path))
        torch.save(self.net.state_dict(), os.path.join(model_path))

    def load_model(self, dir_path):
        """
        make sure that the checkpoint on the disk contains all related variables
        in current network.
        """
        with open(os.path.join(dir_path, 'checkpoint'), 'r') as fH:
            model_full_name = fH.readline().strip()
            print(model_full_name)
            model_path = os.path.join(dir_path, model_full_name)
            disk_state_dict = torch.load(os.path.join(model_path), map_location=lambda storage, loc: storage.cuda())
            try:
                self.net.load_state_dict(disk_state_dict)
            except RuntimeError:
                print('load model with conflicts, please check the difference.')
                state_dict = self.net.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = disk_state_dict[key]
                self.net.load_state_dict(state_dict)
