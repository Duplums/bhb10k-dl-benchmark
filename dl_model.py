import torch
from torch.nn import DataParallel
from metrics import METRICS
from tqdm import tqdm
import logging

class DLModel:

    def __init__(self, net, loss, config, args, loader_train=None, loader_val=None, loader_test=None, scheduler=None):
        """

        Parameters
        ----------
        net: subclass of nn.Module
        loss: callable fn with args (y_pred, y_true)
        config: Config object with hyperparameters
        args: args given including data pathss
        loader_train, loader_val, loader_test: pytorch DataLoaders for training/validation/test
        scheduler (optional)
        """
        super().__init__()
        self.logger = logging.getLogger("DLBenchmark")
        self.loss = loss
        self.model = net
        self.optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = scheduler
        self.loader = loader_train
        self.loader_val = loader_val
        self.loader_test = loader_test
        self.model_path = args.model_path
        self.device = torch.device("cuda" if config.cuda else "cpu")
        if config.cuda and not torch.cuda.is_available():
            raise ValueError("No GPU found: set cuda=False parameter.")
        self.config = config
        self.metrics = {m: METRICS[m] for m in args.metrics}
        self.model = DataParallel(self.model).to(self.device)


    def training(self):
        print(self.loss)
        print(self.optimizer)

        for epoch in range(self.config.nb_epochs):
            ## Training step
            self.model.train()
            nb_batch = len(self.loader)
            training_loss = []
            pbar = tqdm(total=nb_batch, desc="Training")
            for (inputs, labels) in self.loader:
                pbar.update()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                y = self.model(inputs)
                batch_loss = self.loss(y, labels)
                batch_loss.backward()
                self.optimizer.step()
                training_loss += float(batch_loss) / nb_batch
            pbar.close()

            ## Validation step
            nb_batch = len(self.loader_val)
            pbar = tqdm(total=nb_batch, desc="Validation")
            val_loss = 0
            y_val = []
            y_true_val = []
            with torch.no_grad():
                self.model.eval()
                for (inputs, labels) in self.loader_val:
                    pbar.update()
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    y = self.model(inputs)
                    y_val.extend(y)
                    y_true_val.extend(labels)
                    batch_loss = self.loss(y, labels)
                    val_loss += float(batch_loss) / nb_batch
            pbar.close()
            all_metrics = dict()
            for name, metric in self.metrics.items():
                all_metrics[name] = metric(torch.tensor(y_val), torch.tensor(y_true_val))
            all_metrics_str = "\t".join(["{}={:.2f}".format(name, m) for (name, m)in all_metrics.items()])
            print("Epoch [{}/{}] Training loss = {:.4f}\t Validation loss = {:.4f}\t".format(
                epoch+1, self.config.nb_epochs, training_loss, val_loss)+"\t".join(all_metrics)+all_metrics_str,
                  flush=True)

            if self.scheduler is not None:
                self.scheduler.step()


    def testing(self):
        self.load_model(self.model_path)
        nb_batch = len(self.loader_test)
        pbar = tqdm(total=nb_batch, desc="Test")
        y_pred = []
        y_true = []
        with torch.no_grad():
            self.model.eval()
            for (inputs, labels) in self.loader_val:
                pbar.update()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                y = self.model(inputs)
                y_pred.extend(y)
                y_true.extend(labels)
        pbar.close()
        all_metrics = dict()
        for name, metric in self.metrics.items():
            all_metrics[name] = metric(torch.tensor(y_pred), torch.tensor(y_true))
        all_metrics_str = "\t".join(["{}={:.2f}".format(name, m) for (name, m) in all_metrics.items()])
        print(all_metrics_str, flush=True)


    def load_model(self, path):
        checkpoint = None
        try:
            checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        except BaseException as e:
            self.logger.error('Impossible to load the checkpoint: %s' % str(e))
        if checkpoint is not None:
            try:
                if hasattr(checkpoint, "state_dict"):
                    unexpected = self.model.load_state_dict(checkpoint.state_dict())
                    self.logger.info('Model loading info: {}'.format(unexpected))
                elif isinstance(checkpoint, dict):
                    if "model" in checkpoint:
                        unexpected = self.model.load_state_dict(checkpoint["model"], strict=False)
                        self.logger.info('Model loading info: {}'.format(unexpected))
                else:
                    unexpected = self.model.load_state_dict(checkpoint)
                    self.logger.info('Model loading info: {}'.format(unexpected))
            except BaseException as e:
                raise ValueError('Error while loading the model\'s weights: %s' % str(e))





