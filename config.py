class Config:

    def __init__(self):
        self.batch_size = 8
        self.num_classes = 1
        self.nb_epochs_per_saving = 1
        self.pin_mem = True
        self.num_cpu_workers = 8
        self.nb_epochs = 100
        self.cuda = True
        self.concrete_dropout = False # Useful only for MC-Dropout
        # Optimizer
        self.lr = 1e-4
        self.weight_decay = 5e-5
        # Scheduler
        self.gamma_scheduler = 0.9
        self.step_size_scheduler = 10


