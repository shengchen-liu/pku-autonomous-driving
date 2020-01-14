import os

PROJECT_PATH = os.path.dirname(os.path.realpath(__file__).replace('/coding',''))

class DefaultConfigs(object):
    def __init__(self):
        self.data_dir = os.path.join(PROJECT_PATH, "input", "3d-object-detection-for-autonomous-vehicles")
        self.train_images = os.path.join(self.data_dir, "train_images")  # where is your train images
        self.train_data = os.path.join(self.data_dir, "train_data")
        self.train_lidar = os.path.join(self.data_dir, "train_lidar")
        self.train_maps = os.path.join(self.data_dir, "train_maps")
        self.test_images = os.path.join(self.data_dir, "test_images")  # where is your test images
        self.test_data = os.path.join(self.data_dir, "test_data")
        self.test_lidar = os.path.join(self.data_dir, "test_lidar")
        self.test_maps = os.path.join(self.data_dir, "test_maps")

        self.split = os.path.join(PROJECT_PATH, "split")

        self.results = os.path.join(PROJECT_PATH, "results")
        self.pretrain_model = os.path.join(self.results, "pretrain_model")
        self.model_name = "unet" #resnet18-seg-full-softmax-foldb1-1-4balance, resnet34-cls-full-foldb0-0
        self.initial_checkpoint = None # 000_model.pth
        self.back_up = os.path.join(self.results, self.model_name, "backup")
        self.debug = 0
        self.mode = "train" # train, test

        self.logs = os.path.abspath(os.path.join(self.results, "logs"))
        self.weights = os.path.abspath(os.path.join(self.results, "checkpoints"))
        self.best_models = os.path.abspath(os.path.join(self.weights, "best_models"))
        self.submit = os.path.abspath(os.path.join(self.results, "submit"))
        self.lr = 0.001 #0.01, 0.001
        self.batch_size = 8 # 12, 20
        self.epochs = 10
        self.resume = True
        self.gpus = "0"
        self.fold=5
        self.model='baseline_pytorch'
        self.embed_size = 300  # how big is each word vector
        self.max_features = 120000  # how many unique words to use (i.e num rows in embedding vector)
        self.maxlen = 72  # max number of words in a question to use
        self.sample = 0 # for debug
        self.embed_method = "mean" # concat or mean

        # check if the following dirs exist, if not, make it
        out_dir = os.path.join(self.results, self.model_name)
        log_dir = os.path.join(self.logs, self.model_name)
        initial_checkpoint = os.path.join(self.results, self.model_name, 'checkpoint')
        for dirs in [self.results, self.split, self.pretrain_model, out_dir, self.back_up, initial_checkpoint, self.logs, log_dir]:
            if not os.path.exists(dirs):
                os.mkdir(dirs)



config = DefaultConfigs()