import ray
import traceback

from .config import MuZeroConfig
from .storage import SharedStorage, ReplayBuffer
from .selfplay import run_selfplay
from .train import train_network

# MuZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
class Muzero(object):
    def __init__(self, config: MuZeroConfig):
        self.config = config
        self.storage = SharedStorage(config)
#         self.replay_buffer = ReplayBuffer(config)
        
    def launch_job(self, f, *args):
#         f(*args)
        f.remote(*args)
    
    # Helpful for debugging
    def launch_job_serial(self, f, *args):
        f(*args)
#         f.remote(*args)
    
    def run(self):
        # Configure worker processes
#         train_worker = train_network.options(num_gpus=self.config.num_train_gpus)
#         shared_storage_worker = SharedStorage.remote()
        replay_buffer_worker = ReplayBuffer.remote(self.config)
        
        # Launch worker processes
        for _ in range(self.config.num_actors):
            self.launch_job_serial(run_selfplay, self.config, self.storage, replay_buffer_worker)
#         self.launch_job(train_worker, self.config, self.storage, replay_buffer_worker)
        self.launch_job_serial(train_network, self.config, self.storage, replay_buffer_worker)
#         best_network = self.storage.latest_network()