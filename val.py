from modelparts.loadData import ExDarkDataset, custom_collate_fn, ExDark
from modelparts.validation import validate_epoch_without_model

from torch.utils.data import DataLoader
import torch



class BaseConfig:   
        dataset_path = "DatasetExDark"
        experiment_name = "base_experiment"
        batch_size = 32
        num_workers = 8
        target_size = (640, 640)
        alt_loss_pattern = []
        class_filter = [1,2] #Bicycle(1), Boat(2), Bottle(3), Bus(4), Car(5), Cat(6), Chair(7), Cup(8), Dog(9), Motorbike(10), People(11), Table(12)
        light_filter = None                         #Low(1), Ambient(2), Object(3), Single(4), Weak(5), Strong(6), Screen(7), Window(8), Shadow(9), Twilight(10)
        location_filter = None                      #Indoor(1), Outdoor(2)
        num_epochs = 15
        learning_rate = 1e-5
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = BaseConfig()

dataset = ExDark(filepath=config.dataset_path)
validation_image_paths = dataset.load_image_paths_and_classes(config, split_filter=[2])
validation_dataset = ExDarkDataset(dataset, validation_image_paths, config.target_size)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    collate_fn=custom_collate_fn
)

validate_epoch_without_model(0, validation_loader, config)