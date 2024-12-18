# configs.py
import torch
from modelparts.modelStructure import UNet

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
    weight_decay = 0.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Base with Indoor only
class IndoorConfig(BaseConfig):
    location_filter = [1]  # Indoor
    experiment_name = "base_indoor_experiment"

# Base with Outdoor only
class OutdoorConfig(BaseConfig):
    location_filter = [2]  # Outdoor
    experiment_name = "base_outdoor_experiment"



# Base with each lighting type
class LowLightConfig(BaseConfig):
    light_filter = [1]  # Low
    experiment_name = "base_low_light_experiment"

class AmbientLightConfig(BaseConfig):
    light_filter = [2]  # Ambient
    experiment_name = "base_ambient_light_experiment"

class ObjectLightConfig(BaseConfig):
    light_filter = [3]  # Object
    experiment_name = "base_object_light_experiment"

class SingleLightConfig(BaseConfig):
    light_filter = [4]  # Single
    experiment_name = "base_single_light_experiment"

class WeakLightConfig(BaseConfig):
    light_filter = [5]  # Weak
    experiment_name = "base_weak_light_experiment"

class StrongLightConfig(BaseConfig):
    light_filter = [6]  # Strong
    experiment_name = "base_strong_light_experiment"

class ScreenLightConfig(BaseConfig):
    light_filter = [7]  # Screen
    experiment_name = "base_screen_light_experiment"

class WindowLightConfig(BaseConfig):
    light_filter = [8]  # Window
    experiment_name = "base_window_light_experiment"

class ShadowLightConfig(BaseConfig):
    light_filter = [9]  # Shadow
    experiment_name = "base_shadow_light_experiment"

class TwilightLightConfig(BaseConfig):
    light_filter = [10]  # Twilight
    experiment_name = "base_twilight_light_experiment"





#! Now running experiments:

class FullDatasetConfig(BaseConfig):
    experiment_name = "full_dataset_experiment"
    class_filter = [1,2,3,4,5,6,7,8,9,10,11,12]

class AlternateLossConfig(BaseConfig):
    alt_loss_pattern = [0,2,4,6,8,10,12,14]
    num_epochs = 15
    experiment_name = "alternate_loss_experiment"

class MultiLossConfig(BaseConfig):
    class_filter = [1,2]
    alt_loss_pattern = [0,1,2,3,4,5,6,7,8,9]
    num_epochs = 25
    experiment_name = "multi_loss_experiment"

class LowerLearningRate(BaseConfig):
    class_filter = [1,2]
    learning_rate = 1e-4
    experiment_name = "lower_learning_rate_experiment"

class HigherLearningRate(BaseConfig):
    class_filter = [1,2]
    learning_rate = 1e-6
    experiment_name = "higher_learning_rate_experiment"

class SmallBatchConfig(BaseConfig):
    batch_size = 16
    experiment_name = "small_batch_experiment"

class SmallerBatchConfig(BaseConfig):
    batch_size = 8
    experiment_name = "smaller_batch_experiment"

class HighResConfig(BaseConfig):
    target_size = (1280, 1280)
    experiment_name = "high_res_experiment"

class LowResConfig(BaseConfig):
    target_size = (320, 320)
    experiment_name = "low_res_experiment"

class ExtendedEpochConfig(BaseConfig):
    num_epochs = 50
    experiment_name = "extended_epoch_experiment"

class EvenLowerLearningRate(BaseConfig):
    learning_rate = 1e-3
    experiment_name = "even_lower_learning_rate_experiment"

class WeightDecayStandard(BaseConfig):
    weight_decay = 1e-4
    experiment_name = "weight_decay_standard_experiment"

class WeightDecayLow(BaseConfig):
    weight_decay = 1e-5
    experiment_name = "weight_decay_low_experiment"

class WeightDecayHigh(BaseConfig):
    weight_decay = 1e-3
    experiment_name = "weight_decay_high_experiment"

class WeightDecayAggressive(BaseConfig):
    weight_decay = 1e-2
    experiment_name = "weight_decay_aggressive_experiment"

class WeightDecayMinimal(BaseConfig):
    weight_decay = 1e-6
    experiment_name = "weight_decay_minimal_experiment"


class SmallerModel(BaseConfig):
    experiment_name = "smaller_model_experiment"

class EvenSmallerModel(BaseConfig):
    experiment_name = "even_smaller_model_experiment"

class FullDatasetMultiConfig(BaseConfig):
    experiment_name = "full_dataset_multi_experiment"
    class_filter = [1,2,3,4,5,6,7,8,9,10,11,12]
