from torch.utils.data.dataloader import DataLoader
from bayesian_meta_learning.benchmark.benchmark_dataset import BenchmarkDataset
from bayesian_meta_learning.benchmark.sinusoid_affine_benchmark import SinusoidAffineBenchmark
from mtutils.mtutils import BM_DICT

def create_benchmark_dataloaders(config: dict):
    bm_meta, bm_val, bm_test = create_benchmarks(config)
    train_data_loader = DataLoader(BenchmarkDataset(
        bm_meta, config['normalize_benchmark']), shuffle=True)
    val_data_loader = DataLoader(BenchmarkDataset(
        bm_val,  config['normalize_benchmark']), shuffle=True)
    test_data_loader = DataLoader(BenchmarkDataset(
        bm_test,  config['normalize_benchmark']), shuffle=True)
    return train_data_loader, val_data_loader, test_data_loader


def create_benchmarks(config: dict):
    # extend benchmark dict
    BM_DICT['SinusoidAffine1D'] = SinusoidAffineBenchmark
    # create benchmarks
    bm_meta = BM_DICT[config["benchmark"]](
        n_task=config["num_episodes_per_epoch"],
        n_datapoints_per_task=config["num_points_per_train_task"],
        output_noise=config["noise_stddev"],
        seed_task=config["seed_offset"],
        seed_x=config["seed_offset"] + 1,
        seed_noise=config["seed_offset"] + 2,
    )
    bm_val = BM_DICT[config["benchmark"]](
        n_task=config["num_episodes"],
        n_datapoints_per_task=config["num_points_per_train_task"],
        output_noise=config["noise_stddev"],
        seed_task=config["seed_offset"] + 10,
        seed_x=config["seed_offset"] + 20,
        seed_noise=config["seed_offset"] + 30,
    )
    bm_test = BM_DICT[config["benchmark"]](
        n_task=config["num_test_tasks"],
        n_datapoints_per_task=config["num_points_per_test_task"],
        output_noise=config["noise_stddev"],
        seed_task=config["seed_offset_test"],
        seed_x=config["seed_offset_test"] + 1,
        seed_noise=config["seed_offset_test"] + 2,
    )
    return bm_meta, bm_val, bm_test