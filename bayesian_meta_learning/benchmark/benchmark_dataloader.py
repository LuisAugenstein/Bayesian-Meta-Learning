from random import shuffle
from torch.utils.data.dataloader import DataLoader
from bayesian_meta_learning.benchmark.benchmark_dataset import BenchmarkDataset
from bayesian_meta_learning.benchmark.sinusoid_affine_benchmark import SinusoidAffineBenchmark
from mtutils.mtutils import BM_DICT
from metalearning_benchmarks.benchmarks.util import normalize_benchmark

def create_benchmark_dataloaders(config: dict):
    bm_meta, bm_val, bm_test = _create_benchmarks(config)
    train_data_loader = DataLoader(BenchmarkDataset(bm_meta), shuffle=True)
    val_data_loader = DataLoader(BenchmarkDataset(bm_val), shuffle=True)
    test_data_loader = DataLoader(BenchmarkDataset(bm_test), shuffle=True)
    return train_data_loader, val_data_loader, test_data_loader
    
def _create_benchmarks(config: dict):
    # extend benchmark dict
    BM_DICT['SinusoidAffine1D'] = SinusoidAffineBenchmark
    # create benchmarks
    bm_meta = BM_DICT[config["benchmark"]](
        n_task=config["num_train_tasks"],
        n_datapoints_per_task=config["num_points_per_train_task"],
        output_noise=config["noise_stddev"],
        seed_task=config["seed_offset"],
        seed_x=config["seed_offset"] + 1,
        seed_noise=config["seed_offset"] + 2,
    )
    bm_val = BM_DICT[config["benchmark"]](
        n_task=config["num_validation_tasks"],
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
    if config['normalize_benchmark']:
        if bm_meta.n_task > 0:
            bm_meta = normalize_benchmark(bm_meta)
        if bm_val.n_task > 0:
            bm_val = normalize_benchmark(bm_val)
        if bm_test.n_task > 0:
            bm_test = normalize_benchmark(bm_test)
    return bm_meta, bm_val, bm_test


# below are two unused functions that michael used in his implementation
# def _prepare_benchmark(bm: MetaLearningBenchmark, n_points_pred: int, n_task: int):
#     x, y = collate_data(bm)
#     bounds = bm.x_bounds[0, :]
#     lower = bounds[0] - 0.1 * (bounds[1] - bounds[0])
#     higher = bounds[1] + 0.1 * (bounds[1] - bounds[0])
#     x_pred = np.linspace(lower, higher, n_points_pred)[
#         None, :, None].repeat(n_task, axis=0)
#     return x, y, x_pred


# def create_extracted_benchmarks(config: dict):
#     bm_meta, bm_test = create_benchmarks(config)
#     x_meta, y_meta, x_pred_meta = _prepare_benchmark(
#         bm_meta, config['n_points_pred'], config['minibatch'])
#     x_test, y_test, x_pred_test = _prepare_benchmark(
#         bm_test, config['n_points_pred'], config['minibatch_test'])
#     return x_meta, y_meta, x_test, y_test, x_pred_meta, x_pred_test
