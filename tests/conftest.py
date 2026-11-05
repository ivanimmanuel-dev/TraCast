import pytest
import torch


@pytest.fixture(autouse=True)
def restore_torch_global_state():
    """`seed_all` flips process-wide switches; keep one test from leaking into the next."""
    deterministic = torch.are_deterministic_algorithms_enabled()
    mkldnn = torch.backends.mkldnn.enabled
    threads = torch.get_num_threads()
    yield
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.mkldnn.enabled = mkldnn
    torch.set_num_threads(threads)
