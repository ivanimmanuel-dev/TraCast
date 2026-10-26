"""Graph algebra and data-alignment checks use small explicit fixtures."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
import torch

from tracast.topology import adjacency, connected_subset, validate_alignment
from tracast.graph_model import GraphGRU, normalized
from tracast.benchmark import BenchmarkConfig, model_for
from tracast.resources import peak_memory


@pytest.fixture
def distances():
    return pd.DataFrame({"from":["a","a","b","b","c"],"to":["a","b","b","c","c"],"cost":[0.,1.,0.,2.,0.]})


def test_real_distance_kernel_and_direction(distances):
    weights,meta=adjacency(distances,["a","b","c"],threshold=.001)
    scale=np.std([0.,1.,0.,2.,0.])
    assert weights[0,1] == pytest.approx(np.exp(-(1/scale)**2))
    assert weights[1,0] == 0
    assert np.diag(weights).sum() == 0
    assert meta["directed_edges"] == 2


@pytest.mark.parametrize("kind",["negative","infinite","duplicate","schema","missing_endpoint"])
def test_invalid_distances_fail(distances,kind):
    if kind=="negative":
        distances.loc[0,"cost"]=-1
    elif kind=="infinite":
        distances.loc[0,"cost"]=np.inf
    elif kind=="duplicate":
        distances=pd.concat([distances,distances.iloc[:1]],ignore_index=True)
    elif kind=="schema":
        distances=distances.rename(columns={"cost":"made_up"})
    else:
        distances.loc[0,"to"]=None
    with pytest.raises(ValueError):
        adjacency(distances,["a","b","c"])


def test_alignment_and_permutation():
    assert validate_alignment(["b","a"],["a","b"]) == {"a":1,"b":0}
    for columns,ids in [(["a","a"],["a","b"]),(["a","b"],["a","c"])]:
        with pytest.raises(ValueError):
            validate_alignment(columns,ids)


def test_subset_uses_actual_edges():
    graph=np.array([[0,.8,0,0],[0,0,.9,0],[0,0,0,0],[0,0,0,0]],dtype=np.float32)
    ids,weights=connected_subset(graph,["a","b","c","d"],3)
    assert ids==["a","b","c"]
    np.testing.assert_array_equal(weights,graph[:3,:3])
    with pytest.raises(ValueError,match="component"):
        connected_subset(graph,["a","b","c","d"],4)


def test_neighbor_features_respect_edge_orientation():
    weights=torch.tensor([[0.,2.,0.],[0.,0.,1.],[0.,0.,0.]])
    model=GraphGRU(weights,hidden=4,horizon=2)
    features=model.spatial_features(torch.tensor([[[10.,20.,30.]]]))
    torch.testing.assert_close(features[0,0,:,0],torch.tensor([10.,20.,30.]))
    torch.testing.assert_close(features[0,0,:,1],torch.tensor([0.,10.,20.]))
    torch.testing.assert_close(features[0,0,:,2],torch.tensor([20.,30.,0.]))


def test_changing_edges_changes_predictions_with_identical_parameters():
    torch.manual_seed(42)
    first=GraphGRU(torch.tensor([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]]),hidden=4,horizon=2)
    second=GraphGRU(torch.zeros(3,3),hidden=4,horizon=2)
    with torch.no_grad():
        for left,right in zip(first.parameters(),second.parameters()):
            right.copy_(left)
    x=torch.randn(2,12,3)
    assert not torch.allclose(first(x),second(x),atol=1e-6)
    assert first(x).shape == (2,2,3)
    first(x).square().mean().backward()
    assert first.spatial.weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in first.parameters())


def test_node_permutation_aligns_graph_features():
    graph=torch.tensor([[0.,1.,.4],[0.,0.,1.],[1.,0.,0.]])
    x=torch.tensor([[[1.,3.,9.]]])
    perm=torch.tensor([2,0,1])
    original=GraphGRU(graph).spatial_features(x)
    reordered=GraphGRU(graph[perm][:,perm]).spatial_features(x[:,:,perm])
    torch.testing.assert_close(reordered,original[:,:,perm,:])


@pytest.mark.parametrize("weights",[torch.eye(2),torch.tensor([[0.,-1.],[0.,0.]]),torch.ones(2,3)])
def test_invalid_graph_operator(weights):
    with pytest.raises(ValueError):
        normalized(weights)


@pytest.mark.parametrize("changes",[{"seeds":[]},{"seeds":[42,42]},{"models":["fake"]},{"threshold":0},{"epochs":0}])
def test_benchmark_configuration(changes):
    with pytest.raises(ValueError):
        replace(BenchmarkConfig(),**changes).validate()


def test_self_ablation_has_same_trainable_capacity():
    config=BenchmarkConfig(sensors=3)
    graph=np.array([[0,1,0],[0,0,1],[1,0,0]],dtype=np.float32)
    real=model_for("graph_gru",config,graph)
    removed=model_for("graph_self",config,graph)
    assert sum(p.numel() for p in real.parameters())==sum(p.numel() for p in removed.parameters())
    assert real.incoming.count_nonzero() > 0 and removed.incoming.count_nonzero() == 0


def test_process_peak_memory_available():
    result=peak_memory()
    assert result["bytes"] > 0
    assert "process" in result["method"]


def test_lstm_forecast_shape_gradients_and_capacity():
    from tracast.recurrent import LSTM
    model=LSTM(16,32,12)
    output=model(torch.randn(4,12,16))
    assert output.shape==(4,12,16)
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(p.numel() for p in model.parameters())==12736


def test_lstm_factory_and_repeated_initialization():
    from tracast.experiment import seed_all
    config=BenchmarkConfig()
    seed_all(42,1)
    first=model_for("lstm",config,np.zeros((16,16),dtype=np.float32))
    seed_all(42,1)
    second=model_for("lstm",config,np.zeros((16,16),dtype=np.float32))
    x=torch.ones(2,12,16)
    assert torch.equal(first(x),second(x))
