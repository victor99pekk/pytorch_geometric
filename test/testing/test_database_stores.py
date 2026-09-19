"""Unit tests for fake database store implementations in testing module."""
import numpy as np
import torch

from torch_geometric.data.feature_store import TensorAttr
from torch_geometric.data.graph_store import EdgeAttr, EdgeLayout
from torch_geometric.sampler.base import EdgeSamplerInput
from torch_geometric.testing import (
    FakeDatabaseFeatureStore,
    FakeDatabaseGraphStore,
    FakeDatabaseSampler,
)


def _attr(nids, group=None, attr_name='x'):
    """Helper to create TensorAttr with given node IDs."""
    return TensorAttr(
        group_name=group,
        attr_name=attr_name,
        index=torch.tensor(nids, dtype=torch.long),
    )


def _attr_slice(start, stop, step=1, group=None, attr_name='x'):
    """Helper to create TensorAttr with slice index."""
    return TensorAttr(
        group_name=group,
        attr_name=attr_name,
        index=slice(start, stop, step),
    )


def test_fetch_with_slice_index():
    """_fetch_remote_attrs handles slice(start, stop, step) correctly."""
    data = {
        (None, 'x', 0): np.array([1., 0.], dtype=np.float32),
        (None, 'x', 1): np.array([2., 1.], dtype=np.float32),
        (None, 'x', 2): np.array([3., 2.], dtype=np.float32),
        (None, 'x', 3): np.array([4., 3.], dtype=np.float32),
    }
    store = FakeDatabaseFeatureStore(data=data)

    attr = _attr_slice(1, 3)
    records, fetched_nids = store._fetch_remote_attrs(attr)

    assert fetched_nids == [1, 2]
    assert len(records) == 2
    assert records[0]['id'] == 1
    assert records[1]['id'] == 2


def test_fetch_with_slice_step():
    """_fetch_remote_attrs handles slice with step correctly."""
    data = {
        (None, 'x', 0): np.array([1., 0.], dtype=np.float32),
        (None, 'x', 1): np.array([2., 1.], dtype=np.float32),
        (None, 'x', 2): np.array([3., 2.], dtype=np.float32),
        (None, 'x', 3): np.array([4., 3.], dtype=np.float32),
    }
    store = FakeDatabaseFeatureStore(data=data)

    attr = _attr_slice(0, 4, 2)
    records, fetched_nids = store._fetch_remote_attrs(attr)

    assert fetched_nids == [0, 2]
    assert len(records) == 2


def test_put_with_slice_index():
    """_put_tensor_db handles slice index correctly."""
    store = FakeDatabaseFeatureStore(data={})

    attr = _attr_slice(1, 3)
    tensor = torch.tensor([[2., 1.], [3., 2.]], dtype=torch.float32)
    result = store._put_tensor_db(tensor, attr)

    assert result is True
    assert (None, 'x', 1) in store._data
    assert (None, 'x', 2) in store._data
    assert np.allclose(store._data[(None, 'x', 1)], [2., 1.])
    assert np.allclose(store._data[(None, 'x', 2)], [3., 2.])


def test_fetch_with_scalar_index():
    """_fetch_remote_attrs handles scalar index (single nid)."""
    data = {(None, 'x', 5): np.array([1., 0.], dtype=np.float32)}
    store = FakeDatabaseFeatureStore(data=data)

    attr = TensorAttr(group_name=None, attr_name='x', index=5)
    records, fetched_nids = store._fetch_remote_attrs(attr)

    assert fetched_nids == [5]
    assert len(records) == 1
    assert records[0]['id'] == 5


def test_put_with_scalar_index():
    """_put_tensor_db handles scalar index correctly."""
    store = FakeDatabaseFeatureStore(data={})

    attr = TensorAttr(group_name=None, attr_name='x', index=7)
    tensor = torch.tensor([[2., 1.]], dtype=torch.float32)
    result = store._put_tensor_db(tensor, attr)

    assert result is True
    assert (None, 'x', 7) in store._data
    assert np.allclose(store._data[(None, 'x', 7)], [2., 1.])


def test_get_tensor_size_returns_shape():
    """_get_tensor_size returns shape of fetched tensor."""
    data = {(None, 'x', 0): np.array([1., 0., 2.], dtype=np.float32)}
    store = FakeDatabaseFeatureStore(data=data)
    attr = _attr([0])

    result = store._get_tensor_size(attr)
    assert result == (1, 3)


def test_put_edge_index_returns_true():
    """_put_edge_index stub returns True."""
    store = FakeDatabaseGraphStore()
    edge_attr = EdgeAttr(edge_type=('paper', 'cites', 'paper'), layout=EdgeLayout.COO)
    result = store._put_edge_index(torch.zeros((2, 0)), edge_attr)
    assert result is True


def test_get_edge_index_returns_none():
    """_get_edge_index stub returns None."""
    store = FakeDatabaseGraphStore()
    edge_attr = EdgeAttr(edge_type=('paper', 'cites', 'paper'), layout=EdgeLayout.COO)
    result = store._get_edge_index(edge_attr)
    assert result is None


def test_remove_edge_index_returns_true():
    """_remove_edge_index stub returns True."""
    store = FakeDatabaseGraphStore()
    edge_attr = EdgeAttr(edge_type=('paper', 'cites', 'paper'), layout=EdgeLayout.COO)
    result = store._remove_edge_index(edge_attr)
    assert result is True


def test_get_all_edge_attrs_returns_empty():
    """get_all_edge_attrs returns empty list."""
    store = FakeDatabaseGraphStore()
    result = store.get_all_edge_attrs()
    assert result == []


def test_build_edge_sampling_query():
    """_build_edge_sampling_query returns expected query string."""
    gs = FakeDatabaseGraphStore()
    sampler = FakeDatabaseSampler(gs)
    query = sampler._build_edge_sampling_query()
    assert query == "FAKE_EDGE_QUERY"


def test_build_edge_query_params():
    """_build_edge_query_params builds params dict from seeds."""
    gs = FakeDatabaseGraphStore()
    sampler = FakeDatabaseSampler(gs)
    seeds = torch.tensor([10, 20, 30], dtype=torch.long)
    params = sampler._build_edge_query_params(seeds)

    assert params == {"seed_ids": [10, 20, 30]}


def test_decode_node_sampling_record():
    """_decode_node_sampling_record returns None (expected behavior)."""
    gs = FakeDatabaseGraphStore()
    sampler = FakeDatabaseSampler(gs)
    record = {"some": "data"}
    seeds = torch.tensor([1], dtype=torch.long)
    result = sampler._decode_node_sampling_record(record, seeds)
    assert result is None


def test_decode_edge_sampling_record():
    """_decode_edge_sampling_record returns None (expected behavior)."""
    gs = FakeDatabaseGraphStore()
    sampler = FakeDatabaseSampler(gs)
    record = {"some": "data"}
    seeds = torch.tensor([1], dtype=torch.long)
    result = sampler._decode_edge_sampling_record(record, seeds)
    assert result is None


def test_sample_from_edges_works_with_fake_sampler():
    """FakeDatabaseSampler.sample_from_edges uses build_edge_query_params."""
    gs = FakeDatabaseGraphStore(records={
        "FAKE_EDGE_QUERY": {"nodes": [0, 1], "edges": [[0, 1]]}
    })
    sampler = FakeDatabaseSampler(gs)

    edge_input = EdgeSamplerInput(
        input_id=None,
        row=torch.tensor([0, 1], dtype=torch.long),
        col=torch.tensor([1, 0], dtype=torch.long),
    )

    out = sampler.sample_from_edges(edge_input)

    assert len(gs.executed_queries) > 0
    query, params = gs.executed_queries[-1]
    assert query == "FAKE_EDGE_QUERY"
    assert "seed_ids" in params
    assert out is not None
