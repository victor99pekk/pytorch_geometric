"""Unit tests for DatabaseFeatureStore — no real database required."""
import numpy as np
import pytest
import torch

from torch_geometric.data.database_feature_store import LRUFeatureCache
from torch_geometric.data.feature_store import TensorAttr
from torch_geometric.testing import FakeDatabaseFeatureStore


def _make_store(rows=None, cache=None):
    """Return a FakeDatabaseFeatureStore seeded with float32 rows.

    rows: dict mapping nid -> 1-D list/array used for group=None, attr='x'.
    """
    data = {}
    for nid, row in (rows or {}).items():
        data[(None, 'x', nid)] = np.array(row, dtype=np.float32)
    return FakeDatabaseFeatureStore(data=data, cache=cache)


def _attr(nids, group=None, attr_name='x'):
    return TensorAttr(group_name=group, attr_name=attr_name,
                      index=torch.tensor(nids, dtype=torch.long))


def test_multi_get_returns_rows_in_input_nid_order():
    store = _make_store({0: [1., 0.], 1: [0., 1.], 2: [2., 2.]})
    attr = _attr([2, 0, 1])
    result = store._multi_get_tensor([attr])
    assert len(result) == 1
    out = result[0]
    assert out.shape == (3, 2)
    assert torch.allclose(out[0], torch.tensor([2., 2.]))
    assert torch.allclose(out[1], torch.tensor([1., 0.]))
    assert torch.allclose(out[2], torch.tensor([0., 1.]))


def test_multi_get_single_nid():
    store = _make_store({7: [3., 4.]})
    attr = _attr([7])
    result = store._multi_get_tensor([attr])
    assert torch.allclose(result[0], torch.tensor([[3., 4.]]))


def test_multi_get_empty_attrs_returns_empty_list():
    store = _make_store({0: [1., 2.]})
    assert store._multi_get_tensor([]) == []


def test_multi_get_no_records_raises_runtime_error():
    store = FakeDatabaseFeatureStore(data={})
    attr = _attr([99])
    with pytest.raises(RuntimeError, match="Could not determine shape"):
        store._multi_get_tensor([attr])


def test_cache_miss_fetches_from_db():
    store = _make_store(
        {
            0: [1., 0.],
            1: [0., 1.]
        },
        cache=LRUFeatureCache(maxsize=100),
    )
    attr = _attr([0, 1])
    store._multi_get_tensor([attr])
    assert store.fetch_call_count == 1


def test_second_call_hits_cache_entirely():
    store = _make_store(
        {
            0: [1., 0.],
            1: [0., 1.]
        },
        cache=LRUFeatureCache(maxsize=100),
    )
    attr = _attr([0, 1])
    store._multi_get_tensor([attr])
    store._multi_get_tensor([attr])
    assert store.fetch_call_count == 1


def test_partial_cache_hit_narrows_fetch_index():
    """Pre-populate cache for nids 0 and 1; request 0,1,2 — only 2 fetched."""
    cache = LRUFeatureCache(maxsize=100)
    store = _make_store(
        {
            0: [1., 0.],
            1: [0., 1.],
            2: [2., 2.]
        },
        cache=cache,
    )
    # warm cache for 0 and 1
    store._multi_get_tensor([_attr([0, 1])])
    assert store.fetch_call_count == 1

    # now request all three — only 2 should be a DB fetch
    store._multi_get_tensor([_attr([0, 1, 2])])
    assert store.fetch_call_count == 2  # one more fetch, for nid=2 only

    # result still correct
    result = store._multi_get_tensor([_attr([0, 1, 2])])[0]
    assert torch.allclose(result[0], torch.tensor([1., 0.]))
    assert torch.allclose(result[2], torch.tensor([2., 2.]))


def test_put_tensor_invalidates_cache_slice():
    cache = LRUFeatureCache(maxsize=100)
    store = _make_store({0: [1., 0.]}, cache=cache)
    attr = _attr([0])
    first = store._multi_get_tensor([attr])[0]  # fills cache
    assert torch.allclose(first, torch.tensor([[1., 0.]]))
    assert store.fetch_call_count == 1

    # overwrite and invalidate cache
    store._put_tensor(torch.tensor([[9., 9.]]), attr)
    second = store._multi_get_tensor([attr])[0]  # must re-fetch from db
    assert store.fetch_call_count == 2
    assert torch.allclose(second, torch.tensor([[9., 9.]]))


def test_remove_tensor_invalidates_cache():
    cache = LRUFeatureCache(maxsize=100)
    store = _make_store({0: [1., 0.], 1: [0., 1.]}, cache=cache)
    store._multi_get_tensor([_attr([0, 1])])  # fill cache
    store._remove_tensor(_attr([0]))  # drop nid=0 from cache
    store._multi_get_tensor([_attr([0])])  # must re-fetch
    assert store.fetch_call_count == 2


def test_multi_attr_two_attrs_same_group():
    data = {
        (None, 'x', 0): np.array([1., 0.], dtype=np.float32),
        (None, 'y', 0): np.array([3], dtype=np.int64),
    }
    store = FakeDatabaseFeatureStore(data=data)
    attrs = [
        TensorAttr(group_name=None, attr_name='x',
                   index=torch.tensor([0], dtype=torch.long)),
        TensorAttr(group_name=None, attr_name='y',
                   index=torch.tensor([0], dtype=torch.long)),
    ]
    results = store._multi_get_tensor(attrs)
    assert len(results) == 2
    assert torch.allclose(results[0], torch.tensor([[1., 0.]]))
    assert results[1][0, 0].item() == 3


def test_get_all_tensor_attrs_deduplicates():
    data = {
        (None, 'x', 0): np.zeros(2, dtype=np.float32),
        (None, 'x', 1): np.zeros(2, dtype=np.float32),
        (None, 'y', 0): np.zeros(1, dtype=np.int64),
    }
    store = FakeDatabaseFeatureStore(data=data)
    attrs = store.get_all_tensor_attrs()
    attr_names = [(a.group_name, a.attr_name) for a in attrs]
    assert (None, 'x') in attr_names
    assert (None, 'y') in attr_names
    assert len(attrs) == 2  # deduplicated


def test_put_tensor_failure_does_not_invalidate_cache():
    """When _put_tensor_db returns False, cache should not be invalidated."""
    class FailingStore(FakeDatabaseFeatureStore):
        def _put_tensor_db(self, tensor, attr):
            return False  # fail the write

    cache = LRUFeatureCache(maxsize=100)
    data = {(None, 'x', 0): np.array([1., 0.], dtype=np.float32)}
    store = FailingStore(data=data, cache=cache)
    attr = _attr([0])
    
    # Fill cache
    first = store._multi_get_tensor([attr])[0]
    assert torch.allclose(first, torch.tensor([[1., 0.]]))
    assert store.fetch_call_count == 1

    # Attempt write (should fail and NOT invalidate cache)
    result = store._put_tensor(torch.tensor([[2., 2.]]), attr)
    assert result is False

    # Fetch again - should hit cache (no additional fetch)
    second = store._multi_get_tensor([attr])[0]
    assert store.fetch_call_count == 1  # no new fetch
    assert torch.allclose(second, torch.tensor([[1., 0.]]))  # old value still cached


def test_remove_tensor_failure_does_not_invalidate_cache():
    """When _remove_tensor_db returns False, cache should not be invalidated."""
    class FailingStore(FakeDatabaseFeatureStore):
        def _remove_tensor_db(self, attr):
            return False  # fail the remove

    cache = LRUFeatureCache(maxsize=100)
    data = {(None, 'x', 0): np.array([1., 0.], dtype=np.float32)}
    store = FailingStore(data=data, cache=cache)
    attr = _attr([0])
    
    # Fill cache
    store._multi_get_tensor([attr])
    assert store.fetch_call_count == 1

    # Attempt remove (should fail and NOT invalidate cache)
    result = store._remove_tensor(attr)
    assert result is False

    # Fetch again - should hit cache (no additional fetch)
    store._multi_get_tensor([attr])
    assert store.fetch_call_count == 1  # no new fetch


def test_remove_tensor_whole_slice_with_cache():
    """Remove all nids in a slice by passing attr.index=None."""
    cache = LRUFeatureCache(maxsize=100)
    store = _make_store({0: [1., 0.], 1: [0., 1.]}, cache=cache)
    
    # Fill cache
    store._multi_get_tensor([_attr([0, 1])])
    assert store.fetch_call_count == 1

    # Remove whole slice (attr.index=None passes None to cache.invalidate)
    attr_whole_slice = TensorAttr(group_name=None, attr_name='x', index=None)
    store._remove_tensor(attr_whole_slice)

    # Fetch again - should re-fetch (cache was invalidated)
    store._multi_get_tensor([_attr([0, 1])])
    assert store.fetch_call_count == 2


def test_get_tensor_with_single_nid():
    """_get_tensor fetches a single nid and returns tensor."""
    store = _make_store({5: [7., 8.]})
    attr = _attr([5])
    result = store._get_tensor(attr)
    assert torch.allclose(result, torch.tensor([7., 8.]))


def test_lru_cache_evicts_on_max_size():
    """LRUFeatureCache evicts LRU entry when maxsize is exceeded."""
    cache = LRUFeatureCache(maxsize=2)
    
    # Add first entry
    cache.put(None, 'x', {0: np.array([1., 0.], dtype=np.float32)})
    assert len(cache) == 1
    
    # Add second entry
    cache.put(None, 'x', {1: np.array([2., 1.], dtype=np.float32)})
    assert len(cache) == 2
    
    # Add third entry - should evict first
    cache.put(None, 'x', {2: np.array([3., 2.], dtype=np.float32)})
    assert len(cache) == 2
    
    # First entry should be gone
    key0 = (None, 'x', 0)
    assert key0 not in cache


def test_lru_cache_move_to_end_on_hit():
    """LRUFeatureCache moves accessed entry to end (most recently used)."""
    cache = LRUFeatureCache(maxsize=2)
    
    # Add two entries
    row0 = np.array([1., 0.], dtype=np.float32)
    row1 = np.array([2., 1.], dtype=np.float32)
    cache.put(None, 'x', {0: row0})
    cache.put(None, 'x', {1: row1})
    
    # Access entry 0 (moves to end)
    attr = TensorAttr(group_name=None, attr_name='x', 
                     index=torch.tensor([0], dtype=torch.long))
    hits = cache.get(attr)
    assert 0 in hits
    
    # Add entry 2 - should evict entry 1 (LRU), not 0
    cache.put(None, 'x', {2: np.array([3., 2.], dtype=np.float32)})
    assert (None, 'x', 0) in cache
    assert (None, 'x', 1) not in cache


def test_lru_cache_put_existing_key_moves_to_end():
    """LRUFeatureCache moves entry to end when re-putting same key."""
    cache = LRUFeatureCache(maxsize=2)
    
    # Add two entries
    row0 = np.array([1., 0.], dtype=np.float32)
    row1 = np.array([2., 1.], dtype=np.float32)
    cache.put(None, 'x', {0: row0})
    cache.put(None, 'x', {1: row1})
    
    # Re-put entry 0 (moves to end)
    row0_new = np.array([1., 0.], dtype=np.float32)
    cache.put(None, 'x', {0: row0_new})
    
    # Add entry 2 - should evict entry 1 (LRU), not 0
    cache.put(None, 'x', {2: np.array([3., 2.], dtype=np.float32)})
    assert (None, 'x', 0) in cache
    assert (None, 'x', 1) not in cache


def test_lru_cache_invalidate_all_in_slice():
    """LRUFeatureCache.invalidate with nids=None removes all in slice."""
    cache = LRUFeatureCache(maxsize=100)
    
    # Add multiple entries
    cache.put(None, 'x', {0: np.array([1., 0.], dtype=np.float32)})
    cache.put(None, 'x', {1: np.array([2., 1.], dtype=np.float32)})
    cache.put(None, 'y', {0: np.array([3], dtype=np.int64)})
    
    assert len(cache) == 3
    
    # Invalidate all x entries (nids=None)
    cache.invalidate(None, 'x', nids=None)
    
    assert len(cache) == 1  # only y:0 remains
    assert (None, 'x', 0) not in cache
    assert (None, 'x', 1) not in cache
    assert (None, 'y', 0) in cache


def test_lru_cache_clear():
    """LRUFeatureCache.clear removes all entries."""
    cache = LRUFeatureCache(maxsize=100)
    cache.put(None, 'x', {0: np.array([1., 0.], dtype=np.float32)})
    assert len(cache) == 1
    
    cache.clear()
    assert len(cache) == 0
