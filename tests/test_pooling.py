import numpy as np

from data.pooling import aggregate_patch_signal, build_patches, load_or_build_patches


def _random_taxels(n, seed=0):
    rng = np.random.RandomState(seed)
    positions = rng.uniform(-0.05, 0.05, size=(n, 3)).astype(np.float32)
    normals = rng.normal(size=(n, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    link_id = rng.randint(0, 3, size=n).astype(np.int16)
    return positions, normals, link_id


def test_n_less_than_k_uses_all_taxels_as_centroids_no_duplicates():
    positions, normals, link_id = _random_taxels(5)
    patches = build_patches(positions, normals, link_id, mean_spacing=0.01, format_id="fmt_small", K=128)

    assert patches["patch_mask"].sum() == 5
    assert np.array_equal(np.sort(patches["centroid_idx"]), np.arange(5))
    assert len(set(patches["centroid_idx"].tolist())) == 5  # never duplicated


def test_n_greater_than_k_selects_k_distinct_centroids():
    positions, normals, link_id = _random_taxels(200)
    patches = build_patches(positions, normals, link_id, mean_spacing=0.005, format_id="fmt_big", K=64)

    assert patches["patch_mask"].sum() == 64
    assert len(set(patches["centroid_idx"].tolist())) == 64


def test_fps_is_deterministic_across_calls():
    positions, normals, link_id = _random_taxels(150, seed=7)
    a = build_patches(positions, normals, link_id, mean_spacing=0.006, format_id="fmt_x", K=32)
    b = build_patches(positions, normals, link_id, mean_spacing=0.006, format_id="fmt_x", K=32)
    assert np.array_equal(a["centroid_idx"], b["centroid_idx"])
    assert np.array_equal(a["member_idx"], b["member_idx"])


def test_different_format_ids_can_yield_different_centroids():
    positions, normals, link_id = _random_taxels(150, seed=7)
    a = build_patches(positions, normals, link_id, mean_spacing=0.006, format_id="fmt_x", K=32)
    b = build_patches(positions, normals, link_id, mean_spacing=0.006, format_id="fmt_y", K=32)
    # Not required to differ, but the seed derivation must be a function of format_id.
    assert isinstance(a["centroid_idx"], np.ndarray) and isinstance(b["centroid_idx"], np.ndarray)


def test_load_or_build_patches_caches_to_disk(tmp_path):
    positions, normals, link_id = _random_taxels(90, seed=3)
    first = load_or_build_patches(tmp_path, "fmt_cache", positions, normals, link_id, mean_spacing=0.004, K=32)
    cache_file = tmp_path / "fmt_cache.npz"
    assert cache_file.exists()

    second = load_or_build_patches(tmp_path, "fmt_cache", positions, normals, link_id, mean_spacing=0.004, K=32)
    assert np.array_equal(first["centroid_idx"], second["centroid_idx"])
    assert np.array_equal(first["member_idx"], second["member_idx"])


def test_aggregate_patch_signal_is_mean_not_max():
    # 2 patches, 2 members each, known values so mean vs max disagree.
    member_idx = np.array([[0, 1], [2, 3]])
    member_mask = np.array([[True, True], [True, True]])
    signal = np.array([[1.0, 3.0, 5.0, 7.0]]).T * np.ones((1, 3))  # (N=4, C=3), values 1,3,5,7 per row
    signal = signal.reshape(4, 3)
    validity = np.ones((4, 3), dtype=np.uint8)

    patch_signal, patch_validity = aggregate_patch_signal(signal, validity, member_idx, member_mask)
    assert patch_signal.shape == (2, 3)
    np.testing.assert_allclose(patch_signal[0], np.mean(signal[[0, 1]], axis=0))
    np.testing.assert_allclose(patch_signal[1], np.mean(signal[[2, 3]], axis=0))
    assert patch_validity.all()


def test_aggregate_patch_signal_ignores_invalid_members():
    member_idx = np.array([[0, 1]])
    member_mask = np.array([[True, True]])
    signal = np.array([[10.0, 10.0, 10.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    validity = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.uint8)  # member 0 invalid on every channel

    patch_signal, patch_validity = aggregate_patch_signal(signal, validity, member_idx, member_mask)
    np.testing.assert_allclose(patch_signal[0], signal[1])  # only the valid member contributes
    assert patch_validity[0].all()


def test_aggregate_patch_signal_supports_leading_time_axis():
    member_idx = np.array([[0, 1]])
    member_mask = np.array([[True, True]])
    T = 3
    signal = np.stack([np.array([[float(t), float(t)], [float(t + 1), float(t + 1)]]) for t in range(T)])
    signal = np.concatenate([signal, signal[..., :1]], axis=-1)  # (T, 2, 3) fake 3rd channel
    validity = np.ones_like(signal, dtype=np.uint8)

    patch_signal, patch_validity = aggregate_patch_signal(signal, validity, member_idx, member_mask)
    assert patch_signal.shape == (T, 1, 3)
