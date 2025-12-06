# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

"""
Tests for optimized triu/tril on sharded DTensor.

For sharded tensors, triu/tril can be computed locally by adjusting the
diagonal offset: adjusted_k = k + row_offset - col_offset. This avoids
the all-gather that the default replicate strategy would require.
"""

import torch
from torch.distributed.tensor import (
    distribute_tensor,
    Replicate,
    Shard,
)
from torch.distributed.tensor.debug import CommDebugMode
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class TestTriuTrilOptimized(DTensorTestBase):
    """Tests for triu/tril with diagonal offset adjustment."""

    @property
    def world_size(self):
        return 4

    def _create_global_tensor(self, *shape):
        torch.manual_seed(42)
        return torch.randn(*shape, device=self.device_type)

    @with_comms
    def test_triu_sharded(self):
        device_mesh = self.build_device_mesh()
        global_tensor = self._create_global_tensor(8, 8)

        for shard_dim in [0, 1]:
            with self.subTest(shard_dim=shard_dim):
                dt = distribute_tensor(global_tensor, device_mesh, [Shard(shard_dim)])
                result = torch.triu(dt)
                self.assertEqual(result.full_tensor(), torch.triu(global_tensor))
                self.assertEqual(result.placements, (Shard(shard_dim),))

    @with_comms
    def test_tril_sharded(self):
        device_mesh = self.build_device_mesh()
        global_tensor = self._create_global_tensor(8, 8)

        for shard_dim in [0, 1]:
            with self.subTest(shard_dim=shard_dim):
                dt = distribute_tensor(global_tensor, device_mesh, [Shard(shard_dim)])
                result = torch.tril(dt)
                self.assertEqual(result.full_tensor(), torch.tril(global_tensor))
                self.assertEqual(result.placements, (Shard(shard_dim),))

    @with_comms
    def test_triu_tril_with_offset(self):
        device_mesh = self.build_device_mesh()
        global_tensor = self._create_global_tensor(8, 8)

        for op in [torch.triu, torch.tril]:
            for shard_dim in [0, 1]:
                dt = distribute_tensor(global_tensor, device_mesh, [Shard(shard_dim)])
                for k in [-7, -3, -1, 0, 1, 3, 7]:
                    with self.subTest(op=op.__name__, shard_dim=shard_dim, k=k):
                        result = op(dt, diagonal=k)
                        expected = op(global_tensor, diagonal=k)
                        self.assertEqual(result.full_tensor(), expected)

    @with_comms
    def test_triu_tril_no_allgather(self):
        """Verify triu/tril avoid all-gather on sharded tensors."""
        device_mesh = self.build_device_mesh()
        global_tensor = self._create_global_tensor(8, 8)

        for shard_dim in [0, 1]:
            for op in [torch.triu, torch.tril]:
                with self.subTest(op=op.__name__, shard_dim=shard_dim):
                    dt = distribute_tensor(
                        global_tensor, device_mesh, [Shard(shard_dim)]
                    )
                    with CommDebugMode() as comm_mode:
                        _ = op(dt)
                    comm_counts = comm_mode.get_comm_counts()
                    self.assertEqual(
                        comm_counts.get("all_gather_into_tensor", 0), 0
                    )
                    self.assertEqual(comm_counts.get("all_gather", 0), 0)

    @with_comms
    def test_triu_batched(self):
        """triu on 3D tensor sharded on batch or matrix dims."""
        device_mesh = self.build_device_mesh()
        global_tensor = self._create_global_tensor(4, 8, 8)

        for shard_dim in [0, 1]:
            with self.subTest(shard_dim=shard_dim):
                dt = distribute_tensor(global_tensor, device_mesh, [Shard(shard_dim)])
                result = torch.triu(dt)
                self.assertEqual(result.full_tensor(), torch.triu(global_tensor))

    @with_comms
    def test_triu_non_square(self):
        """triu works on non-square matrices too."""
        device_mesh = self.build_device_mesh()
        global_tensor = self._create_global_tensor(8, 12)
        dt = distribute_tensor(global_tensor, device_mesh, [Shard(0)])
        result = torch.triu(dt)
        self.assertEqual(result.full_tensor(), torch.triu(global_tensor))

    @with_comms
    def test_triu_replicated(self):
        """triu on replicated tensor works without optimization."""
        device_mesh = self.build_device_mesh()
        global_tensor = self._create_global_tensor(8, 8)
        dt = distribute_tensor(global_tensor, device_mesh, [Replicate()])
        result = torch.triu(dt)
        self.assertEqual(result.full_tensor(), torch.triu(global_tensor))


if __name__ == "__main__":
    run_tests()
