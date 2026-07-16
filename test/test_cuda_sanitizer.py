# Owner(s): ["module: cuda"]

import sys
import traceback

import torch
import torch.cuda._sanitizer as csan
from torch.cuda._sanitizer import DataPtr, EventId, StreamId
from torch.testing._internal.common_utils import NoTest, run_tests, TEST_CUDA, TestCase
from torch.testing._internal.two_tensor import TwoTensor


if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)
    TestCase = NoTest


class TestArgumentHandler(TestCase):
    def test_add(self):
        add_func = torch.ops.aten.add.Tensor
        a = torch.ones(5, 3, device="cuda")
        b = torch.randn(5, 3, device="cuda")

        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(add_func._schema, (a, b), {}, is_factory=False)
        c = torch.add(a, b)
        argument_handler.parse_outputs(add_func._schema, c, is_factory=False)

        self.assertEqual({a.data_ptr(), b.data_ptr()}, argument_handler.dataptrs_read)
        self.assertEqual({c.data_ptr()}, argument_handler.dataptrs_written)

    def test_cat(self):
        cat_func = torch.ops.aten.cat.default
        a = torch.ones(2, 4, 5, device="cuda")
        b = torch.zeros(2, 1, 5, device="cuda")
        c = torch.rand(2, 7, 5, device="cuda")

        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(
            cat_func._schema, ([a, b, c], 1), {}, is_factory=False
        )
        d = torch.cat((a, b, c), dim=1)
        argument_handler.parse_outputs(cat_func._schema, d, is_factory=False)

        self.assertEqual(
            {a.data_ptr(), b.data_ptr(), c.data_ptr()}, argument_handler.dataptrs_read
        )
        self.assertEqual({d.data_ptr()}, argument_handler.dataptrs_written)

    def test_split(self):
        split_func = torch.ops.aten.split.Tensor
        a = torch.arange(10, device="cuda").reshape(5, 2)

        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(split_func._schema, (a, 2), {}, is_factory=False)
        out = torch.split(a, 2)
        argument_handler.parse_outputs(split_func._schema, out, is_factory=False)

        # Split is a view op, no data is read or written!
        self.assertEqual(len(argument_handler.dataptrs_read), 0)
        self.assertEqual(len(argument_handler.dataptrs_written), 0)

    def test_inplace(self):
        add_inplace_func = torch.ops.aten.add_.Tensor
        a = torch.rand(4, 2, device="cuda")

        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(
            add_inplace_func._schema, (a, 5), {}, is_factory=False
        )
        a.add_(5)
        argument_handler.parse_outputs(add_inplace_func._schema, a, is_factory=False)

        self.assertEqual(set(), argument_handler.dataptrs_read)
        self.assertEqual({a.data_ptr()}, argument_handler.dataptrs_written)

    def test_out(self):
        mul_out_func = torch.ops.aten.mul.out
        a = torch.arange(8, device="cuda")
        b = torch.empty(8, device="cuda")

        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(
            mul_out_func._schema, (a, 3), {"out": b}, is_factory=False
        )
        torch.mul(a, 3, out=b)
        argument_handler.parse_outputs(mul_out_func._schema, b, is_factory=False)

        self.assertEqual({a.data_ptr()}, argument_handler.dataptrs_read)
        self.assertEqual({b.data_ptr()}, argument_handler.dataptrs_written)

    def test_nonzero(self):
        nonzero_func = torch.ops.aten.nonzero.default
        a = torch.ones(5, 3, 2, device="cuda")

        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(
            nonzero_func._schema, (a,), {"as_tuple": True}, is_factory=False
        )
        out = torch.nonzero(a, as_tuple=True)
        argument_handler.parse_outputs(nonzero_func._schema, out, is_factory=False)

        outputs = {out[0].data_ptr(), out[1].data_ptr(), out[2].data_ptr()}
        self.assertEqual({a.data_ptr()}, argument_handler.dataptrs_read)
        self.assertEqual(outputs, argument_handler.dataptrs_written)

    def test_tensor_names(self):
        addr_func = torch.ops.aten.addr.default
        vec = torch.arange(1, 4, device="cuda")
        M = torch.zeros(3, 3, device="cuda")

        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(
            addr_func._schema, (M, vec, vec), {}, is_factory=False
        )
        out = torch.addr(M, vec, vec)
        argument_handler.parse_outputs(addr_func._schema, out, is_factory=False)

        self.assertEqual(
            argument_handler.tensor_aliases,
            {
                M.data_ptr(): ["self"],
                vec.data_ptr(): ["vec1", "vec2"],
                out.data_ptr(): [],
            },
        )
        self.assertEqual({out.data_ptr()}, argument_handler.outputs)


def tensor_id(i: int) -> DataPtr:
    return i


def stream_id(i: int) -> StreamId:
    return 1000 + i


def event_id(i: int) -> EventId:
    return 2000 + i


BLOCK_SIZE = 1024


class TestEventHandler(TestCase):
    def setUp(self):
        super().setUp()
        self.handler = csan.EventHandler()

    def kernel_launch(
        self,
        stream: StreamId,
        read_only: list[DataPtr] | None = None,
        read_write: list[DataPtr] | None = None,
    ) -> list[csan.SynchronizationError]:
        if read_only is None:
            read_only = []
        if read_write is None:
            read_write = []
        return self.handler._handle_kernel_launch(
            stream,
            read_only,
            read_write,
            {},
            "",
            {k: [""] for k in read_only + read_write},
        )

    def assert_good_kernel_launch(
        self,
        stream: StreamId,
        read_only: list[DataPtr] | None = None,
        read_write: list[DataPtr] | None = None,
    ) -> None:
        self.assertEqual(self.kernel_launch(stream, read_only, read_write), [])

    def assert_bad_kernel_launch(
        self,
        number_of_errors: int,
        stream: StreamId,
        read_only: list[DataPtr] | None = None,
        read_write: list[DataPtr] | None = None,
    ) -> None:
        errors = self.kernel_launch(stream, read_only, read_write)
        self.assertEqual(len(errors), number_of_errors)

    def test_empty_kernel_launch(self):
        self.assert_good_kernel_launch(stream_id(0))

    def test_simple_passing(self):
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(1)])

    def test_simple_error(self):
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        self.assert_bad_kernel_launch(1, stream_id(2), read_write=[tensor_id(1)])

    def test_simple_sync(self):
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        self.handler._handle_event_record(event_id(0), stream_id(1))
        self.handler._handle_event_wait(event_id(0), stream_id(2))
        self.assert_good_kernel_launch(stream_id(2), read_write=[tensor_id(1)])

    def test_reads_check_last_write(self):
        # Tests that not only the first read operation checks if it is in conflict
        # with the last write operation, but all read operations do.

        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_event_record(event_id(0), stream_id(1))
        self.handler._handle_event_wait(event_id(0), stream_id(2))
        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(1)])

        self.assert_bad_kernel_launch(1, stream_id(3), read_only=[tensor_id(1)])

    def test_branch_sync(self):
        # Tests that two streams can read after both waiting for a third, but they
        # cannot write without further synchronization.

        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_event_record(event_id(0), stream_id(1))
        self.handler._handle_event_wait(event_id(0), stream_id(2))
        self.handler._handle_event_wait(event_id(0), stream_id(3))
        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(3), read_only=[tensor_id(1)])

        self.assert_bad_kernel_launch(1, stream_id(2), read_write=[tensor_id(1)])

    def test_chain_sync(self):
        iterations = 10

        self.assert_good_kernel_launch(stream_id(0), read_only=[tensor_id(1)])
        for i in range(iterations):
            self.handler._handle_event_record(event_id(i), stream_id(i))
            self.handler._handle_event_wait(event_id(i), stream_id(i + 1))
        self.assert_good_kernel_launch(stream_id(iterations), read_write=[tensor_id(1)])

    def test_expired_record(self):
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        self.handler._handle_event_record(event_id(0), stream_id(1))
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        self.handler._handle_event_wait(event_id(0), stream_id(2))

        self.assert_bad_kernel_launch(1, stream_id(2), read_write=[tensor_id(1)])

    def test_deleted_record(self):
        for should_delete, should_create in [
            (True, True),
            (True, False),
            (False, True),
        ]:
            self.setUp()
            with self.subTest(should_delete=should_delete, should_create=should_create):
                self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
                self.handler._handle_event_record(event_id(0), stream_id(1))

                if should_delete:
                    self.handler._handle_event_deletion(event_id(0))
                if should_create:
                    self.handler._handle_event_creation(event_id(0))

                self.handler._handle_event_wait(event_id(0), stream_id(2))
                self.assert_bad_kernel_launch(
                    1, stream_id(2), read_write=[tensor_id(1)]
                )

    def test_all_reads_checked_failing(self):
        iterations = 10
        for i in range(1, iterations):
            self.assert_good_kernel_launch(stream_id(i), read_only=[tensor_id(1)])
            self.handler._handle_event_record(event_id(i), stream_id(i))

        for i in range(1, iterations):
            self.handler._handle_event_wait(event_id(i), stream_id(0))

        self.assert_good_kernel_launch(stream_id(iterations), read_only=[tensor_id(1)])
        self.handler._handle_event_record(event_id(iterations), stream_id(i))

        # Does not synchronize with the last read.
        self.assert_bad_kernel_launch(1, stream_id(0), read_write=[tensor_id(1)])

    def test_all_reads_checked_passing(self):
        iterations = 10
        for i in range(1, iterations):
            self.assert_good_kernel_launch(stream_id(i), read_only=[tensor_id(1)])
            self.handler._handle_event_record(event_id(i), stream_id(i))

        for i in range(1, iterations):
            self.handler._handle_event_wait(event_id(i), stream_id(0))

        self.assert_good_kernel_launch(stream_id(0), read_write=[tensor_id(1)])

    def test_multiple_errors(self):
        iterations = 10
        self.assert_good_kernel_launch(
            stream_id(0), read_write=[tensor_id(i) for i in range(iterations)]
        )
        self.assert_bad_kernel_launch(
            iterations,
            stream_id(1),
            read_write=[tensor_id(i) for i in range(iterations)],
        )

    def test_correct_state_merging(self):
        # Tests that after waiting for an event, a stream's state is indeed set
        # to the pointwise maximum of its old state and the recorded state.

        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(2), read_write=[tensor_id(2)])
        self.handler._handle_event_record(event_id(1), stream_id(1))
        self.handler._handle_event_record(event_id(2), stream_id(2))

        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(2), read_write=[tensor_id(2)])
        self.handler._handle_event_wait(event_id(1), stream_id(2))
        self.handler._handle_event_wait(event_id(2), stream_id(1))

        self.handler._handle_event_record(event_id(3), stream_id(2))
        self.handler._handle_event_wait(event_id(3), stream_id(1))
        self.assert_good_kernel_launch(
            stream_id(1), read_write=[tensor_id(1), tensor_id(2)]
        )

    def test_record_override(self):
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(2)])
        self.handler._handle_event_record(event_id(1), stream_id(1))
        self.handler._handle_event_record(event_id(1), stream_id(2))

        self.handler._handle_event_wait(event_id(1), stream_id(3))
        self.assert_bad_kernel_launch(1, stream_id(3), read_write=[tensor_id(1)])

    def test_multiple_wait(self):
        # Tests that a wait operation can be performed multiple times on the same event
        # by different streams.

        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_event_record(event_id(1), stream_id(1))
        self.handler._handle_event_wait(event_id(1), stream_id(2))
        self.handler._handle_event_wait(event_id(1), stream_id(3))

        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(3), read_only=[tensor_id(1)])

    def test_device_synchronize(self):
        # Tests that a device synchronization does correctly cause all streams
        # to synchronize with each other.

        iterations = 10
        for i in range(1, iterations):
            self.assert_good_kernel_launch(stream_id(i), read_write=[tensor_id(i)])

        self.handler._handle_device_synchronization()
        self.assert_good_kernel_launch(
            stream_id(0), read_write=[tensor_id(i) for i in range(1, iterations)]
        )

    def test_device_synchronization_expired(self):
        # Tests that a device synchronization is a one-time synchronization.
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_device_synchronization()
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])

        self.assert_bad_kernel_launch(1, stream_id(2), read_write=[tensor_id(1)])

    def test_new_stream_is_synchronized(self):
        # Tests that after synchronizing operations with the host, any newly created
        # stream is guaranteed to be synchronized with them as well.

        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_device_synchronization()
        self.handler._handle_stream_creation(stream_id(2))
        self.assert_good_kernel_launch(stream_id(2), read_write=[tensor_id(1)])

    def test_stream_synchronize(self):
        # Tests that a stream synchronization does correctly cause all streams to wait
        # for one specific stream, but does not synchronize all streams with each other.

        self.assert_good_kernel_launch(stream_id(0), read_write=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(2)])
        self.handler._handle_stream_synchronization(stream_id(0))

        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(3), read_only=[tensor_id(1)])
        self.assert_bad_kernel_launch(1, stream_id(4), read_only=[tensor_id(2)])

    def test_event_synchronize(self):
        # Tests that an event synchronization does correctly cause all streams to wait
        # for a recorded event, but does not guarantee synchronization with the current
        # state of the stream that recorded the event.

        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_event_record(event_id(1), stream_id(1))
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(2)])

        self.handler._handle_event_synchronization(event_id(1))
        self.assert_good_kernel_launch(stream_id(2), read_write=[tensor_id(1)])
        self.assert_bad_kernel_launch(1, stream_id(2), read_write=[tensor_id(2)])

    def test_allocator_reuse_race(self):
        self.handler._handle_memory_allocation(tensor_id(1), BLOCK_SIZE)
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_memory_deallocation(tensor_id(1), BLOCK_SIZE)
        self.handler._handle_memory_allocation(tensor_id(1), BLOCK_SIZE)
        errors = self.kernel_launch(stream_id(2), read_write=[tensor_id(1)])
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], csan.AllocatorReuseRaceError)

    def test_allocator_reuse_same_stream_safe(self):
        self.handler._handle_memory_allocation(tensor_id(1), BLOCK_SIZE)
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_memory_deallocation(tensor_id(1), BLOCK_SIZE)
        self.handler._handle_memory_allocation(tensor_id(1), BLOCK_SIZE)
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])

    def test_allocator_reuse_with_record_stream(self):
        self.handler._handle_memory_allocation(tensor_id(1), BLOCK_SIZE)
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_record_stream(tensor_id(1), stream_id(1))
        self.handler._handle_memory_deallocation(tensor_id(1), BLOCK_SIZE)
        self.handler._handle_memory_allocation(tensor_id(1), BLOCK_SIZE)
        self.assert_good_kernel_launch(stream_id(2), read_write=[tensor_id(1)])

    def test_allocator_reuse_partial_record_stream(self):
        self.handler._handle_memory_allocation(tensor_id(1), BLOCK_SIZE)
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        self.handler._handle_event_record(event_id(0), stream_id(1))
        self.handler._handle_event_wait(event_id(0), stream_id(2))
        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(1)])
        self.handler._handle_record_stream(tensor_id(1), stream_id(1))
        self.handler._handle_memory_deallocation(tensor_id(1), BLOCK_SIZE)
        self.handler._handle_memory_allocation(tensor_id(1), BLOCK_SIZE)
        errors = self.kernel_launch(stream_id(3), read_write=[tensor_id(1)])
        reuse_errors = [
            e for e in errors if isinstance(e, csan.AllocatorReuseRaceError)
        ]
        self.assertEqual(len(reuse_errors), 1)

    def test_allocator_reuse_with_explicit_sync(self):
        self.handler._handle_memory_allocation(tensor_id(1), BLOCK_SIZE)
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_event_record(event_id(0), stream_id(1))
        self.handler._handle_memory_deallocation(tensor_id(1), BLOCK_SIZE)
        self.handler._handle_memory_allocation(tensor_id(1), BLOCK_SIZE)
        self.handler._handle_event_wait(event_id(0), stream_id(2))
        self.assert_good_kernel_launch(stream_id(2), read_write=[tensor_id(1)])

    def test_allocator_reuse_cleanup_on_device_sync(self):
        self.handler._handle_memory_allocation(tensor_id(1), BLOCK_SIZE)
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_memory_deallocation(tensor_id(1), BLOCK_SIZE)
        self.handler._handle_device_synchronization()
        self.handler._handle_memory_allocation(tensor_id(1), BLOCK_SIZE)
        self.assert_good_kernel_launch(stream_id(2), read_write=[tensor_id(1)])

    def test_allocator_reuse_prior_reads_checked_by_write(self):
        self.handler._handle_memory_allocation(tensor_id(1), BLOCK_SIZE)
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        self.handler._handle_memory_deallocation(tensor_id(1), BLOCK_SIZE)
        self.handler._handle_memory_allocation(tensor_id(1), BLOCK_SIZE)
        errors = self.kernel_launch(stream_id(2), read_write=[tensor_id(1)])
        reuse_errors = [
            e for e in errors if isinstance(e, csan.AllocatorReuseRaceError)
        ]
        self.assertEqual(len(reuse_errors), 2)

    def test_record_stream_tracking(self):
        self.handler._handle_memory_allocation(tensor_id(1), BLOCK_SIZE)
        self.handler._handle_record_stream(tensor_id(1), stream_id(1))
        self.handler._handle_record_stream(tensor_id(1), stream_id(2))
        self.assertEqual(
            self.handler.pledged_streams[tensor_id(1)],
            {stream_id(1), stream_id(2)},
        )

    def test_allocator_reuse_write_clears_prior(self):
        self.handler._handle_memory_allocation(tensor_id(1), BLOCK_SIZE)
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_memory_deallocation(tensor_id(1), BLOCK_SIZE)
        self.handler._handle_memory_allocation(tensor_id(1), BLOCK_SIZE)
        errors = self.kernel_launch(stream_id(2), read_write=[tensor_id(1)])
        self.assertEqual(len(errors), 1)
        self.handler._handle_event_record(event_id(0), stream_id(2))
        self.handler._handle_event_wait(event_id(0), stream_id(3))
        errors = self.kernel_launch(stream_id(3), read_write=[tensor_id(1)])
        reuse_errors = [
            e for e in errors if isinstance(e, csan.AllocatorReuseRaceError)
        ]
        self.assertEqual(len(reuse_errors), 0)

    def test_allocator_reuse_split_block(self):
        base = tensor_id(1)
        self.handler._handle_memory_allocation(base, BLOCK_SIZE)
        self.handler._handle_stream_creation(stream_id(1))
        self.handler._handle_stream_creation(stream_id(2))
        self.assert_good_kernel_launch(stream_id(1), read_write=[base])
        self.handler._handle_memory_deallocation(base, BLOCK_SIZE)
        split_ptr = base + BLOCK_SIZE // 2
        self.handler._handle_memory_allocation(split_ptr, BLOCK_SIZE // 2)
        errors = self.kernel_launch(stream_id(2), read_write=[split_ptr])
        reuse_errors = [
            e for e in errors if isinstance(e, csan.AllocatorReuseRaceError)
        ]
        self.assertEqual(len(reuse_errors), 1)

    def test_allocator_reuse_coalesce_blocks(self):
        base = tensor_id(1)
        upper = base + BLOCK_SIZE
        self.handler._handle_memory_allocation(base, BLOCK_SIZE)
        self.handler._handle_memory_allocation(upper, BLOCK_SIZE)
        self.handler._handle_stream_creation(stream_id(1))
        self.handler._handle_stream_creation(stream_id(2))
        self.handler._handle_stream_creation(stream_id(3))
        self.assert_good_kernel_launch(stream_id(1), read_write=[base])
        self.assert_good_kernel_launch(stream_id(2), read_write=[upper])
        self.handler._handle_memory_deallocation(base, BLOCK_SIZE)
        self.handler._handle_memory_deallocation(upper, BLOCK_SIZE)
        self.handler._handle_memory_allocation(base, BLOCK_SIZE * 2)
        errors = self.kernel_launch(stream_id(3), read_write=[base])
        reuse_errors = [
            e for e in errors if isinstance(e, csan.AllocatorReuseRaceError)
        ]
        self.assertGreaterEqual(len(reuse_errors), 1)

    def test_allocator_reuse_no_overlap(self):
        base = tensor_id(1)
        far = base + BLOCK_SIZE * 2
        self.handler._handle_memory_allocation(base, BLOCK_SIZE)
        self.handler._handle_stream_creation(stream_id(1))
        self.handler._handle_stream_creation(stream_id(2))
        self.assert_good_kernel_launch(stream_id(1), read_write=[base])
        self.handler._handle_memory_deallocation(base, BLOCK_SIZE)
        self.handler._handle_memory_allocation(far, BLOCK_SIZE)
        errors = self.kernel_launch(stream_id(2), read_write=[far])
        self.assertEqual(len(errors), 0)


class TestMessages(TestCase):
    def setUp(self):
        super().setUp()
        self.handler = csan.EventHandler()

    def test_ensure_exists(self):
        ARG = 0
        with self.subTest(func=self.handler._handle_event_deletion):
            with self.assertLogs() as captured:
                self.handler._handle_event_deletion(ARG)
            self.assertIn("Found Event with id: 0", captured.records[0].getMessage())
        with self.subTest(func=self.handler._handle_memory_deallocation):
            with self.assertLogs() as captured:
                self.handler._handle_memory_deallocation(ARG, BLOCK_SIZE)
            self.assertIn(
                "Found tensor with pointer: 0", captured.records[0].getMessage()
            )

    def test_ensure_does_not_exist(self):
        ARG = 0
        self.handler._handle_event_creation(ARG)
        self.handler._handle_stream_creation(ARG)
        for func, out in [
            (
                self.handler._handle_event_creation,
                "Found duplicate event creation in the trace for event with "
                f"id: {ARG}. Assuming the trace for event deletion wasn't caught "
                "and backfilling it now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
            ),
            (
                self.handler._handle_stream_creation,
                "Found duplicate Stream creation in the trace for Stream with "
                f"id: {ARG}. PyTorch Streams are only created once, so this "
                "trace entry is ignored.",
            ),
        ]:
            with self.subTest(func=func, out=out):
                with self.assertLogs() as captured:
                    func(ARG)
                self.assertEqual(captured.records[0].getMessage(), out)

    def test_error_message(self):
        current_access = csan.Access(
            type=csan.AccessType.WRITE,
            seq_num=1,
            stream=stream_id(1),
            operator="schema",
            aliases=["b"],
            is_output=True,
            stack_trace=traceback.StackSummary.from_list(
                [("file", 0, "name", "trace a")]
            ),
        )
        previous_access = csan.Access(
            type=csan.AccessType.READ,
            seq_num=2,
            stream=stream_id(0),
            operator="schema",
            aliases=["a"],
            is_output=False,
            stack_trace=traceback.StackSummary.from_list(
                [("file", 0, "name", "trace b")]
            ),
        )
        error = csan.UnsynchronizedAccessError(
            data_ptr=tensor_id(1),
            allocation_stack_trace=traceback.StackSummary.from_list(
                [("file", 0, "name", "alloc")]
            ),
            current_access=current_access,
            previous_access=previous_access,
        )
        error_str = str(error)
        self.assertIn("CSAN detected a possible data race", error_str)
        self.assertIn("data pointer 1", error_str)
        self.assertIn("stream 1001", error_str)
        self.assertIn("stream 1000", error_str)
        self.assertIn("trace a", error_str)
        self.assertIn("trace b", error_str)
        self.assertIn("alloc", error_str)
        self.assertIn("have never synchronized", error_str)
        self.assertIn("wait_stream", error_str)

    def test_error_shows_sync_frontier(self):
        current_access = csan.Access(
            type=csan.AccessType.WRITE,
            seq_num=10,
            stream=stream_id(1),
            operator="op",
            aliases=[],
            is_output=False,
            stack_trace=traceback.StackSummary.from_list([("f", 0, "n", "")]),
        )
        previous_access = csan.Access(
            type=csan.AccessType.WRITE,
            seq_num=5,
            stream=stream_id(0),
            operator="op",
            aliases=[],
            is_output=False,
            stack_trace=traceback.StackSummary.from_list([("f", 0, "n", "")]),
        )
        error = csan.UnsynchronizedAccessError(
            tensor_id(1),
            None,
            current_access,
            previous_access,
            known_seq=3,
        )
        error_str = str(error)
        self.assertIn("up to seq 3", error_str)
        self.assertIn("seq 5", error_str)
        self.assertNotIn("never synchronized", error_str)

    def test_error_never_synced(self):
        current_access = csan.Access(
            type=csan.AccessType.WRITE,
            seq_num=10,
            stream=stream_id(1),
            operator="op",
            aliases=[],
            is_output=False,
            stack_trace=traceback.StackSummary.from_list([("f", 0, "n", "")]),
        )
        previous_access = csan.Access(
            type=csan.AccessType.WRITE,
            seq_num=5,
            stream=stream_id(0),
            operator="op",
            aliases=[],
            is_output=False,
            stack_trace=traceback.StackSummary.from_list([("f", 0, "n", "")]),
        )
        error = csan.UnsynchronizedAccessError(
            tensor_id(1),
            None,
            current_access,
            previous_access,
            known_seq=-1,
        )
        self.assertIn("have never synchronized", str(error))

    def test_allocator_reuse_error_message(self):
        current_access = csan.Access(
            type=csan.AccessType.WRITE,
            seq_num=10,
            stream=stream_id(1),
            operator="op",
            aliases=[],
            is_output=False,
            stack_trace=traceback.StackSummary.from_list([("f", 0, "n", "")]),
        )
        previous_access = csan.Access(
            type=csan.AccessType.WRITE,
            seq_num=5,
            stream=stream_id(0),
            operator="op",
            aliases=[],
            is_output=False,
            stack_trace=traceback.StackSummary.from_list([("f", 0, "n", "")]),
        )
        error = csan.AllocatorReuseRaceError(
            tensor_id(1),
            None,
            current_access,
            previous_access,
            None,
        )
        error_str = str(error)
        self.assertIn("allocator memory reuse", error_str)
        self.assertIn("record_stream", error_str)

    def test_subclass(self):
        class MyT(torch.Tensor):
            def __new__(cls, data):
                new_data = data.clone()
                return new_data.as_subclass(cls)

        try:
            csan.enable_cuda_sanitizer()

            # These two tests ensure that subclass creation
            # happens smoothly under the mode used by csan
            TwoTensor(torch.rand(2), torch.rand(2))
            MyT(torch.rand(2))
        finally:
            csan.cuda_sanitizer.disable()


class TestDeduplication(TestCase):
    """Tests for race signature deduplication (S2 iteration periodicity)."""

    def setUp(self):
        super().setUp()
        self.handler = csan.EventHandler()

    def kernel_launch(
        self,
        stream: StreamId,
        read_only: list[DataPtr] | None = None,
        read_write: list[DataPtr] | None = None,
        operator: str = "",
    ) -> list[csan.SynchronizationError]:
        if read_only is None:
            read_only = []
        if read_write is None:
            read_write = []
        return self.handler._handle_kernel_launch(
            stream,
            read_only,
            read_write,
            {},
            operator,
            {k: [""] for k in read_only + read_write},
        )

    def test_race_dedup_same_signature(self):
        mode = csan.CUDASanitizerDispatchMode()
        mode.accumulate = True
        mode.event_handler = self.handler
        for _ in range(5):
            self.kernel_launch(stream_id(1), read_write=[tensor_id(1)], operator="op_a")
            errors = self.kernel_launch(
                stream_id(2), read_write=[tensor_id(1)], operator="op_b"
            )
            mode._report_errors(errors)
            self.handler._handle_device_synchronization()
        self.assertEqual(len(mode.accumulated_errors), 1)
        sig = mode.accumulated_errors[0].race_signature
        self.assertEqual(mode._seen_race_sigs[sig], 5)

    def test_race_dedup_different_signatures(self):
        mode = csan.CUDASanitizerDispatchMode()
        mode.accumulate = True
        mode.event_handler = self.handler
        self.kernel_launch(stream_id(1), read_write=[tensor_id(1)], operator="op_a")
        errors1 = self.kernel_launch(
            stream_id(2), read_write=[tensor_id(1)], operator="op_b"
        )
        mode._report_errors(errors1)
        self.handler._handle_device_synchronization()
        self.kernel_launch(stream_id(1), read_write=[tensor_id(2)], operator="op_c")
        errors2 = self.kernel_launch(
            stream_id(3), read_write=[tensor_id(2)], operator="op_d"
        )
        mode._report_errors(errors2)
        self.assertEqual(len(mode.accumulated_errors), 2)

    def test_known_seq_in_errors(self):
        self.kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_event_record(event_id(0), stream_id(1))
        self.kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_event_wait(event_id(0), stream_id(2))
        errors = self.kernel_launch(stream_id(2), read_write=[tensor_id(1)])
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], csan.UnsynchronizedAccessError)
        self.assertGreater(errors[0].known_seq, -1)


class TestNCCLStreamResolution(TestCase):
    """Tests for correct NCCL stream attribution in c10d collectives."""

    def test_extract_async_op_from_positional(self):
        schema = torch.ops.c10d.allreduce_.default._schema
        async_op_idx = None
        for i, arg in enumerate(schema.arguments):
            if arg.name == "async_op":
                async_op_idx = i
                break
        self.assertIsNotNone(async_op_idx, "allreduce_ schema must have async_op")
        args = [None] * (async_op_idx + 1)
        args[async_op_idx] = False
        result = csan.CUDASanitizerDispatchMode._extract_async_op(
            schema, tuple(args), {}
        )
        self.assertFalse(result)

    def test_extract_async_op_from_kwargs(self):
        schema = torch.ops.c10d.allreduce_.default._schema
        result = csan.CUDASanitizerDispatchMode._extract_async_op(
            schema, (), {"async_op": False}
        )
        self.assertFalse(result)

    def test_extract_async_op_default_is_true(self):
        schema = torch.ops.c10d.allreduce_.default._schema
        result = csan.CUDASanitizerDispatchMode._extract_async_op(schema, (), {})
        self.assertTrue(result)

    def test_non_c10d_op_is_not_async_collective(self):
        mode = csan.CUDASanitizerDispatchMode()
        schema = torch.ops.aten.add.Tensor._schema
        self.assertFalse(mode._is_async_nccl_collective(schema, (), {}))

    def test_c10d_barrier_is_not_async_collective(self):
        mode = csan.CUDASanitizerDispatchMode()
        schema = torch.ops.c10d.barrier.default._schema
        self.assertFalse(mode._is_async_nccl_collective(schema, (), {}))

    def test_c10d_sync_op_is_not_async_collective(self):
        mode = csan.CUDASanitizerDispatchMode()
        schema = torch.ops.c10d.allreduce_.default._schema
        async_op_idx = None
        for i, arg in enumerate(schema.arguments):
            if arg.name == "async_op":
                async_op_idx = i
                break
        args = [None] * (async_op_idx + 1)
        args[async_op_idx] = False
        self.assertFalse(mode._is_async_nccl_collective(schema, tuple(args), {}))

    def test_c10d_async_op_is_async_collective(self):
        mode = csan.CUDASanitizerDispatchMode()
        schema = torch.ops.c10d.allreduce_.default._schema
        self.assertTrue(mode._is_async_nccl_collective(schema, (), {}))


class TestCUDASanitizerEndToEnd(TestCase):
    """End-to-end tests that run on real CUDA hardware.

    The other test classes exercise the vector-clock logic with synthetic IDs.
    These tests verify the full pipeline: real kernels on real streams, traced
    by the GPU trace callbacks, analyzed by EventHandler, reported through the
    context-manager accumulation API.
    """

    def test_catches_cross_stream_race(self):
        with csan.cuda_sanitizer as san:
            s1 = torch.cuda.Stream()
            t = torch.zeros(1024, device="cuda")
            with torch.cuda.stream(s1):
                t.fill_(1.0)
            # No sync -- reading on default stream is a race
            _ = t.sum()
        self.assertGreater(len(san.errors), 0)

    def test_no_false_positive_with_sync(self):
        with csan.cuda_sanitizer as san:
            s1 = torch.cuda.Stream()
            t = torch.zeros(1024, device="cuda")
            s1.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s1):
                t.fill_(1.0)
            torch.cuda.current_stream().wait_stream(s1)
            _ = t.sum()
        self.assertEqual(len(san.errors), 0)

    def test_context_manager_reuse(self):
        for i in range(3):
            with csan.cuda_sanitizer as san:
                t = torch.zeros(64, device="cuda")
                _ = t.sum()
            self.assertEqual(len(san.errors), 0, f"Unexpected error on iteration {i}")

    def test_accumulates_multiple_errors(self):
        with csan.cuda_sanitizer as san:
            for _ in range(3):
                s1 = torch.cuda.Stream()
                t = torch.zeros(256, device="cuda")
                with torch.cuda.stream(s1):
                    t.fill_(1.0)
                _ = t.sum()
                torch.cuda.synchronize()
        self.assertGreaterEqual(len(san.errors), 3)

    def test_catches_overlapping_view_race(self):
        with csan.cuda_sanitizer as san:
            s1 = torch.cuda.Stream()
            t = torch.zeros(100, device="cuda")
            a = t[:60]
            b = t[40:]
            s1.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s1):
                a.fill_(1.0)
            # b overlaps a, read on default stream without sync -- race
            _ = b.sum()
        overlap_errors = [
            e for e in san.errors if isinstance(e, csan.OverlappingViewAccessError)
        ]
        self.assertGreater(len(overlap_errors), 0)

    def test_no_false_positive_non_overlapping_views(self):
        with csan.cuda_sanitizer as san:
            s1 = torch.cuda.Stream()
            t = torch.zeros(100, device="cuda")
            a = t[:40]
            b = t[60:]
            s1.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s1):
                a.fill_(1.0)
            torch.cuda.current_stream().wait_stream(s1)
            _ = b.sum()
        overlap_errors = [
            e for e in san.errors if isinstance(e, csan.OverlappingViewAccessError)
        ]
        self.assertEqual(len(overlap_errors), 0)


class TestByteRangeComputation(TestCase):
    def test_contiguous_1d(self):
        t = torch.randn(10, device="cuda")
        start, end = csan._tensor_byte_range(t)
        self.assertEqual(start, t.data_ptr())
        self.assertEqual(end, t.data_ptr() + 10 * t.element_size())

    def test_contiguous_2d(self):
        t = torch.randn(3, 4, device="cuda")
        start, end = csan._tensor_byte_range(t)
        self.assertEqual(start, t.data_ptr())
        self.assertEqual(end, t.data_ptr() + 12 * t.element_size())

    def test_slice_view(self):
        t = torch.randn(10, device="cuda")
        v = t[2:8]
        start, end = csan._tensor_byte_range(v)
        self.assertEqual(start, v.data_ptr())
        self.assertEqual(end, v.data_ptr() + 6 * v.element_size())

    def test_transpose_view(self):
        t = torch.randn(3, 4, device="cuda")
        v = t.t()
        start, end = csan._tensor_byte_range(v)
        self.assertEqual(start, t.data_ptr())
        self.assertEqual(end, t.data_ptr() + 12 * t.element_size())

    def test_zero_stride(self):
        t = torch.randn(1, 10, device="cuda")
        v = t.expand(5, 10)
        start, end = csan._tensor_byte_range(v)
        self.assertEqual(start, t.data_ptr())
        self.assertEqual(end, t.data_ptr() + 10 * t.element_size())

    def test_scalar(self):
        t = torch.tensor(1.0, device="cuda")
        start, end = csan._tensor_byte_range(t)
        self.assertEqual(start, t.data_ptr())
        self.assertEqual(end, t.data_ptr() + t.element_size())

    def test_empty(self):
        t = torch.empty(0, device="cuda")
        start, end = csan._tensor_byte_range(t)
        self.assertEqual(start, end)


class TestViewOverlapDetection(TestCase):
    """Tests for view-overlap race detection in EventHandler."""

    def setUp(self):
        super().setUp()
        self.handler = csan.EventHandler()

    def kernel_launch_with_ranges(
        self,
        stream: StreamId,
        read_only: list[DataPtr] | None = None,
        read_write: list[DataPtr] | None = None,
        byte_ranges: dict[DataPtr, tuple[int, int]] | None = None,
    ) -> list[csan.SynchronizationError]:
        if read_only is None:
            read_only = []
        if read_write is None:
            read_write = []
        return self.handler._handle_kernel_launch(
            stream,
            read_only,
            read_write,
            {},
            "",
            {k: [""] for k in read_only + read_write},
            byte_ranges=byte_ranges,
        )

    def test_overlapping_write_read_race(self):
        errors = self.kernel_launch_with_ranges(
            stream_id(1),
            read_write=[tensor_id(1)],
            byte_ranges={tensor_id(1): (100, 200)},
        )
        self.assertEqual(errors, [])
        errors = self.kernel_launch_with_ranges(
            stream_id(2),
            read_only=[tensor_id(2)],
            byte_ranges={tensor_id(2): (150, 250)},
        )
        overlap_errors = [
            e for e in errors if isinstance(e, csan.OverlappingViewAccessError)
        ]
        self.assertEqual(len(overlap_errors), 1)

    def test_overlapping_write_write_race(self):
        self.kernel_launch_with_ranges(
            stream_id(1),
            read_write=[tensor_id(1)],
            byte_ranges={tensor_id(1): (100, 200)},
        )
        errors = self.kernel_launch_with_ranges(
            stream_id(2),
            read_write=[tensor_id(2)],
            byte_ranges={tensor_id(2): (150, 250)},
        )
        overlap_errors = [
            e for e in errors if isinstance(e, csan.OverlappingViewAccessError)
        ]
        self.assertEqual(len(overlap_errors), 1)

    def test_overlapping_read_read_safe(self):
        self.kernel_launch_with_ranges(
            stream_id(1),
            read_only=[tensor_id(1)],
            byte_ranges={tensor_id(1): (100, 200)},
        )
        errors = self.kernel_launch_with_ranges(
            stream_id(2),
            read_only=[tensor_id(2)],
            byte_ranges={tensor_id(2): (150, 250)},
        )
        overlap_errors = [
            e for e in errors if isinstance(e, csan.OverlappingViewAccessError)
        ]
        self.assertEqual(len(overlap_errors), 0)

    def test_overlapping_with_sync(self):
        self.kernel_launch_with_ranges(
            stream_id(1),
            read_write=[tensor_id(1)],
            byte_ranges={tensor_id(1): (100, 200)},
        )
        self.handler._handle_event_record(event_id(0), stream_id(1))
        self.handler._handle_event_wait(event_id(0), stream_id(2))
        errors = self.kernel_launch_with_ranges(
            stream_id(2),
            read_write=[tensor_id(2)],
            byte_ranges={tensor_id(2): (150, 250)},
        )
        overlap_errors = [
            e for e in errors if isinstance(e, csan.OverlappingViewAccessError)
        ]
        self.assertEqual(len(overlap_errors), 0)

    def test_non_overlapping_adjacent(self):
        self.kernel_launch_with_ranges(
            stream_id(1),
            read_write=[tensor_id(1)],
            byte_ranges={tensor_id(1): (100, 200)},
        )
        errors = self.kernel_launch_with_ranges(
            stream_id(2),
            read_write=[tensor_id(2)],
            byte_ranges={tensor_id(2): (200, 300)},
        )
        overlap_errors = [
            e for e in errors if isinstance(e, csan.OverlappingViewAccessError)
        ]
        self.assertEqual(len(overlap_errors), 0)

    def test_no_byte_ranges_fallback(self):
        self.kernel_launch_with_ranges(
            stream_id(1),
            read_write=[tensor_id(1)],
        )
        errors = self.kernel_launch_with_ranges(
            stream_id(2),
            read_write=[tensor_id(2)],
        )
        overlap_errors = [
            e for e in errors if isinstance(e, csan.OverlappingViewAccessError)
        ]
        self.assertEqual(len(overlap_errors), 0)

    def test_empty_range_no_overlap(self):
        self.kernel_launch_with_ranges(
            stream_id(1),
            read_write=[tensor_id(1)],
            byte_ranges={tensor_id(1): (100, 100)},
        )
        errors = self.kernel_launch_with_ranges(
            stream_id(2),
            read_write=[tensor_id(2)],
            byte_ranges={tensor_id(2): (100, 200)},
        )
        overlap_errors = [
            e for e in errors if isinstance(e, csan.OverlappingViewAccessError)
        ]
        self.assertEqual(len(overlap_errors), 0)


class TestOverlappingViewMessage(TestCase):
    def test_overlap_error_message(self):
        current_access = csan.Access(
            type=csan.AccessType.WRITE,
            seq_num=2,
            stream=stream_id(1),
            operator="aten.fill_",
            aliases=["self"],
            is_output=False,
            stack_trace=traceback.StackSummary.from_list(
                [("file.py", 10, "fn", "fill_(1.0)")]
            ),
            byte_range=(160, 400),
        )
        previous_access = csan.Access(
            type=csan.AccessType.WRITE,
            seq_num=1,
            stream=stream_id(0),
            operator="aten.mul_",
            aliases=["self"],
            is_output=False,
            stack_trace=traceback.StackSummary.from_list(
                [("file.py", 5, "fn", "mul_(2.0)")]
            ),
            byte_range=(0, 240),
        )
        error = csan.OverlappingViewAccessError(
            current_data_ptr=tensor_id(2),
            current_byte_range=(160, 400),
            previous_data_ptr=tensor_id(1),
            previous_byte_range=(0, 240),
            current_access=current_access,
            previous_access=previous_access,
        )
        error_str = str(error)
        self.assertIn("overlapping views", error_str)
        self.assertIn(str(tensor_id(1)), error_str)
        self.assertIn(str(tensor_id(2)), error_str)
        self.assertIn("[160, 240)", error_str)
        self.assertIn("stream 1001", error_str)
        self.assertIn("stream 1000", error_str)
        self.assertIn("wait_stream", error_str)
        self.assertIn("have never synchronized", error_str)


class TestCUDAGraphStreamSafety(TestCase):
    """Tests that the sanitizer detects cross-stream races involving CUDA graph replay.

    CUDA graph replay executes as a single cudaGraphLaunch call invisible to
    __torch_dispatch__. The sanitizer hooks CUDAGraph.replay() to emit synthetic
    read/write events from the capture-time tensor profile, enabling vector-clock
    race detection between graph replays and surrounding ops.
    """

    def test_graph_replay_cross_stream_race(self):
        g = torch.cuda.CUDAGraph()
        s1 = torch.cuda.Stream()
        static_input = torch.zeros(1024, device="cuda")
        static_output = torch.zeros(1024, device="cuda")

        with csan.cuda_sanitizer as san:
            with torch.cuda.stream(s1):
                with torch.cuda.graph(g, stream=s1):
                    static_output.copy_(static_input + 1)

            # Replay on s1
            with torch.cuda.stream(s1):
                g.replay()

            # Read static_output on default stream without sync -- race!
            _ = static_output.sum()

        self.assertGreater(len(san.errors), 0)

    def test_graph_replay_no_false_positive_with_sync(self):
        g = torch.cuda.CUDAGraph()
        s1 = torch.cuda.Stream()
        static_input = torch.zeros(1024, device="cuda")
        static_output = torch.zeros(1024, device="cuda")

        with csan.cuda_sanitizer as san:
            with torch.cuda.stream(s1):
                with torch.cuda.graph(g, stream=s1):
                    static_output.copy_(static_input + 1)

            with torch.cuda.stream(s1):
                g.replay()

            # Sync before reading on default stream
            torch.cuda.current_stream().wait_stream(s1)
            _ = static_output.sum()

        self.assertEqual(len(san.errors), 0)

    def test_graph_input_modification_race(self):
        g = torch.cuda.CUDAGraph()
        s1 = torch.cuda.Stream()
        static_input = torch.zeros(1024, device="cuda")
        static_output = torch.zeros(1024, device="cuda")

        with csan.cuda_sanitizer as san:
            with torch.cuda.stream(s1):
                with torch.cuda.graph(g, stream=s1):
                    static_output.copy_(static_input + 1)

            # Modify static_input on default stream
            static_input.fill_(42.0)

            # Replay on s1 reads static_input -- race with the fill_ above!
            with torch.cuda.stream(s1):
                g.replay()

            torch.cuda.synchronize()

        self.assertGreater(len(san.errors), 0)

    def test_graph_replay_same_stream_safe(self):
        g = torch.cuda.CUDAGraph()
        s1 = torch.cuda.Stream()
        static_input = torch.zeros(1024, device="cuda")
        static_output = torch.zeros(1024, device="cuda")

        with csan.cuda_sanitizer as san:
            with torch.cuda.stream(s1):
                with torch.cuda.graph(g, stream=s1):
                    static_output.copy_(static_input + 1)

            # Replay and consume on the same stream -- no race
            with torch.cuda.stream(s1):
                g.replay()
                _ = static_output.sum()

            torch.cuda.synchronize()

        self.assertEqual(len(san.errors), 0)

    def test_graph_recapture_updates_profile(self):
        g = torch.cuda.CUDAGraph()
        s1 = torch.cuda.Stream()
        t1 = torch.zeros(512, device="cuda")
        t2 = torch.zeros(512, device="cuda")

        with csan.cuda_sanitizer as san:
            # First capture: touches t1
            with torch.cuda.stream(s1):
                with torch.cuda.graph(g, stream=s1):
                    t1.fill_(1.0)

            profile1 = san.dispatch._graph_profiles.get(id(g))
            self.assertIsNotNone(profile1)
            self.assertIn(t1.data_ptr(), profile1.writes)

            # Reset and re-capture: now touches t2 instead
            g.reset()
            self.assertNotIn(id(g), san.dispatch._graph_profiles)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.stream(s1):
                with torch.cuda.graph(g, stream=s1):
                    t2.fill_(2.0)

            profile2 = san.dispatch._graph_profiles.get(id(g))
            self.assertIsNotNone(profile2)
            self.assertIn(t2.data_ptr(), profile2.writes)

        self.assertEqual(len(san.errors), 0)


if __name__ == "__main__":
    run_tests()
