# mypy: allow-untyped-defs
r"""
This module introduces CUDA Sanitizer, a tool for detecting synchronization errors between kernels ran on different streams.

It stores information on accesses to tensors to determine if they are synchronized
or not. When enabled in a python program and a possible data race is detected, a
detailed warning will be printed and the program will exit.

It can be enabled either by importing this module and calling
:func:`enable_cuda_sanitizer()` or by exporting the ``TORCH_CUDA_SANITIZER``
environment variable.
"""

import enum
import functools
import inspect
import io
import logging
import re
import sys
import textwrap
import traceback
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, TypeVar

import torch
import torch.cuda._gpu_trace as gpu_trace
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode


aten = torch.ops.aten

DEFAULT_STREAM_ID = 0

TK = TypeVar("TK")
TVa = TypeVar("TVa")
TVb = TypeVar("TVb")

DataPtr = int
StreamId = int
EventId = int
SeqNum = int

logger = logging.getLogger(__name__)

# Note that this is only factories that take Tensor as input as they are
# the ones we care about.
FACTORY_FUNCTION_REGEX = re.compile("(new_.*|.*_like)")


def _tensor_byte_range(tensor: torch.Tensor) -> tuple[int, int]:
    """Bounding-box byte range of a tensor's accessible memory."""
    ptr = tensor.data_ptr()
    if ptr == 0 or tensor.numel() == 0:
        return (ptr, ptr)
    min_offset = 0
    max_offset = 0
    for size, stride in zip(tensor.shape, tensor.stride()):
        if size > 1:
            step = (size - 1) * stride
            if step > 0:
                max_offset += step
            else:
                min_offset += step
    elem_size = tensor.element_size()
    return (ptr + min_offset * elem_size, ptr + max_offset * elem_size + elem_size)


class AccessType(enum.Enum):
    READ = enum.auto()
    WRITE = enum.auto()

    def __str__(self):
        return "reading from" if self is AccessType.READ else "writing to"


@dataclass
class Access:
    r"""Stores information about a single access to a tensor by a kernel.

    Args:
        type: either AccessType.READ or AccessType.Write.
        seq_num: the sequential number of the kernel performing the access.
        stream: the stream id of the stream executing the kernel.
        operator: the schema of the launched kernel, which lists the
            arguments and return type.
        aliases: the arguments in the schema this access corresponds to.
        is_output: Whether the tensor was an output of the kernel.
        stack_trace: the stack summary object captured during access.
    """

    type: AccessType
    seq_num: SeqNum
    stream: StreamId
    operator: str
    aliases: list[str]
    is_output: bool
    stack_trace: traceback.StackSummary
    byte_range: tuple[int, int] | None = None


class SynchronizationError(Exception):
    """Base class for errors detected by CUDA Sanitizer."""


class UnsynchronizedAccessError(SynchronizationError):
    """Stores information about two unsynchronized accesses to one data pointer."""

    def __init__(
        self,
        data_ptr: DataPtr,
        allocation_stack_trace: traceback.StackSummary | None,
        current_access: Access,
        previous_access: Access,
        known_seq: SeqNum = -1,
    ):
        self.data_ptr = data_ptr
        self.allocation_stack_trace = allocation_stack_trace
        self.current_access = current_access
        self.previous_access = previous_access
        self.known_seq = known_seq

    @property
    def race_signature(self) -> tuple:
        return (
            self.current_access.stream,
            str(self.current_access.operator),
            self.current_access.type,
            self.previous_access.stream,
            str(self.previous_access.operator),
            self.previous_access.type,
        )

    def __str__(self):
        def format_access(access: Access):
            message.write(f"{access.operator}\n{access.type}")
            if access.aliases:
                message.write(" argument(s) " + ", ".join(access.aliases))
                if access.is_output:
                    message.write(", and to")
            if access.is_output:
                message.write(" the output")
            message.write(
                f"\nWith stack trace:\n{''.join(access.stack_trace.format())}\n"
            )

        with io.StringIO() as message:
            message.write(
                textwrap.dedent(
                    f"""\
                    ============================
                    CSAN detected a possible data race on tensor with data pointer {self.data_ptr}
                    Access by stream {self.current_access.stream} during kernel:
                    """
                )
            )
            format_access(self.current_access)

            message.write(
                f"Previous access by stream {self.previous_access.stream} during kernel:\n"
            )
            format_access(self.previous_access)

            if self.allocation_stack_trace:
                message.write(
                    "Tensor was allocated with stack trace:\n"
                    f"{''.join(self.allocation_stack_trace.format())}\n"
                )
            else:
                message.write("Trace for tensor allocation not found.\n")

            if self.known_seq == -1:
                message.write(
                    f"\nStreams {self.current_access.stream} and "
                    f"{self.previous_access.stream} have never synchronized.\n"
                )
            else:
                message.write(
                    f"\nLast sync: stream {self.current_access.stream} synced with "
                    f"stream {self.previous_access.stream} up to seq {self.known_seq}, "
                    f"but the conflicting access was at seq "
                    f"{self.previous_access.seq_num}.\n"
                )

            message.write(
                "\nTo fix: synchronize the streams before the second access:\n"
                "  torch.cuda.current_stream().wait_stream(other_stream)\n"
            )
            return message.getvalue()


class AllocatorReuseRaceError(SynchronizationError):
    """Race between a freed tensor's in-flight ops and a new allocation at the same address."""

    def __init__(
        self,
        data_ptr: DataPtr,
        allocation_stack_trace: traceback.StackSummary | None,
        current_access: Access,
        previous_access: Access,
        previous_alloc_stack_trace: traceback.StackSummary | None,
        known_seq: SeqNum = -1,
    ):
        self.data_ptr = data_ptr
        self.allocation_stack_trace = allocation_stack_trace
        self.current_access = current_access
        self.previous_access = previous_access
        self.previous_alloc_stack_trace = previous_alloc_stack_trace
        self.known_seq = known_seq

    @property
    def race_signature(self) -> tuple:
        return (
            self.current_access.stream,
            str(self.current_access.operator),
            self.current_access.type,
            self.previous_access.stream,
            str(self.previous_access.operator),
            self.previous_access.type,
        )

    def __str__(self):
        def format_access(access: Access):
            message.write(f"{access.operator}\n{access.type}")
            if access.aliases:
                message.write(" argument(s) " + ", ".join(access.aliases))
                if access.is_output:
                    message.write(", and to")
            if access.is_output:
                message.write(" the output")
            message.write(
                f"\nWith stack trace:\n{''.join(access.stack_trace.format())}\n"
            )

        with io.StringIO() as message:
            message.write(
                textwrap.dedent(
                    f"""\
                    ============================
                    CSAN detected a possible data race from allocator memory reuse
                    on tensor with data pointer {self.data_ptr}
                    New access by stream {self.current_access.stream} during kernel:
                    """
                )
            )
            format_access(self.current_access)

            message.write(
                f"Previous-lifetime access by stream {self.previous_access.stream} during kernel:\n"
            )
            format_access(self.previous_access)

            if self.previous_alloc_stack_trace:
                message.write(
                    "Previous tensor was allocated with stack trace:\n"
                    f"{''.join(self.previous_alloc_stack_trace.format())}\n"
                )

            if self.allocation_stack_trace:
                message.write(
                    "New tensor was allocated with stack trace:\n"
                    f"{''.join(self.allocation_stack_trace.format())}"
                )

            message.write(
                "\nTo fix: call tensor.record_stream(stream) before freeing "
                "the tensor,\nor synchronize the streams before the new allocation:\n"
                "  torch.cuda.current_stream().wait_stream(other_stream)\n"
            )
            return message.getvalue()


class OverlappingViewAccessError(SynchronizationError):
    """Race between overlapping views of the same storage on different streams."""

    def __init__(
        self,
        current_data_ptr: DataPtr,
        current_byte_range: tuple[int, int],
        previous_data_ptr: DataPtr,
        previous_byte_range: tuple[int, int],
        current_access: Access,
        previous_access: Access,
        known_seq: SeqNum = -1,
    ):
        self.current_data_ptr = current_data_ptr
        self.current_byte_range = current_byte_range
        self.previous_data_ptr = previous_data_ptr
        self.previous_byte_range = previous_byte_range
        self.current_access = current_access
        self.previous_access = previous_access
        self.known_seq = known_seq

    @property
    def race_signature(self) -> tuple:
        return (
            self.current_access.stream,
            str(self.current_access.operator),
            self.current_access.type,
            self.previous_access.stream,
            str(self.previous_access.operator),
            self.previous_access.type,
        )

    def __str__(self):
        def format_access(access: Access):
            message.write(f"{access.operator}\n{access.type}")
            if access.aliases:
                message.write(" argument(s) " + ", ".join(access.aliases))
                if access.is_output:
                    message.write(", and to")
            if access.is_output:
                message.write(" the output")
            message.write(
                f"\nWith stack trace:\n{''.join(access.stack_trace.format())}\n"
            )

        overlap_start = max(self.current_byte_range[0], self.previous_byte_range[0])
        overlap_end = min(self.current_byte_range[1], self.previous_byte_range[1])

        with io.StringIO() as message:
            message.write(
                textwrap.dedent(
                    f"""\
                    ============================
                    CSAN detected a possible data race between overlapping views
                    Current tensor at {self.current_data_ptr} (bytes [{self.current_byte_range[0]}, {self.current_byte_range[1]}))
                    Previous tensor at {self.previous_data_ptr} (bytes [{self.previous_byte_range[0]}, {self.previous_byte_range[1]}))
                    Overlapping region: bytes [{overlap_start}, {overlap_end})
                    Access by stream {self.current_access.stream} during kernel:
                    """
                )
            )
            format_access(self.current_access)

            message.write(
                f"Previous access by stream {self.previous_access.stream} during kernel:\n"
            )
            format_access(self.previous_access)

            if self.known_seq == -1:
                message.write(
                    f"\nStreams {self.current_access.stream} and "
                    f"{self.previous_access.stream} have never synchronized.\n"
                )
            else:
                message.write(
                    f"\nLast sync: stream {self.current_access.stream} synced with "
                    f"stream {self.previous_access.stream} up to seq {self.known_seq}, "
                    f"but the conflicting access was at seq "
                    f"{self.previous_access.seq_num}.\n"
                )

            message.write(
                "\nTo fix: synchronize the streams before the second access:\n"
                "  torch.cuda.current_stream().wait_stream(other_stream)\n"
            )
            return message.getvalue()


class CUDASanitizerErrors(Exception):
    """Wrapper class for errors reported by CUDA Sanitizer."""

    def __init__(self, errors: list[SynchronizationError]):
        self.errors = errors

    def __str__(self):
        return f"detected {len(self.errors)} errors"


@dataclass
class _PendingFree:
    """Access history stashed when a tensor is freed, pending allocator reuse."""

    write: Access | None
    reads: list[Access]
    allocation_stack_trace: traceback.StackSummary | None
    pledged_streams: set[StreamId]


@dataclass
class _PriorLifecycleAccesses:
    """Uncovered accesses from a previous allocation at the same data_ptr."""

    write: Access | None
    reads: list[Access]
    allocation_stack_trace: traceback.StackSummary | None


@dataclass
class TensorInfo:
    r"""Stores information about a single tensor and recent accesses to it.

    Args:
        allocation_stack_trace: the stack summary object captured during tensor
            allocation. Can be ``None`` if the allocation wasn't caught by CSAN.
        reads: list of read accesses to the tensor that were performed since
            the last write.
        write: the last write access to the tensor.
        prior_lifecycle: uncovered accesses from a previous allocation at this
            data_ptr, checked against new accesses to detect allocator reuse races.
    """

    allocation_stack_trace: traceback.StackSummary | None
    reads: list[Access] = field(default_factory=list)
    write: Access | None = None
    prior_lifecycle: _PriorLifecycleAccesses | None = None


class _TensorsAccessed:
    def __init__(self) -> None:
        self.accesses: dict[DataPtr, TensorInfo] = {}

    def ensure_tensor_exists(self, data_ptr: DataPtr) -> None:
        if data_ptr not in self.accesses:
            logger.info(
                "Found tensor with pointer: %s, but no matching tensor "
                "allocation in the trace. Backfilling the trace now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                data_ptr,
            )
            self.create_tensor(data_ptr, None)

    def ensure_tensor_does_not_exist(self, data_ptr: DataPtr) -> None:
        if data_ptr in self.accesses:
            logger.info(
                "Found duplicate tensor allocation in the trace for tensor with "
                "pointer: %s. Assuming the trace for tensor deallocation "
                "wasn't caught and backfilling it now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                data_ptr,
            )
            self.delete_tensor(data_ptr)

    def create_tensor(
        self, data_ptr: DataPtr, stack_trace: traceback.StackSummary | None
    ) -> None:
        self.accesses[data_ptr] = TensorInfo(stack_trace)

    def delete_tensor(self, data_ptr: DataPtr) -> None:
        del self.accesses[data_ptr]

    def were_there_reads_since_last_write(self, data_ptr: DataPtr) -> bool:
        return bool(self.accesses[data_ptr].reads)

    def get_allocation_stack_trace(
        self, data_ptr: DataPtr
    ) -> traceback.StackSummary | None:
        return self.accesses[data_ptr].allocation_stack_trace

    def get_write(self, data_ptr: DataPtr) -> Access | None:
        return self.accesses[data_ptr].write

    def get_reads(self, data_ptr: DataPtr) -> list[Access]:
        return self.accesses[data_ptr].reads

    def add_read(self, data_ptr: DataPtr, access: Access) -> None:
        self.accesses[data_ptr].reads.append(access)

    def set_write(self, data_ptr: DataPtr, access: Access) -> None:
        self.accesses[data_ptr].write = access
        self.accesses[data_ptr].reads = []


class StreamSynchronizations:
    def __init__(self) -> None:
        self.current_sync_states: dict[StreamId, dict[StreamId, SeqNum]] = {}
        self.recorded_sync_states: dict[EventId, dict[StreamId, SeqNum]] = {}
        self.host_sync_state: dict[StreamId, SeqNum] = {}
        self.create_stream(DEFAULT_STREAM_ID)

    def _ensure_stream_exists(self, stream: StreamId) -> None:
        if stream not in self.current_sync_states:
            logger.info(
                "Found Stream with id: %s, but no matching stream "
                "creation in the trace. Backfilling the trace now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                stream,
            )
            self.create_stream(stream)

    def _ensure_event_exists(self, event: EventId) -> None:
        if event not in self.recorded_sync_states:
            logger.info(
                "Found Event with id: %s, but no matching event "
                "creation in the trace. Backfilling the trace now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                event,
            )
            self.create_event(event)

    def _ensure_event_does_not_exist(self, event: EventId) -> None:
        if event in self.recorded_sync_states:
            logger.info(
                "Found duplicate event creation in the trace for event with "
                "id: %s. Assuming the trace for event deletion wasn't caught "
                "and backfilling it now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                event,
            )
            self.delete_event(event)

    def create_stream(self, stream: StreamId) -> None:
        if stream in self.current_sync_states:
            logger.info(
                "Found duplicate Stream creation in the trace for Stream with "
                "id: %s. PyTorch Streams are only created once, so this "
                "trace entry is ignored.",
                stream,
            )
        else:
            self.host_sync_state[stream] = 0
            self.current_sync_states[stream] = self.host_sync_state.copy()

    def create_event(self, event: EventId) -> None:
        self._ensure_event_does_not_exist(event)
        self.recorded_sync_states[event] = {}

    def delete_event(self, event: EventId) -> None:
        self._ensure_event_exists(event)
        del self.recorded_sync_states[event]

    def update_seq_num(self, stream: StreamId, seq_num: SeqNum) -> None:
        self._ensure_stream_exists(stream)
        self.current_sync_states[stream][stream] = seq_num

    def record_state(self, event: EventId, stream: StreamId) -> None:
        self._ensure_event_exists(event)
        self._ensure_stream_exists(stream)
        self.recorded_sync_states[event] = self.current_sync_states[stream].copy()

    def _state_wait_for_other(
        self, state: dict[StreamId, SeqNum], other: dict[StreamId, SeqNum]
    ) -> None:
        for stream, seq_num in other.items():
            state[stream] = max(state.get(stream, -1), seq_num)

    def stream_wait_for_event(self, stream: StreamId, event: EventId) -> None:
        self._ensure_stream_exists(stream)
        self._ensure_event_exists(event)
        self._state_wait_for_other(
            self.current_sync_states[stream], self.recorded_sync_states[event]
        )

    def all_streams_wait_for_event(self, event: EventId) -> None:
        self._ensure_event_exists(event)
        for stream in self.current_sync_states:
            self.stream_wait_for_event(stream, event)

        self._state_wait_for_other(
            self.host_sync_state, self.recorded_sync_states[event]
        )

    def all_streams_wait_for_stream(self, stream: StreamId) -> None:
        self._ensure_stream_exists(stream)
        for state in self.current_sync_states.values():
            self._state_wait_for_other(state, self.current_sync_states[stream])

        self._state_wait_for_other(
            self.host_sync_state, self.current_sync_states[stream]
        )

    def sync_all_streams(self) -> None:
        for stream, state in self.current_sync_states.items():
            self.host_sync_state[stream] = state[stream]

        for state in self.current_sync_states.values():
            self._state_wait_for_other(state, self.host_sync_state)

    def is_ordered_after(
        self, current_stream: StreamId, seq_num: SeqNum, other_stream: StreamId
    ) -> bool:
        self._ensure_stream_exists(current_stream)
        self._ensure_stream_exists(other_stream)
        return seq_num <= self.current_sync_states[current_stream].get(other_stream, -1)


class EventHandler:
    """Analyzes CSAN trace for synchronization errors.

    Stores information on each stream's synchronizations with other streams as well
    as tensor accesses to determine whether a given kernel launch might cause a
    data race.
    """

    def __init__(self) -> None:
        self.tensors_accessed = _TensorsAccessed()
        self.syncs = StreamSynchronizations()
        self.seq_num: SeqNum = 0
        self.pledged_streams: dict[DataPtr, set[StreamId]] = {}
        self.pending_frees: list[tuple[int, int, _PendingFree]] = []

    def reset(self) -> None:
        self.__init__()

    def _handle_kernel_launch(
        self,
        stream: StreamId,
        read_only: set[DataPtr],
        read_write: set[DataPtr],
        outputs: set[DataPtr],
        operator: str,
        tensor_aliases: dict[int, list[str]],
        byte_ranges: dict[DataPtr, tuple[int, int]] | None = None,
    ) -> list[SynchronizationError]:
        def _get_known_seq(current: Access, previous: Access) -> SeqNum:
            return self.syncs.current_sync_states.get(current.stream, {}).get(
                previous.stream, -1
            )

        def check_conflict(
            data_ptr: DataPtr, current_access: Access, previous_access: Access | None
        ) -> None:
            if previous_access is None:
                return
            if not self.syncs.is_ordered_after(
                current_access.stream, previous_access.seq_num, previous_access.stream
            ):
                error_list.append(
                    UnsynchronizedAccessError(
                        data_ptr,
                        self.tensors_accessed.get_allocation_stack_trace(data_ptr),
                        current_access,
                        previous_access,
                        _get_known_seq(current_access, previous_access),
                    )
                )

        def check_prior_lifecycle(data_ptr: DataPtr, current_access: Access) -> None:
            prior = self.tensors_accessed.accesses[data_ptr].prior_lifecycle
            if prior is None:
                return
            if current_access.type is AccessType.WRITE:
                if prior.write is not None and not self.syncs.is_ordered_after(
                    current_access.stream, prior.write.seq_num, prior.write.stream
                ):
                    error_list.append(
                        AllocatorReuseRaceError(
                            data_ptr,
                            self.tensors_accessed.get_allocation_stack_trace(data_ptr),
                            current_access,
                            prior.write,
                            prior.allocation_stack_trace,
                            _get_known_seq(current_access, prior.write),
                        )
                    )
                for prev_read in prior.reads:
                    if not self.syncs.is_ordered_after(
                        current_access.stream, prev_read.seq_num, prev_read.stream
                    ):
                        error_list.append(
                            AllocatorReuseRaceError(
                                data_ptr,
                                self.tensors_accessed.get_allocation_stack_trace(
                                    data_ptr
                                ),
                                current_access,
                                prev_read,
                                prior.allocation_stack_trace,
                                _get_known_seq(current_access, prev_read),
                            )
                        )
                self.tensors_accessed.accesses[data_ptr].prior_lifecycle = None
            else:
                if prior.write is not None and not self.syncs.is_ordered_after(
                    current_access.stream, prior.write.seq_num, prior.write.stream
                ):
                    error_list.append(
                        AllocatorReuseRaceError(
                            data_ptr,
                            self.tensors_accessed.get_allocation_stack_trace(data_ptr),
                            current_access,
                            prior.write,
                            prior.allocation_stack_trace,
                            _get_known_seq(current_access, prior.write),
                        )
                    )

        error_list: list[SynchronizationError] = []
        self.seq_num += 1
        self.syncs.update_seq_num(stream, self.seq_num)
        stack_trace = traceback.StackSummary.extract(
            traceback.walk_stack(inspect.currentframe()), lookup_lines=False
        )
        # The stack trace generated in this way is in the inverse order, so it must be
        # reversed.
        stack_trace.reverse()

        for data_ptr in read_only:
            self.tensors_accessed.ensure_tensor_exists(data_ptr)
            current_access = Access(
                AccessType.READ,
                self.seq_num,
                stream,
                operator,
                tensor_aliases[data_ptr],
                data_ptr in outputs,
                stack_trace,
                byte_range=byte_ranges.get(data_ptr) if byte_ranges else None,
            )
            check_conflict(
                data_ptr, current_access, self.tensors_accessed.get_write(data_ptr)
            )
            check_prior_lifecycle(data_ptr, current_access)
            self.tensors_accessed.add_read(data_ptr, current_access)

        for data_ptr in read_write:
            self.tensors_accessed.ensure_tensor_exists(data_ptr)
            current_access = Access(
                AccessType.WRITE,
                self.seq_num,
                stream,
                operator,
                tensor_aliases[data_ptr],
                data_ptr in outputs,
                stack_trace,
                byte_range=byte_ranges.get(data_ptr) if byte_ranges else None,
            )
            if self.tensors_accessed.were_there_reads_since_last_write(data_ptr):
                for previous_access in self.tensors_accessed.get_reads(data_ptr):
                    check_conflict(data_ptr, current_access, previous_access)
            else:
                check_conflict(
                    data_ptr, current_access, self.tensors_accessed.get_write(data_ptr)
                )
            check_prior_lifecycle(data_ptr, current_access)
            self.tensors_accessed.set_write(data_ptr, current_access)

        if byte_ranges:
            self._check_view_overlaps(
                stream, read_only, read_write, byte_ranges, error_list, _get_known_seq
            )

        return error_list

    def _check_view_overlaps(
        self,
        stream: StreamId,
        read_only: set[DataPtr],
        read_write: set[DataPtr],
        byte_ranges: dict[DataPtr, tuple[int, int]],
        error_list: list[SynchronizationError],
        _get_known_seq,
    ) -> None:
        current_accesses: list[tuple[DataPtr, tuple[int, int], Access]] = []
        for data_ptr in read_only:
            br = byte_ranges.get(data_ptr)
            if br and br[0] < br[1]:
                reads = self.tensors_accessed.get_reads(data_ptr)
                if reads:
                    current_accesses.append((data_ptr, br, reads[-1]))
        for data_ptr in read_write:
            br = byte_ranges.get(data_ptr)
            if br and br[0] < br[1]:
                write = self.tensors_accessed.get_write(data_ptr)
                if write:
                    current_accesses.append((data_ptr, br, write))

        for cur_ptr, cur_range, cur_access in current_accesses:
            for other_ptr, other_info in self.tensors_accessed.accesses.items():
                if other_ptr == cur_ptr:
                    continue
                if other_info.write is not None:
                    prev_br = other_info.write.byte_range
                    if (
                        prev_br
                        and prev_br[0] < cur_range[1]
                        and cur_range[0] < prev_br[1]
                    ):
                        if not self.syncs.is_ordered_after(
                            cur_access.stream,
                            other_info.write.seq_num,
                            other_info.write.stream,
                        ):
                            error_list.append(
                                OverlappingViewAccessError(
                                    cur_ptr,
                                    cur_range,
                                    other_ptr,
                                    prev_br,
                                    cur_access,
                                    other_info.write,
                                    _get_known_seq(cur_access, other_info.write),
                                )
                            )
                if cur_access.type is AccessType.WRITE:
                    for prev_read in other_info.reads:
                        prev_br = prev_read.byte_range
                        if (
                            prev_br
                            and prev_br[0] < cur_range[1]
                            and cur_range[0] < prev_br[1]
                        ):
                            if not self.syncs.is_ordered_after(
                                cur_access.stream, prev_read.seq_num, prev_read.stream
                            ):
                                error_list.append(
                                    OverlappingViewAccessError(
                                        cur_ptr,
                                        cur_range,
                                        other_ptr,
                                        prev_br,
                                        cur_access,
                                        prev_read,
                                        _get_known_seq(cur_access, prev_read),
                                    )
                                )

    def _handle_event_creation(self, event: EventId) -> None:
        self.syncs.create_event(event)

    def _handle_event_deletion(self, event: EventId) -> None:
        self.syncs.delete_event(event)

    def _handle_event_record(self, event: EventId, stream: StreamId) -> None:
        self.syncs.record_state(event, stream)

    def _handle_event_wait(self, event: EventId, stream: StreamId) -> None:
        self.syncs.stream_wait_for_event(stream, event)

    def _handle_memory_allocation(self, data_ptr: DataPtr, size: int) -> None:
        self.tensors_accessed.ensure_tensor_does_not_exist(data_ptr)
        stack_trace = traceback.StackSummary.extract(
            traceback.walk_stack(inspect.currentframe()), lookup_lines=False
        )
        # The stack trace generated in this way is in the inverse order, so it must be
        # reversed.
        stack_trace.reverse()
        self.tensors_accessed.create_tensor(
            data_ptr,
            stack_trace,
        )
        alloc_end = data_ptr + max(size, 1)
        overlapping = []
        remaining = []
        for start, end, pf in self.pending_frees:
            if start < alloc_end and end > data_ptr:
                overlapping.append(pf)
            else:
                remaining.append((start, end, pf))
        self.pending_frees = remaining
        if overlapping:
            prior_write = None
            prior_reads: list[Access] = []
            alloc_stack = None
            for pf in overlapping:
                if pf.write and pf.write.stream not in pf.pledged_streams:
                    prior_write = pf.write
                prior_reads.extend(
                    r for r in pf.reads if r.stream not in pf.pledged_streams
                )
                if pf.allocation_stack_trace:
                    alloc_stack = pf.allocation_stack_trace
            if prior_write is not None or prior_reads:
                self.tensors_accessed.accesses[
                    data_ptr
                ].prior_lifecycle = _PriorLifecycleAccesses(
                    prior_write, prior_reads, alloc_stack
                )

    def _handle_memory_deallocation(self, data_ptr: DataPtr, size: int) -> None:
        self.tensors_accessed.ensure_tensor_exists(data_ptr)
        info = self.tensors_accessed.accesses[data_ptr]
        pledged = self.pledged_streams.pop(data_ptr, set())
        free_end = data_ptr + max(size, 1)
        self.pending_frees.append(
            (
                data_ptr,
                free_end,
                _PendingFree(
                    write=info.write,
                    reads=list(info.reads),
                    allocation_stack_trace=info.allocation_stack_trace,
                    pledged_streams=pledged,
                ),
            )
        )
        self.tensors_accessed.delete_tensor(data_ptr)

    def _handle_record_stream(self, data_ptr: DataPtr, stream: StreamId) -> None:
        self.pledged_streams.setdefault(data_ptr, set()).add(stream)

    def _handle_stream_creation(self, stream: StreamId) -> None:
        self.syncs.create_stream(stream)

    def _handle_device_synchronization(self) -> None:
        self.syncs.sync_all_streams()
        self.pending_frees.clear()
        for info in self.tensors_accessed.accesses.values():
            info.prior_lifecycle = None

    def _handle_stream_synchronization(self, stream: StreamId) -> None:
        self.syncs.all_streams_wait_for_stream(stream)

    def _handle_event_synchronization(self, event: EventId) -> None:
        self.syncs.all_streams_wait_for_event(event)


def zip_by_key(a: dict[TK, TVa], b: dict[TK, TVb]) -> Iterator[tuple[TK, TVa, TVb]]:
    for arg, value in a.items():
        if arg in b:
            yield arg, value, b[arg]


def zip_arguments(
    schema: torch.FunctionSchema, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Iterator[tuple[torch.Argument, Any]]:
    schema_args = schema.arguments[: len(args)]
    schema_kwargs = {arg.name: arg for arg in schema.arguments[len(args) :]}

    yield from zip(schema_args, args)

    for _, argument, value in zip_by_key(schema_kwargs, kwargs):
        yield (argument, value)


class ArgumentHandler:
    def __init__(self) -> None:
        self.dataptrs_read: set[DataPtr] = set()
        self.dataptrs_written: set[DataPtr] = set()
        self.tensor_aliases: dict[DataPtr, list[str]] = {}
        self.outputs: set[DataPtr] = set()
        self.byte_ranges: dict[DataPtr, tuple[int, int]] = {}

    def _handle_argument(
        self,
        value: Any,
        is_write: bool,
        metadata_only: bool,
        name: str | None = None,
        is_output: bool = False,
    ) -> None:
        if isinstance(value, torch.Tensor) and value.is_cuda:
            # data_ptr() is preferred, but distinguish Tensors with null data_ptr()
            # otherwise two empty Tensors could incorrectly match as a conflict
            raw_ptr = value.data_ptr()
            data_ptr = raw_ptr if raw_ptr else id(value)
            if is_write:
                self.dataptrs_written.add(data_ptr)
            elif not metadata_only:
                self.dataptrs_read.add(data_ptr)

            self.tensor_aliases.setdefault(data_ptr, [])
            if name is not None:
                self.tensor_aliases[data_ptr].append(name)
            if is_output:
                self.outputs.add(data_ptr)

            if raw_ptr:
                br = _tensor_byte_range(value)
                if br[0] < br[1]:
                    existing = self.byte_ranges.get(data_ptr)
                    if existing is not None:
                        self.byte_ranges[data_ptr] = (
                            min(existing[0], br[0]),
                            max(existing[1], br[1]),
                        )
                    else:
                        self.byte_ranges[data_ptr] = br

    def parse_inputs(
        self,
        schema: torch.FunctionSchema,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        is_factory: bool,
    ) -> None:
        for argument, value in zip_arguments(schema, args, kwargs):
            is_write = argument.alias_info is not None and argument.alias_info.is_write
            # A change is metadata only if it is a view or a factory function that
            # reads only metadata
            metadata_only = is_factory or (
                argument.alias_info is not None and not argument.alias_info.is_write
            )
            pytree.tree_map_(
                functools.partial(
                    self._handle_argument,
                    is_write=is_write,
                    name=argument.name,
                    metadata_only=metadata_only,
                ),
                value,
            )

    def parse_outputs(
        self, schema: torch.FunctionSchema, outputs: Any, *, is_factory: bool
    ) -> None:
        for res, value in zip(schema.returns, (outputs,)):
            metadata_only = is_factory or (
                res.alias_info is not None and not res.alias_info.is_write
            )
            pytree.tree_map_(
                functools.partial(
                    self._handle_argument,
                    is_write=not metadata_only,
                    is_output=True,
                    metadata_only=metadata_only,
                ),
                value,
            )


@dataclass
class _GraphProfile:
    reads: set[int]
    writes: set[int]


class CUDASanitizerDispatchMode(TorchDispatchMode):
    def __init__(self) -> None:
        self.event_handler = EventHandler()
        self.accumulated_errors: list[SynchronizationError] = []
        self.accumulate: bool = False
        self._seen_race_sigs: dict[tuple, int] = {}
        self._graph_profiles: dict[int, _GraphProfile] = {}
        self._capturing_graph: Any = None
        self._capture_reads: set[int] = set()
        self._capture_writes: set[int] = set()
        self._graph_hooks_installed = False
        torch._C._activate_gpu_trace()
        gpu_trace.register_callback_for_event_creation(
            self.event_handler._handle_event_creation
        )
        gpu_trace.register_callback_for_event_deletion(
            self.event_handler._handle_event_deletion
        )
        gpu_trace.register_callback_for_event_record(
            self.event_handler._handle_event_record
        )
        gpu_trace.register_callback_for_event_wait(
            self.event_handler._handle_event_wait
        )
        gpu_trace.register_callback_for_memory_allocation(
            self.event_handler._handle_memory_allocation
        )
        gpu_trace.register_callback_for_memory_deallocation(
            self.event_handler._handle_memory_deallocation
        )
        gpu_trace.register_callback_for_stream_creation(
            self.event_handler._handle_stream_creation
        )
        gpu_trace.register_callback_for_device_synchronization(
            self.event_handler._handle_device_synchronization
        )
        gpu_trace.register_callback_for_stream_synchronization(
            self.event_handler._handle_stream_synchronization
        )
        gpu_trace.register_callback_for_event_synchronization(
            self.event_handler._handle_event_synchronization
        )

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func is aten.record_stream.default:
            result = func(*args, **kwargs)
            tensor = args[0]  # pyrefly: ignore
            if tensor.is_cuda and tensor.data_ptr():
                stream_arg = args[1]  # pyrefly: ignore
                cuda_stream = torch.cuda.Stream(
                    stream_id=stream_arg.stream_id,
                    device_index=stream_arg.device_index,
                    device_type=stream_arg.device_type,
                )
                self.event_handler._handle_record_stream(
                    tensor.data_ptr(), cuda_stream.cuda_stream
                )
            return result

        is_factory = bool(FACTORY_FUNCTION_REGEX.match(func._schema.name))

        argument_handler = ArgumentHandler()
        argument_handler.parse_inputs(func._schema, args, kwargs, is_factory=is_factory)

        if self._is_async_nccl_collective(func._schema, args, kwargs):
            return self._dispatch_async_collective(func, argument_handler, args, kwargs)

        outputs = func(*args, **kwargs)

        argument_handler.parse_outputs(func._schema, outputs, is_factory=is_factory)

        errors = self.event_handler._handle_kernel_launch(
            torch.cuda.current_stream().cuda_stream,
            argument_handler.dataptrs_read - argument_handler.dataptrs_written,
            argument_handler.dataptrs_written,
            argument_handler.outputs,
            func._schema,
            argument_handler.tensor_aliases,
            byte_ranges=argument_handler.byte_ranges,
        )
        self._report_errors(errors)

        if self._capturing_graph is not None:
            self._capture_reads |= argument_handler.dataptrs_read
            self._capture_writes |= argument_handler.dataptrs_written

        return outputs

    def _dispatch_async_collective(self, func, argument_handler, args, kwargs):
        """Handle async NCCL collectives with correct stream attribution.

        Two fixes vs. the default path:

        1. **Access ordering**: record BEFORE func so the access seq_num
           is captured by ncclStartEvent (recorded on currentStream inside
           ProcessGroupNCCL) and propagated through ncclEndEvent.  Recording
           AFTER func gives a seq_num past ncclEndEvent's snapshot, so
           work.wait() can't cover it in the vector clock.

        2. **Write classification**: c10d schemas lack alias_info, so
           ArgumentHandler classifies tensor args as read-only.  Collectives
           modify tensors in-place, so all tensor ptrs are treated as
           read+write here.
        """
        all_ptrs = argument_handler.dataptrs_read | argument_handler.dataptrs_written
        errors = self.event_handler._handle_kernel_launch(
            torch.cuda.current_stream().cuda_stream,
            set(),
            all_ptrs,
            set(),
            func._schema,
            argument_handler.tensor_aliases,
            byte_ranges=argument_handler.byte_ranges,
        )
        self._report_errors(errors)

        return func(*args, **kwargs)

    def _report_errors(self, errors: list[SynchronizationError]) -> None:
        if not errors:
            return
        if self.accumulate:
            for error in errors:
                sig = getattr(error, "race_signature", None)
                if sig is not None and sig in self._seen_race_sigs:
                    self._seen_race_sigs[sig] += 1
                else:
                    if sig is not None:
                        self._seen_race_sigs[sig] = 1
                    self.accumulated_errors.append(error)
        else:
            for error in errors:
                print(error, file=sys.stderr)
            raise CUDASanitizerErrors(errors)

    def _is_async_nccl_collective(
        self, schema: torch.FunctionSchema, args: tuple, kwargs: dict
    ) -> bool:
        schema_name = schema.name
        if not schema_name.startswith("c10d::") or schema_name == "c10d::barrier":
            return False
        return self._extract_async_op(schema, args, kwargs)

    @staticmethod
    def _extract_async_op(
        schema: torch.FunctionSchema,
        args: tuple,
        kwargs: dict,
    ) -> bool:
        for i, arg in enumerate(schema.arguments):
            if arg.name == "async_op":
                if i < len(args):
                    return bool(args[i])
                return bool(kwargs.get("async_op", True))
        return True

    def _install_graph_hooks(self) -> None:
        if self._graph_hooks_installed:
            return
        from torch.cuda.graphs import CUDAGraph

        self._orig_capture_begin = CUDAGraph.capture_begin
        self._orig_capture_end = CUDAGraph.capture_end
        self._orig_replay = CUDAGraph.replay
        self._orig_reset = CUDAGraph.reset

        mode = self

        @functools.wraps(CUDAGraph.capture_begin)
        def patched_capture_begin(self, *args, **kwargs):  # pyrefly: ignore
            mode._on_capture_begin(self)
            return mode._orig_capture_begin(self, *args, **kwargs)

        @functools.wraps(CUDAGraph.capture_end)
        def patched_capture_end(self):  # pyrefly: ignore
            mode._orig_capture_end(self)
            mode._on_capture_end(self)

        @functools.wraps(CUDAGraph.replay)
        def patched_replay(self):  # pyrefly: ignore
            mode._orig_replay(self)
            mode._on_graph_replay(self)

        @functools.wraps(CUDAGraph.reset)
        def patched_reset(self):  # pyrefly: ignore
            mode._graph_profiles.pop(id(self), None)
            return mode._orig_reset(self)

        CUDAGraph.capture_begin = patched_capture_begin
        CUDAGraph.capture_end = patched_capture_end
        CUDAGraph.replay = patched_replay
        CUDAGraph.reset = patched_reset
        self._graph_hooks_installed = True

    def _remove_graph_hooks(self) -> None:
        if not self._graph_hooks_installed:
            return
        from torch.cuda.graphs import CUDAGraph

        CUDAGraph.capture_begin = self._orig_capture_begin
        CUDAGraph.capture_end = self._orig_capture_end
        CUDAGraph.replay = self._orig_replay
        CUDAGraph.reset = self._orig_reset
        self._graph_hooks_installed = False

    def _on_capture_begin(self, graph) -> None:
        self._graph_profiles.pop(id(graph), None)
        self._capturing_graph = graph
        self._capture_reads = set()
        self._capture_writes = set()

    def _on_capture_end(self, graph) -> None:
        if self._capturing_graph is not graph:
            return
        self._graph_profiles[id(graph)] = _GraphProfile(
            reads=self._capture_reads,
            writes=self._capture_writes,
        )
        self._capturing_graph = None

    def _on_graph_replay(self, graph) -> None:
        profile = self._graph_profiles.get(id(graph))
        if profile is None:
            return
        errors = self.event_handler._handle_kernel_launch(
            torch.cuda.current_stream().cuda_stream,
            profile.reads - profile.writes,
            profile.writes,
            set(),
            "CUDAGraph.replay()",
            defaultdict(list),
        )
        self._report_errors(errors)


class CUDASanitizer:
    """Manages the lifetime of a CUDASanitizer dispatch mode object.

    The CUDASanitizer class wraps the entering/exiting functions of the dispatch mode
    context manager in the enable function/destructor, respectively. This is to
    explicitly set the lifetime of the dispatch mode object to that of the application.
    This approach was deemed more elegant than using the atexit module.

    Can also be used as a context manager with error accumulation::

        with cuda_sanitizer as san:
            # ... training loop ...
        assert len(san.errors) == 0

    In context-manager mode the sanitizer collects all detected races instead of
    raising on the first one, and automatically disables on exit.
    """

    def __init__(self) -> None:
        self.dispatch = CUDASanitizerDispatchMode()
        self.enabled = False

    def enable(self):
        self.dispatch._install_graph_hooks()
        self.dispatch.__enter__()
        self.enabled = True

    def disable(self):
        self.dispatch._remove_graph_hooks()
        self.dispatch.__exit__(None, None, None)
        self.enabled = False

    @property
    def errors(self) -> list[SynchronizationError]:
        """Errors accumulated during context-manager usage."""
        return self.dispatch.accumulated_errors

    def __enter__(self):
        self.dispatch.event_handler.reset()
        self.dispatch.accumulated_errors.clear()
        self.dispatch._seen_race_sigs.clear()
        self.dispatch.accumulate = True
        if not self.enabled:
            self.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dispatch.accumulate = False
        if self.enabled:
            self.disable()
        return False

    def __del__(self):
        # Since this object lifetime is linked to the `torch.cuda._sanitizer` python
        # module, it often gets deleted as part of the overall `torch` module cleanup
        # At that time, depending on CPython version, the torch.* module might be in
        # different states of being already cleaned up.
        # Similarly other imports might already have been cleaned up so `sys` might
        # be already gone as well.
        # Skip exiting the mode if it outlived the runtime.
        if (sys is not None) and (not sys.is_finalizing()) and self.enabled:
            self.disable()


def enable_cuda_sanitizer():
    """Enable CUDA Sanitizer.

    The sanitizer will begin to analyze low-level CUDA calls invoked by torch functions
    for synchronization errors. All data races found will be printed to the standard
    error output along with stack traces of suspected causes. For best results, the
    sanitizer should be enabled at the very beginning of the program.
    """
    cuda_sanitizer.enable()


def disable_cuda_sanitizer():
    """Disable CUDA Sanitizer."""
    cuda_sanitizer.disable()


cuda_sanitizer = CUDASanitizer()
