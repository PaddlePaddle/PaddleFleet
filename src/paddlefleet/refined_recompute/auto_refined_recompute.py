# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Skip one operator's recompute by retaining its inner autograd graph.

Contract for the boundary function:

* positional ``paddle.Tensor`` inputs only; capture constants by closure;
* returns one Tensor, or a flat tuple/list of ``Tensor``/``None``;
* pure computation with no random-state use -- the recompute pass never runs
  the boundary, so an RNG-consuming op inside it (e.g. dropout) would advance
  the first pass's random state but not the recompute pass's, silently
  misaligning every random op that follows in the same recompute region;
* floating point inputs are treated as requiring gradient and non-floating ones
  as not, so a float input that is semantically constant belongs in the closure;
* not nested inside another boundary.

Only valid inside a full-recompute region, and only when a backward will follow.
The pass is told apart solely by ``tracer._has_grad``, which cannot distinguish
the first recompute pass from plain inference, so the helper does not self-guard:
the call site must fall back to plain execution when recompute is off or during
inference. Otherwise ``_first_forward`` retains a frame that no backward ever
consumes (a leak), and a grad-enabled call outside recompute hits an empty queue.

Frames are paired by queue order, which relies on the scheduler replaying a
boundary's recompute in the same order as its first forward. Schedules that
reorder replay relative to forward are not supported.
"""

from collections import deque
from contextlib import contextmanager

import paddle
from paddle.base import framework

__all__ = ["AutoRefinedRecompute"]

_SHARE_GRAD_FLAG = "FLAGS_share_tensor_for_grad_tensor_holder"


@contextmanager
def _share_grad_holder():
    previous = paddle.get_flags([_SHARE_GRAD_FLAG])
    paddle.set_flags({_SHARE_GRAD_FLAG: True})
    try:
        yield
    finally:
        paddle.set_flags(previous)


def _pack(outputs):
    """Flatten outputs to ``(tensors, spec)``; ``spec`` None means one Tensor."""
    if isinstance(outputs, paddle.Tensor):
        return (outputs,), None
    if not isinstance(outputs, (tuple, list)):
        raise TypeError(
            f"boundary must return Tensor or flat tuple, "
            f"got {type(outputs).__name__}"
        )
    tensors = tuple(item for item in outputs if item is not None)
    if not tensors:
        raise ValueError("boundary returned no Tensor")
    mask = tuple(item is not None for item in outputs)
    return tensors, (type(outputs), mask)


def _unpack(tensors, spec):
    if spec is None:
        return tensors[0]
    container, mask = spec
    remaining = iter(tensors)
    return container([next(remaining) if present else None for present in mask])


class _Frame:
    """One invocation's retained graph, consumed once in backward."""

    __slots__ = ("graph_inputs", "graph_outputs", "spec")

    def __init__(self, graph_inputs, graph_outputs, spec):
        self.graph_inputs = graph_inputs
        self.graph_outputs = graph_outputs
        self.spec = spec

    def release(self):
        self.graph_inputs = ()
        self.graph_outputs = ()
        self.spec = None


class _Reconnect(paddle.autograd.PyLayer):
    """Splice the recompute pass's inputs onto the retained graph."""

    @staticmethod
    def forward(ctx, frame, *inputs):
        # ``frame`` is not a Tensor, so it takes no gradient slot in `backward`.
        graph_inputs = frame.graph_inputs
        if len(graph_inputs) != len(inputs):
            raise RuntimeError(
                f"input count changed: {len(graph_inputs)} -> {len(inputs)}"
            )
        for graph_input, recomputed in zip(graph_inputs, inputs, strict=True):
            # `_share_buffer_to` writes into the first pass's leaf, so a
            # mismatch here would corrupt the retained graph silently. Metadata
            # outlives `_clear_data`, so comparing it needs no stored copy.
            if (
                recomputed.shape != graph_input.shape
                or recomputed.dtype != graph_input.dtype
            ):
                raise RuntimeError(
                    f"input changed between passes: {graph_input.shape} "
                    f"{graph_input.dtype} -> {recomputed.shape} "
                    f"{recomputed.dtype}"
                )
            recomputed._share_buffer_to(graph_input)

        ctx.frame = frame
        ctx.skip_input = tuple(tensor.stop_gradient for tensor in inputs)
        outputs = tuple(output.detach() for output in frame.graph_outputs)
        return outputs[0] if len(outputs) == 1 else outputs

    @staticmethod
    def backward(ctx, *grads):
        frame = ctx.frame
        with _share_grad_holder():
            pairs = [
                (output, grad)
                for output, grad in zip(frame.graph_outputs, grads, strict=True)
                if not output.stop_gradient
            ]
            if pairs:
                outputs, output_grads = zip(*pairs)
                paddle.autograd.backward(list(outputs), list(output_grads))
        # `.grad` is populated only for leaves; `_first_forward` enforces that
        # invariant. Paddle rejects a gradient at a stop_gradient position, so
        # the outer input's own flag decides, not the graph input's.
        input_grads = tuple(
            None if skip or tensor.grad is None else tensor.grad
            for tensor, skip in zip(
                frame.graph_inputs, ctx.skip_input, strict=True
            )
        )
        frame.release()
        return input_grads


class AutoRefinedRecompute:
    """One refined-recompute boundary; create one per call site::

        self._q_proj_rr = AutoRefinedRecompute("q_proj")
        ...
        q = self._q_proj_rr(self.q_proj, x)

    The instance is the boundary's identity, so frames carry no point id or
    layer number. The queue holds more than one frame while several invocations
    are in flight, as under pipeline parallelism.
    """

    def __init__(self, name):
        self.name = name
        self._frames = deque()

    @property
    def pending(self):
        """Frames waiting for their recompute pass."""
        return len(self._frames)

    def __call__(self, function, *inputs):
        for tensor in inputs:
            if not isinstance(tensor, paddle.Tensor):
                raise TypeError(
                    f"[{self.name}] boundary inputs must be Tensors, got "
                    f"{type(tensor).__name__}"
                )
        # From the tracer, not from a recompute config flag: those are set for
        # both passes.
        if framework._dygraph_tracer()._has_grad:
            return self._recompute_forward(inputs)
        return self._first_forward(function, inputs)

    def _first_forward(self, function, inputs):
        graph_inputs = []
        for tensor in inputs:
            graph_input = tensor.detach().view_as(tensor)
            # The first pass runs under full recompute's `no_grad`, where every
            # non-leaf reports stop_gradient=True -- and real boundary inputs
            # are all non-leaves. Inheriting it would build a graph that cannot
            # be differentiated, so decide by dtype.
            graph_input.stop_gradient = not paddle.is_floating_point(
                graph_input
            )
            if not graph_input.is_leaf:
                raise RuntimeError(f"[{self.name}] graph input is not a leaf")
            graph_inputs.append(graph_input)
        graph_inputs = tuple(graph_inputs)

        with paddle.enable_grad():
            outputs = function(*graph_inputs)

        graph_outputs, spec = _pack(outputs)
        detached = tuple(output.detach() for output in graph_outputs)

        # Drop the input data; the recompute pass shares its own buffers back
        # in. The graph and its saved tensors stay alive.
        for graph_input in graph_inputs:
            graph_input._clear_data()
        self._frames.append(_Frame(graph_inputs, graph_outputs, spec))
        return _unpack(detached, spec)

    def _recompute_forward(self, inputs):
        if not self._frames:
            raise RuntimeError(f"[{self.name}] no frame to replay")
        frame = self._frames.popleft()
        # Read before `apply`: `backward` releases the frame.
        spec = frame.spec
        outputs = _Reconnect.apply(frame, *inputs)
        if isinstance(outputs, paddle.Tensor):
            outputs = (outputs,)
        return _unpack(outputs, spec)
