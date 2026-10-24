"""Batched GPU inference for exported Microduck policies.

Every policy exported by ``scripts/export.py`` (and every official policy in
``pollen-robotics/microduck-policies``) is the same tiny graph: the baked-in
observation normalizer (``Sub``, ``Div``) followed by an MLP (``Gemm`` and
``Elu``). onnxruntime only runs these at batch size 1, so this module
re-evaluates the graph with torch to run every vectorized env in one call.
That is what lets frozen ONNX skills drive thousands of training envs.
"""

import onnx
import onnx.numpy_helper
import torch

_SUPPORTED_OPS = {"Sub", "Div", "Gemm", "Elu"}


class OnnxMlp:
    def __init__(self, path: str, device: str | torch.device):
        model = onnx.load(path)
        unknown = {n.op_type for n in model.graph.node} - _SUPPORTED_OPS
        if unknown:
            raise ValueError(f"{path}: unsupported ONNX ops {sorted(unknown)}")
        self.path = path
        self.weights = {
            init.name: torch.tensor(onnx.numpy_helper.to_array(init), device=device)
            for init in model.graph.initializer
        }
        self.nodes = [
            (n.op_type, list(n.input), n.output[0],
             {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute})
            for n in model.graph.node
        ]
        self.input_name = model.graph.input[0].name
        self.output_name = model.graph.output[0].name
        dims = model.graph.input[0].type.tensor_type.shape.dim
        self.obs_dim = dims[-1].dim_value

    @torch.no_grad()
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        values = {self.input_name: obs, **self.weights}
        for op, inputs, output, attrs in self.nodes:
            ins = [values[name] for name in inputs]
            if op == "Sub":
                out = ins[0] - ins[1]
            elif op == "Div":
                out = ins[0] / ins[1]
            elif op == "Elu":
                out = torch.nn.functional.elu(ins[0], alpha=attrs.get("alpha", 1.0))
            else:  # Gemm: alpha * A' @ B' + beta * C
                a = ins[0].T if attrs.get("transA", 0) else ins[0]
                b = ins[1].T if attrs.get("transB", 0) else ins[1]
                out = attrs.get("alpha", 1.0) * (a @ b)
                if len(ins) > 2:
                    out = out + attrs.get("beta", 1.0) * ins[2]
            values[output] = out
        return values[self.output_name]
