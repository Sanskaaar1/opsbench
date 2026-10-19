import unittest

import torch

from operators import OPERATOR_SPECS, arithmetic_intensity, workload_label


class OperatorSpecTest(unittest.TestCase):
    def test_matmul_uses_batched_bsh_and_bhh_inputs(self) -> None:
        spec = OPERATOR_SPECS["matmul"]
        shape = (2, 3, 4)
        left, right = spec.input_factory(shape, torch.float32, torch.device("cpu"))
        output = spec.fn(left, right)

        self.assertEqual(left.shape, (2, 3, 4))
        self.assertEqual(right.shape, (2, 4, 4))
        self.assertEqual(output.shape, (2, 3, 4))
        self.assertEqual(spec.flops_fn(shape), 192.0)
        self.assertEqual(spec.bytes_fn(shape, torch.float32), 320)
        self.assertEqual(arithmetic_intensity(spec, shape, torch.float32), 192 / 320)
        self.assertEqual(
            spec.shapes,
            [
                (1, 512, 2048),
                (8, 512, 2048),
                (32, 512, 2048),
                (8, 128, 2048),
                (8, 4096, 2048),
                (8, 512, 768),
                (8, 512, 8192),
            ],
        )

    def test_softmax_uses_bsh_input_and_last_dimension(self) -> None:
        spec = OPERATOR_SPECS["softmax"]
        shape = (2, 3, 4)
        (inputs,) = spec.input_factory(shape, torch.float32, torch.device("cpu"))
        output = spec.fn(inputs)

        self.assertEqual(output.shape, shape)
        self.assertTrue(torch.allclose(output.sum(dim=-1), torch.ones((2, 3))))
        self.assertEqual(spec.flops_fn(shape), 120.0)
        self.assertEqual(spec.bytes_fn(shape, torch.float32), 192)

    def test_linear_gelu_uses_shared_linear_weight(self) -> None:
        spec = OPERATOR_SPECS["linear_gelu"]
        shape = (2, 3, 4)
        inputs, weight = spec.input_factory(shape, torch.float32, torch.device("cpu"))

        self.assertEqual(weight.shape, (4, 4))
        self.assertEqual(spec.fn(inputs, weight).shape, shape)
        self.assertEqual(spec.flops_fn(shape), 384.0)
        self.assertEqual(spec.bytes_fn(shape, torch.float32), 256)

    def test_layernorm_normalizes_the_hidden_dimension(self) -> None:
        spec = OPERATOR_SPECS["layernorm"]
        shape = (2, 3, 4)
        (inputs,) = spec.input_factory(shape, torch.float32, torch.device("cpu"))
        output = spec.fn(inputs)

        self.assertEqual(output.shape, shape)
        self.assertTrue(torch.allclose(output.mean(dim=-1), torch.zeros((2, 3)), atol=1e-6))
        self.assertEqual(spec.flops_fn(shape), 120.0)
        self.assertEqual(spec.bytes_fn(shape, torch.float32), 192)

    def test_first_milestone_exports_only_plug_in_operators(self) -> None:
        self.assertEqual(set(OPERATOR_SPECS), {"matmul", "softmax", "linear_gelu", "layernorm"})
        self.assertEqual(workload_label((32, 128, 768)), "B32_S128_H768")
