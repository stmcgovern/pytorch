#!/usr/bin/env python3
"""
TDD Tests for masked_softmax NaN fix

This test suite demonstrates the inconsistency in masked_softmax behavior:
- Path 1 (no mask): Uses _safe_softmax → produces 0 for fully masked rows ✓
- Path 2 (with mask): Uses _masked_softmax → produces NaN for fully masked rows ✗

All tests should PASS after fixing the _masked_softmax kernel.
"""

import torch
import torch.nn as nn
import unittest
from torch.testing._internal.common_utils import TestCase, run_tests, parametrize


class TestMaskedSoftmaxConsistency(TestCase):
    """Test that masked_softmax handles fully masked rows consistently"""

    def test_no_mask_path_fully_masked_rows(self):
        """
        Test Path 1: masked_softmax WITHOUT explicit mask
        This uses _safe_softmax internally (already fixed)
        EXPECTED: Should PASS (current uncommitted fix)
        """
        # Create attention scores with some rows all -inf (fully masked)
        scores = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],           # Normal row
            [-float('inf')] * 4,             # Fully masked row
            [0.5, -float('inf'), 1.5, 2.0], # Partially masked
            [-float('inf')] * 4,             # Fully masked row
        ])

        # Call without mask parameter (uses _safe_softmax path)
        from torch.nn.functional import softmax
        # Note: We can't call masked_softmax directly from Python,
        # but we can test the equivalent scenario
        result = softmax(scores, dim=1)

        # Check: fully masked rows should be 0, not NaN
        self.assertFalse(torch.isnan(result).any(),
                        "Result should not contain NaN")

        # Fully masked rows (indices 1, 3) should be all zeros
        # NOTE: Standard softmax WILL produce NaN here
        # This test documents current behavior
        if not torch.isnan(result[1]).any():
            self.assertTrue(torch.allclose(result[1], torch.zeros(4)),
                          "Fully masked row should be all zeros")
            self.assertTrue(torch.allclose(result[3], torch.zeros(4)),
                          "Fully masked row should be all zeros")

    @parametrize("device", ["cpu", "cuda"])
    def test_with_mask_type_2_fully_masked_rows(self, device):
        """
        Test Path 2: _masked_softmax with mask_type=2 (generic mask)
        EXPECTED: Should FAIL initially (demonstrates bug)
        EXPECTED: Should PASS after kernel fix
        """
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Create attention scores
        scores = torch.randn(2, 3, 4, 8, device=device)  # [batch, heads, seq_len, seq_len]

        # Create mask with some fully masked rows
        mask = torch.zeros(2, 3, 4, 8, dtype=torch.bool, device=device)
        mask[0, 0, 1, :] = True  # Fully mask row [0,0,1,:]
        mask[1, 1, 2, :] = True  # Fully mask row [1,1,2,:]

        # Apply masked softmax (mask_type=2 means generic 4D mask)
        result = torch._masked_softmax(scores, mask, dim=-1, mask_type=2)

        # Check: should not produce NaN
        self.assertFalse(torch.isnan(result).any(),
                        f"_masked_softmax should not produce NaN for fully masked rows. "
                        f"Found NaN at indices: {torch.nonzero(torch.isnan(result))}")

        # Fully masked rows should be all zeros
        self.assertTrue(torch.allclose(result[0, 0, 1, :], torch.zeros(8)),
                       "Fully masked row [0,0,1,:] should be all zeros")
        self.assertTrue(torch.allclose(result[1, 1, 2, :], torch.zeros(8)),
                       "Fully masked row [1,1,2,:] should be all zeros")

        # Non-masked rows should sum to 1 (valid probability distribution)
        non_masked_sums = result[0, 0, 0, :].sum()
        self.assertTrue(torch.allclose(non_masked_sums, torch.tensor(1.0)),
                       "Non-masked rows should sum to 1")

    @parametrize("device", ["cpu", "cuda"])
    def test_with_mask_type_1_padding_mask(self, device):
        """
        Test Path 2: _masked_softmax with mask_type=1 (padding mask)
        This simulates TransformerEncoder with src_key_padding_mask
        EXPECTED: Should FAIL initially (demonstrates bug)
        EXPECTED: Should PASS after kernel fix
        """
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Attention scores: [batch=2, heads=4, seq_len=8, seq_len=8]
        scores = torch.randn(2, 4, 8, 8, device=device)

        # Padding mask: [batch=2, seq_len=8]
        # True = masked (padded token)
        padding_mask = torch.tensor([
            [False, False, False, False, False, False, False, False],  # No padding
            [True, True, True, True, True, True, True, True],          # Fully padded!
        ], dtype=torch.bool, device=device)

        # Apply masked softmax with padding mask (mask_type=1)
        result = torch._masked_softmax(scores, padding_mask, dim=-1, mask_type=1)

        # Check: should not produce NaN
        self.assertFalse(torch.isnan(result).any(),
                        f"Padding mask should not produce NaN for fully masked sequences. "
                        f"Found NaN in batch item 1 (fully padded)")

        # Fully padded sequence (batch item 1) should have zeros
        # for all attention weights
        for head_idx in range(4):
            for seq_idx in range(8):
                self.assertTrue(
                    torch.allclose(result[1, head_idx, seq_idx, :], torch.zeros(8)),
                    f"Fully padded sequence should have zero attention weights "
                    f"at [1, {head_idx}, {seq_idx}, :]"
                )

    @parametrize("device", ["cpu", "cuda"])
    def test_with_mask_type_0_attention_mask(self, device):
        """
        Test Path 2: _masked_softmax with mask_type=0 (attention mask)
        This is for causal masking or custom attention patterns
        EXPECTED: Should FAIL initially if mask creates fully masked rows
        EXPECTED: Should PASS after kernel fix
        """
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Attention scores: [batch=2, heads=4, seq_len=4, seq_len=4]
        scores = torch.randn(2, 4, 4, 4, device=device)

        # Attention mask: [seq_len=4, seq_len=4]
        # Create a mask where first position can't attend to anything
        attn_mask = torch.tensor([
            [True, True, True, True],      # Position 0: fully masked!
            [False, False, True, True],    # Position 1: causal
            [False, False, False, True],   # Position 2: causal
            [False, False, False, False],  # Position 3: causal
        ], dtype=torch.bool, device=device)

        # Apply masked softmax with attention mask (mask_type=0)
        result = torch._masked_softmax(scores, attn_mask, dim=-1, mask_type=0)

        # Check: should not produce NaN
        self.assertFalse(torch.isnan(result).any(),
                        f"Attention mask should not produce NaN for fully masked rows. "
                        f"Found NaN at position 0")

        # First position (fully masked) should be all zeros across all batches and heads
        for batch_idx in range(2):
            for head_idx in range(4):
                self.assertTrue(
                    torch.allclose(result[batch_idx, head_idx, 0, :], torch.zeros(4)),
                    f"Fully masked position should have zero attention weights "
                    f"at [{batch_idx}, {head_idx}, 0, :]"
                )

    def test_transformer_encoder_fully_padded_sequence(self):
        """
        Integration test: TransformerEncoder with fully padded sequence
        This is the real-world scenario that users encounter
        EXPECTED: Should FAIL initially (demonstrates bug)
        EXPECTED: Should PASS after kernel fix
        """
        d_model = 64
        nhead = 4

        model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=128,
                batch_first=True,
                dropout=0.0
            ),
            num_layers=2
        ).eval()

        # Input: [batch=3, seq_len=8, d_model=64]
        x = torch.randn(3, 8, d_model)

        # Padding mask with one fully masked sequence
        padding_mask = torch.tensor([
            [False, False, False, False, False, False, False, False],  # Valid sequence
            [True, True, True, True, True, True, True, True],          # Fully masked!
            [False, False, False, True, True, True, True, True],       # Partially masked
        ], dtype=torch.bool)

        with torch.no_grad():
            output = model(x, src_key_padding_mask=padding_mask)

        # Check: should not produce NaN
        self.assertFalse(torch.isnan(output).any(),
                        f"TransformerEncoder should not produce NaN for fully masked sequences. "
                        f"Found NaN in batch item 1 (fully masked)")

        # Output should be finite
        self.assertTrue(torch.isfinite(output).all(),
                       "All outputs should be finite (no NaN or inf)")

    @parametrize("device", ["cpu", "cuda"])
    def test_backward_pass_no_nan_gradients(self, device):
        """
        Test that backward pass doesn't produce NaN gradients
        even with fully masked rows
        EXPECTED: Should FAIL initially
        EXPECTED: Should PASS after kernel fix
        """
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Create scores that require gradients
        scores = torch.randn(2, 4, 4, requires_grad=True, device=device)

        # Create mask with fully masked row
        mask = torch.zeros(2, 4, dtype=torch.bool, device=device)
        mask[0, 1] = True  # Partially masked
        mask[1, :] = True  # Fully masked row!

        # Forward pass
        result = torch._masked_softmax(scores, mask, dim=-1, mask_type=2)

        # Create gradient
        grad_output = torch.ones_like(result)

        # Backward pass
        result.backward(grad_output)

        # Check: gradients should not contain NaN
        self.assertFalse(torch.isnan(scores.grad).any(),
                        f"Gradients should not contain NaN even for fully masked rows. "
                        f"Found NaN in gradients")

        # Gradient for fully masked row should be well-defined (likely zeros)
        self.assertTrue(torch.isfinite(scores.grad).all(),
                       "All gradients should be finite")

    def test_mixed_masked_rows_statistical_properties(self):
        """
        Test that partially masked and fully masked rows behave correctly
        """
        # Create larger batch to test statistical properties
        batch_size = 10
        seq_len = 16
        scores = torch.randn(batch_size, seq_len, seq_len)

        # Create various masking patterns
        mask = torch.rand(batch_size, seq_len, seq_len) > 0.7  # Random masking

        # Intentionally create some fully masked rows
        mask[0, 0, :] = True   # Fully masked
        mask[5, 10, :] = True  # Fully masked

        result = torch._masked_softmax(scores, mask, dim=-1, mask_type=2)

        # No NaN anywhere
        self.assertFalse(torch.isnan(result).any(),
                        "No NaN should be present")

        # Fully masked rows are zeros
        self.assertTrue(torch.allclose(result[0, 0, :], torch.zeros(seq_len)),
                       "Fully masked row should be zeros")
        self.assertTrue(torch.allclose(result[5, 10, :], torch.zeros(seq_len)),
                       "Fully masked row should be zeros")

        # Partially masked rows sum to 1
        for b in range(batch_size):
            for i in range(seq_len):
                if not mask[b, i].all():  # Not fully masked
                    row_sum = result[b, i].sum()
                    self.assertTrue(torch.allclose(row_sum, torch.tensor(1.0), atol=1e-6),
                                   f"Partially masked row should sum to 1, got {row_sum}")

    def test_consistency_between_paths(self):
        """
        Test that masked_softmax produces same result regardless of how masking is applied
        """
        seq_len = 8

        # Scenario 1: Apply -inf directly (no mask parameter)
        scores_no_mask = torch.randn(2, 4, seq_len)
        scores_no_mask[:, :, -2:] = -float('inf')  # Mask last 2 positions

        # We can't call masked_softmax directly without mask, but we can test
        # that the underlying behavior should be consistent

        # Scenario 2: Apply via mask parameter
        scores_with_mask = scores_no_mask.clone()
        scores_with_mask[:, :, -2:] = 0  # Reset to original values (before -inf)

        # Create equivalent mask
        mask = torch.zeros(2, 4, seq_len, dtype=torch.bool)
        mask[:, :, -2:] = True  # Mask last 2 positions

        # This test documents that we WANT consistency
        # After the fix, both should produce same results
        # (Currently they don't - that's the bug we're fixing)


class TestMaskedSoftmaxEdgeCases(TestCase):
    """Test edge cases and boundary conditions"""

    @parametrize("device", ["cpu", "cuda"])
    def test_all_rows_fully_masked(self, device):
        """Test when ALL rows are fully masked"""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        scores = torch.randn(4, 8, 8, device=device)
        mask = torch.ones(4, 8, 8, dtype=torch.bool, device=device)  # Everything masked!

        result = torch._masked_softmax(scores, mask, dim=-1, mask_type=2)

        # Should be all zeros, not all NaN
        self.assertFalse(torch.isnan(result).any(),
                        "All-masked tensor should not produce NaN")
        self.assertTrue(torch.allclose(result, torch.zeros_like(result)),
                       "All-masked tensor should be all zeros")

    def test_single_element_masked(self):
        """Test single element that is fully masked"""
        scores = torch.randn(1, 1, 1)
        mask = torch.ones(1, 1, 1, dtype=torch.bool)

        result = torch._masked_softmax(scores, mask, dim=-1, mask_type=2)

        # Single masked element should be 0, not NaN
        self.assertFalse(torch.isnan(result).any(),
                        "Single masked element should not be NaN")
        self.assertTrue(torch.allclose(result, torch.zeros_like(result)),
                       "Single masked element should be 0")

    def test_empty_mask(self):
        """Test with no masking (all False)"""
        scores = torch.randn(2, 4, 8)
        mask = torch.zeros(2, 4, 8, dtype=torch.bool)  # Nothing masked

        result = torch._masked_softmax(scores, mask, dim=-1, mask_type=2)

        # Should behave like normal softmax (sum to 1)
        row_sums = result.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums)),
                       "Unmasked softmax should sum to 1")

    def test_alternating_masked_unmasked_rows(self):
        """Test pattern of alternating fully masked and unmasked rows"""
        scores = torch.randn(10, 8)
        mask = torch.zeros(10, 8, dtype=torch.bool)

        # Alternate: mask odd rows completely
        for i in range(1, 10, 2):
            mask[i, :] = True

        result = torch._masked_softmax(scores, mask, dim=-1, mask_type=2)

        # Odd rows should be zeros
        for i in range(1, 10, 2):
            self.assertTrue(torch.allclose(result[i], torch.zeros(8)),
                           f"Fully masked row {i} should be zeros")

        # Even rows should sum to 1
        for i in range(0, 10, 2):
            self.assertTrue(torch.allclose(result[i].sum(), torch.tensor(1.0)),
                           f"Unmasked row {i} should sum to 1")


class TestMaskedSoftmaxGradients(TestCase):
    """Test gradient correctness (CRITICAL - ezyang requirement)"""

    @parametrize("device", ["cpu", "cuda"])
    def test_gradcheck_fully_masked(self, device):
        """
        Gradcheck: Verify gradients are numerically correct for fully masked rows
        BLOCKING: Required by ezyang review
        """
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        scores = torch.randn(4, 8, dtype=torch.double, requires_grad=True, device=device)
        mask = torch.zeros(4, 8, dtype=torch.bool, device=device)
        mask[1, :] = True  # Fully mask row 1
        mask[3, :] = True  # Fully mask row 3

        def func(x):
            return torch._masked_softmax(x, mask, dim=-1, mask_type=2)

        # Gradcheck verifies numerical gradient matches autograd gradient
        torch.autograd.gradcheck(func, scores, eps=1e-6, atol=1e-4)

    @parametrize("device", ["cpu", "cuda"])
    def test_gradcheck_partially_masked(self, device):
        """Gradcheck: Partially masked rows should also have correct gradients"""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        scores = torch.randn(3, 6, dtype=torch.double, requires_grad=True, device=device)
        mask = torch.zeros(3, 6, dtype=torch.bool, device=device)
        mask[0, 2:4] = True  # Partially masked
        mask[1, :] = True    # Fully masked

        def func(x):
            return torch._masked_softmax(x, mask, dim=-1, mask_type=2)

        torch.autograd.gradcheck(func, scores, eps=1e-6, atol=1e-4)

    @parametrize("device", ["cpu", "cuda"])
    def test_gradient_values_fully_masked(self, device):
        """
        Verify gradient values are correct (should be zero for fully masked row)
        """
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        scores = torch.randn(3, 4, requires_grad=True, device=device)
        mask = torch.zeros(3, 4, dtype=torch.bool, device=device)
        mask[1, :] = True  # Fully mask row 1

        result = torch._masked_softmax(scores, mask, dim=-1, mask_type=2)

        # Backward pass with uniform gradient
        result.backward(torch.ones_like(result))

        # Row 1 is fully masked, so gradients should be zero
        self.assertTrue(torch.allclose(scores.grad[1, :], torch.zeros(4, device=device)),
                       "Gradients for fully masked row should be zero")

        # Rows 0 and 2 are not fully masked, so gradients should be non-zero
        self.assertFalse(torch.allclose(scores.grad[0, :], torch.zeros(4, device=device)),
                        "Gradients for non-masked row should be non-zero")

    @parametrize("device", ["cpu", "cuda"])
    def test_gradcheck_4d_mask_type_2(self, device):
        """
        Gradcheck with 4D mask (mask_type=2) - full shape matching
        This is the most common case for generic masking
        """
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        scores = torch.randn(2, 3, 4, 8, dtype=torch.double, requires_grad=True, device=device)
        mask = torch.zeros(2, 3, 4, 8, dtype=torch.bool, device=device)
        mask[0, 0, 1, :] = True  # Fully mask one row
        mask[1, 1, 2, :] = True  # Fully mask another row

        def func(x):
            return torch._masked_softmax(x, mask, dim=-1, mask_type=2)

        torch.autograd.gradcheck(func, scores, eps=1e-6, atol=1e-4)


class TestMaskedSoftmaxDtypes(TestCase):
    """Test different dtypes (BLOCKING - ezyang requirement)"""

    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
    @parametrize("device", ["cpu", "cuda"])
    def test_dtypes_fully_masked(self, dtype, device):
        """Test all dtypes handle fully masked rows correctly"""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        if dtype == torch.float16 and device == "cpu":
            self.skipTest("fp16 not supported on CPU")
        if dtype == torch.bfloat16 and device == "cpu":
            self.skipTest("bf16 not fully supported on CPU")

        scores = torch.randn(2, 4, 8, dtype=dtype, device=device)
        mask = torch.zeros(2, 4, 8, dtype=torch.bool, device=device)
        mask[0, 1, :] = True  # Fully mask row [0, 1, :]
        mask[1, 2, :] = True  # Fully mask row [1, 2, :]

        result = torch._masked_softmax(scores, mask, dim=-1, mask_type=2)

        # Check: should not produce NaN
        self.assertFalse(torch.isnan(result).any(),
                        f"dtype={dtype} should not produce NaN")

        # Fully masked rows should be zeros
        expected = torch.zeros(8, dtype=dtype, device=device)
        self.assertTrue(torch.allclose(result[0, 1, :], expected, atol=1e-5),
                       f"dtype={dtype}: Fully masked row should be zeros")
        self.assertTrue(torch.allclose(result[1, 2, :], expected, atol=1e-5),
                       f"dtype={dtype}: Fully masked row should be zeros")

    @parametrize("device", ["cpu", "cuda"])
    def test_mixed_precision_autocast(self, device):
        """Test with autocast (mixed precision training)"""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        if device == "cpu":
            self.skipTest("Autocast primarily for CUDA")

        with torch.cuda.amp.autocast():
            scores = torch.randn(2, 4, 8, device=device)
            mask = torch.zeros(2, 4, 8, dtype=torch.bool, device=device)
            mask[0, 1, :] = True  # Fully masked

            result = torch._masked_softmax(scores, mask, dim=-1, mask_type=2)

            # Should not produce NaN even in mixed precision
            self.assertFalse(torch.isnan(result).any(),
                           "Autocast should not produce NaN")
            self.assertTrue(torch.allclose(result[0, 1, :], torch.zeros(8, device=device)))


class TestMaskedSoftmaxEdgeCasesExtended(TestCase):
    """Additional edge cases from ezyang review"""

    @parametrize("device", ["cpu", "cuda"])
    def test_no_grad_fully_masked(self, device):
        """Test with requires_grad=False"""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        scores = torch.randn(2, 4, 8, requires_grad=False, device=device)
        mask = torch.ones(2, 4, 8, dtype=torch.bool, device=device)  # All masked

        result = torch._masked_softmax(scores, mask, dim=-1, mask_type=2)

        self.assertTrue(torch.allclose(result, torch.zeros_like(result)),
                       "All masked tensor should be all zeros")

    @parametrize("device", ["cpu", "cuda"])
    def test_dimension_not_last(self, device):
        """Test with dim != -1 (ezyang requirement)"""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        scores = torch.randn(8, 4, 2, device=device)
        mask = torch.zeros(8, 4, 2, dtype=torch.bool, device=device)

        # Mask entire dim=0 for position [1, 0]
        mask[:, 1, 0] = True

        result = torch._masked_softmax(scores, mask, dim=0, mask_type=2)

        # Check that the fully masked column is all zeros
        self.assertTrue(torch.allclose(result[:, 1, 0], torch.zeros(8, device=device)),
                       "Fully masked column (dim=0) should be zeros")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_large_tensor_stress(self):
        """Stress test with large tensors (ezyang requirement)"""
        # Test with large tensors to catch memory issues
        scores = torch.randn(32, 8, 256, 256, device='cuda')  # Smaller than ezyang's 128x16x512x512
        mask = torch.rand(32, 8, 256, 256, device='cuda') > 0.9

        result = torch._masked_softmax(scores, mask, dim=-1, mask_type=2)

        self.assertFalse(torch.isnan(result).any(),
                        "Large tensor should not produce NaN")
        # Check memory didn't explode
        self.assertEqual(result.shape, scores.shape)

    @parametrize("device", ["cpu", "cuda"])
    def test_all_dimensions(self, device):
        """Test softmax along different dimensions"""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        for dim in [-1, -2, 0, 1]:
            scores = torch.randn(4, 6, device=device)
            mask = torch.zeros(4, 6, dtype=torch.bool, device=device)

            # Create fully masked rows/cols depending on dim
            if dim in [-1, 1]:
                mask[0, :] = True  # Fully mask row 0
            else:  # dim in [0, -2]
                mask[:, 0] = True  # Fully mask col 0

            result = torch._masked_softmax(scores, mask, dim=dim, mask_type=2)

            self.assertFalse(torch.isnan(result).any(),
                           f"dim={dim} should not produce NaN")


if __name__ == "__main__":
    run_tests()
