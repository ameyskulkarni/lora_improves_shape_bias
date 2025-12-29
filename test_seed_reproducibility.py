"""Test that same seed gives same results."""
import torch
from vit_lora import create_vit_lora
from seed_setting import set_seed


def test_seed_reproducibility():
    """Run model twice with same seed, verify identical outputs."""
    seed = 42

    # Run 1
    set_seed(seed, deterministic=True)
    model1, _ = create_vit_lora(lora_r=8)
    input1 = torch.randn(2, 3, 224, 224)
    output1 = model1(input1)

    # Run 2 (same seed)
    set_seed(seed, deterministic=True)
    model2, _ = create_vit_lora(lora_r=8)
    input2 = torch.randn(2, 3, 224, 224)
    output2 = model2(input2)

    # Check
    assert torch.allclose(input1, input2), "Inputs should be identical"
    assert torch.allclose(output1.logits, output2.logits), "Outputs should be identical"

    print("✓ Reproducibility test passed!")


if __name__ == "__main__":
    test_seed_reproducibility()