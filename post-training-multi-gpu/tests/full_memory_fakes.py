"""Small real HF vision tower inside the existing fake CPU OpenPI model."""
import torch

from test_policy_export import FakeModel


class TinyVisionFakeModel(FakeModel):
    def __init__(self):
        super().__init__()
        from transformers import SiglipVisionConfig, SiglipVisionModel
        config = SiglipVisionConfig(hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, image_size=28, patch_size=14,
            attention_dropout=0.0)
        self.paligemma_with_expert.paligemma.vision_tower = SiglipVisionModel(config)

    def forward(self, observation, actions):
        errors = super().forward(observation, actions + .05 * torch.randn_like(actions))
        vision = self.paligemma_with_expert.paligemma.vision_tower
        # Inputs need no gradient; all vision weights still receive gradients.
        pixels = torch.linspace(-1, 1, 3 * 28 * 28).reshape(1, 3, 28, 28).expand(len(actions), -1, -1, -1)
        features = vision(pixel_values=pixels).last_hidden_state[:, 0, 0]
        return errors + features[:, None, None].square()
