from unittest import TestCase

from itertools import product
from simple_einet.einet import Einet, EinetConfig
import torch
from parameterized import parameterized

from simple_einet.abstract_layers import logits_to_log_weights
from simple_einet.layers.distributions.binomial import Binomial
from simple_einet.layers.linsum import LinsumLayer
from simple_einet.sampling_utils import index_one_hot


class TestEinet(TestCase):
    def make_einet(self, num_classes, num_repetitions):
        config = EinetConfig(
            num_features=self.num_features,
            num_channels=self.num_channels,
            depth=self.depth,
            num_sums=self.num_sums,
            num_leaves=self.num_leaves,
            num_repetitions=num_repetitions,
            num_classes=num_classes,
            leaf_type=self.leaf_type,
            leaf_kwargs=self.leaf_kwargs,
            layer_type="linsum",
            dropout=0.0,
        )
        return Einet(config)

    def setUp(self) -> None:
        self.num_features = 8
        self.num_channels = 3
        self.num_sums = 5
        self.num_leaves = 2
        self.depth = 3
        self.leaf_type = Binomial
        self.leaf_kwargs = {"total_count": 255}

    @parameterized.expand(product([False, True], [1, 3], [1, 4]))
    def test_sampling_shapes(self, differentiable: bool, num_classes: int, num_repetitions: int):
        model = self.make_einet(num_classes=num_classes, num_repetitions=num_repetitions)
        N = 2

        # Sample without evidence
        samples = model.sample(num_samples=N, is_differentiable=differentiable)
        self.assertEqual(samples.shape, (N, self.num_channels, self.num_features))

        # Sample with evidence
        evidence = torch.randint(0, 2, size=(N, self.num_channels, self.num_features))
        samples = model.sample(evidence=evidence, is_differentiable=differentiable)
        self.assertEqual(samples.shape, (N, self.num_channels, self.num_features))

    @parameterized.expand(product([False, True], [1, 3], [1, 4]))
    def test_mpe_shapes(self, differentiable: bool, num_classes: int, num_repetitions: int):
        model = self.make_einet(num_classes=num_classes, num_repetitions=num_repetitions)
        N = 2

        # MPE without evidence
        mpe = model.mpe(is_differentiable=differentiable)
        self.assertEqual(mpe.shape, (1, self.num_channels, self.num_features))

        # MPE with evidence
        evidence = torch.randint(0, 2, size=(N, self.num_channels, self.num_features))
        mpe = model.mpe(evidence=evidence, is_differentiable=differentiable)
        self.assertEqual(mpe.shape, (N, self.num_channels, self.num_features))
