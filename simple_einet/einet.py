import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Type, Union

import numpy as np
import torch
from torch import nn
from torch import distributions as dist

from simple_einet.layers.distributions.abstract_leaf import AbstractLeaf
from simple_einet.layers.einsum import (
    EinsumLayer,
)
from simple_einet.layers.mixing import MixingLayer
from simple_einet.layers.factorized_leaf import FactorizedLeaf
from simple_einet.layers.linsum import LinsumLayer
from simple_einet.sampling_utils import sampling_context, SamplingContext
from simple_einet.layers.sum import SumLayer
from simple_einet.type_checks import check_valid

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EinetConfig:
    """Class for the configuration of an Einet."""

    num_features: int = None  # Number of input features
    num_channels: int = 1  # Number of data input channels per feature
    num_sums: int = 10  # Number of sum nodes at each layer
    num_leaves: int = 10  # Number of distributions for each scope at the leaf layer
    num_repetitions: int = 5  # Number of repetitions
    num_classes: int = 1  # Number of root heads / Number of classes
    depth: int = 1  # Tree depth
    dropout: float = 0.0  # Dropout probabilities for leaves and sum layers
    leaf_type: Type = None  # Type of the leaf base class (Normal, Bernoulli, etc)
    leaf_kwargs: Dict[str, Any] = field(default_factory=dict)  # Parameters for the leaf base class
    layer_type: str = "linsum"  # Indicates the intermediate layer type: linsum or einsum

    def assert_valid(self):
        """Check whether the configuration is valid."""

        # Check that each dimension is valid
        check_valid(self.depth, int, 0)
        check_valid(self.num_features, int, 2)
        check_valid(self.num_channels, int, 1)
        check_valid(self.num_classes, int, 1)
        check_valid(self.num_sums, int, 1)
        check_valid(self.num_repetitions, int, 1)
        check_valid(self.num_leaves, int, 1)
        check_valid(self.dropout, float, 0.0, 1.0, allow_none=True)
        assert self.leaf_type is not None, "EinetConfig.leaf_type parameter was not set!"

        assert isinstance(self.leaf_type, type) and issubclass(
            self.leaf_type, AbstractLeaf
        ), f"Parameter EinetConfig.leaf_base_class must be a subclass type of Leaf but was {self.leaf_type}."

        # If the leaf layer is multivariate distribution, extract its cardinality
        if "cardinality" in self.leaf_kwargs:
            cardinality = self.leaf_kwargs["cardinality"]
        else:
            cardinality = 1

        # Get minimum number of features present at the lowest layer (num_features is the actual input dimension,
        # cardinality in multivariate distributions reduces this dimension since it merges groups of size #cardinality)
        min_num_features = np.ceil(self.num_features // cardinality)
        assert (
            2**self.depth <= min_num_features
        ), f"The tree depth D={self.depth} must be <= {np.floor(np.log2(min_num_features))} (log2(in_features // cardinality))."


class Einet(nn.Module):
    """
    Einet RAT SPN PyTorch implementation with layer-wise tensors.

    See also:
    - RAT SPN: https://arxiv.org/abs/1806.01910
    - EinsumNetworks: https://arxiv.org/abs/2004.06231
    """

    def __init__(self, config: EinetConfig):
        """
        Create a Einet based on a configuration object.

        Args:
            config (EinetConfig): Einet configuration object.
        """
        super().__init__()
        config.assert_valid()
        self.config = config

        # Construct the architecture
        self._build()

    def forward(self, x: torch.Tensor, marginalized_scopes: torch.Tensor = None, skip_leaf = False) -> torch.Tensor:
        """
        Inference pass for the Einet model.

        Args:
          x (torch.Tensor): Input data of shape [N, C, D], where C is the number of input channels (useful for images) and D is the number of features/random variables (H*W for images).
          marginalized_scopes: torch.Tensor:  (Default value = None)

        Returns:
            Log-likelihood tensor of the input: p(X) or p(X | C) if number of classes > 1.
        """

        if not skip_leaf:
            # Add channel dimension if not present
            if x.dim() == 2:  # [N, D]
                x = x.unsqueeze(1)

            if x.dim() == 4:  # [N, C, H, W]
                x = x.view(x.shape[0], self.config.num_channels, -1)

            assert x.dim() == 3
            assert (
                x.shape[1] == self.config.num_channels
            ), f"Number of channels in input ({x.shape[1]}) does not match number of channels specified in config ({self.config.num_channels})."
            assert (
                    x.shape[2] == self.config.num_features
            ), f"Number of features in input ({x.shape[0]}) does not match number of features specified in config ({self.config.num_features})."

            # Apply leaf distributions (replace marginalization indicators with 0.0 first)
            x = self.leaf(x, marginalized_scopes)
        else:
            x = self.leaf.forward(x, marginalized_scopes, skip_leaf=True)

        # Pass through intermediate layers
        x = self._forward_layers(x)

        # Merge results from the different repetitions into the channel dimension
        batch_size, features, channels, repetitions = x.size()
        assert features == 1  # number of features should be 1 at this point
        assert channels == self.config.num_classes

        # If model has multiple reptitions, perform repetition mixing
        if self.config.num_repetitions > 1:
            # Mix repetitions
            x = self.mixing(x)
        else:
            # Remove repetition index
            x = x.squeeze(-1)

        # Remove feature dimension
        x = x.squeeze(1)

        # Final shape check
        assert x.shape == (batch_size, self.config.num_classes)

        return x
    
    def integrate(self, interval: torch.Tensor, marginalized_scopes: torch.Tensor = None) -> torch.Tensor:
        """
        Inference pass for the Einet model.

        Args:
          interval (torch.Tensor): Integration interval of shape [N, C, D, 2], where C is the number of input channels (useful for images) and D is the number of features/random variables (H*W for images).
          marginalized_scopes: torch.Tensor:  (Default value = None)

        Returns:
            The integration of probability density over the interval: ∫p(X)  or ∫p(X | C) if number of classes > 1.
        """
        if interval.dim() == 2: # [D, 2]
            interval = interval.unsqueeze(0)

        # Add channel dimension if not present
        if interval.dim() == 3:  # [N, D, 2]
            interval = interval.unsqueeze(1)

        if interval.dim() == 5:  # [N, C, H, W, 2]
            interval = interval.view(interval.shape[0], self.config.num_channels, interval.shape[2] * interval.shape[3], 2)

        if marginalized_scopes is not None:
            raise NotImplementedError

        assert interval.dim() == 4
        assert (
            interval.shape[1] == self.config.num_channels
        ), f"Number of channels in input ({interval.shape[1]}) does not match number of channels specified in config ({self.config.num_channels})."
        assert (
                interval.shape[2] == self.config.num_features
        ), f"Number of features in input ({interval.shape[0]}) does not match number of features specified in config ({self.config.num_features})."
        assert (
                interval.shape[3] == 2
        ), f"The bounds of each interval should be exactly 2."

        # Apply leaf distributions (replace marginalization indicators with 0.0 first)
        result = self.leaf.integrate(interval)

        # Pass through intermediate layers
        result = self._forward_layers(result)

        # Merge results from the different repetitions into the channel dimension
        batch_size, features, channels, repetitions = result.size()
        assert features == 1  # number of features should be 1 at this point
        assert channels == self.config.num_classes

        # If model has multiple reptitions, perform repetition mixing
        if self.config.num_repetitions > 1:
            # Mix repetitions
            result = self.mixing(result)
        else:
            # Remove repetition index
            result = result.squeeze(-1)

        # Remove feature dimension
        result = result.squeeze(1)

        # Final shape check
        assert result.shape == (batch_size, self.config.num_classes)

        return result.exp()

    def query_per_leaf(self, query: list[float | tuple | None], marginalized_scopes: torch.Tensor = None) -> torch.Tensor:
        """
        Inference pass for the Einet model.

        Args:
          interval (torch.Tensor): Integration interval of shape [N, C, D, 2], where C is the number of input channels (useful for images) and D is the number of features/random variables (H*W for images).
          marginalized_scopes: torch.Tensor:  (Default value = None)

        Returns:
            The integration of probability density over the interval: ∫p(X)  or ∫p(X | C) if number of classes > 1.
        """
        
        d = self.leaf.base_leaf._get_base_distribution()
        loc = d.loc.squeeze()
        scale = d.scale.squeeze()

        assert(loc.shape[0] == len(query), "Dimesion does not match!")

        L = self.config.num_leaves
        R = self.config.num_repetitions
        epsilon = 0.000001

        res = []
        for i in range(len(query)):
            tmp = query[i]
            if tmp is None:
                tsr = torch.full((1, L, R), 0.0)
                res.append(tsr)
            elif type(tmp) is float:
                norm = dist.Normal(loc=loc[i], scale=scale[i])
                tsr = norm.log_prob(torch.tensor([tmp], requires_grad=False)).unsqueeze(0)
                res.append(tsr)
            elif type(tmp) is tuple:
                norm = dist.Normal(loc=loc[i], scale=scale[i])
                a, b = tmp
                low = norm.cdf(torch.tensor([a], requires_grad=False)).unsqueeze(0)
                high = norm.cdf(torch.tensor([b], requires_grad=False)).unsqueeze(0)
                tsr = high - low
                tsr = torch.where(tsr < epsilon, epsilon, tsr)
                tsr = tsr.log()
                res.append(tsr)
            else:
                raise TypeError(f"Unaccpeted Type:{type(tmp)}")

        return torch.vstack(res)

    def _forward_layers(self, x):
        """
        Forward pass through the inner sum and product layers.

        Args:
            x: Input.

        Returns:
            torch.Tensor: Output of the last layer before the root layer.
        """
        # Forward to inner product and sum layers
        for layer in self.layers:
            x = layer(x)
        return x

    def posterior(self, x) -> torch.Tensor:
        """
        Compute the posterior probability logp(y | x) of the data.

        Args:
          x: Data input.

        Returns:
            Posterior logp(y | x).
        """
        assert self.config.num_classes > 1, "Cannot compute posterior without classes."

        # logp(x | y)
        ll_x_g_y = self(x)  # [N, C]

        return posterior(ll_x_g_y, self.config.num_classes)

    def _build(self):
        """Construct the internal architecture of the Einet."""
        # Build the SPN bottom up:
        # Definition from RAT Paper
        # Leaf Region:      Create I leaf nodes
        # Root Region:      Create C sum nodes
        # Internal Region:  Create S sum nodes
        # Partition:        Cross products of all child-regions

        intermediate_layers: List[Union[EinsumLayer, LinsumLayer]] = []

        # Construct layers from top to bottom
        for i in np.arange(start=1, stop=self.config.depth + 1):
            # Choose number of input sum nodes
            # - if this is an intermediate layer, use the number of sum nodes from the previous layer
            # - if this is the first layer, use the number of leaves as the leaf layer is below the first sum layer
            if i < self.config.depth:
                _num_sums_in = self.config.num_sums
            else:
                _num_sums_in = self.config.num_leaves

            # Choose number of output sum nodes
            # - if this is the last layer, use the number of classes
            # - otherwise use the number of sum nodes from the next layer
            if i == 1:
                _num_sums_out = self.config.num_classes
            else:
                _num_sums_out = self.config.num_sums

            # Calculate number of input features: since we represent a binary tree, each layer merges two partitions,
            # hence viewing this from top down we have 2**i input features at the i-th layer
            in_features = 2**i

            if self.config.layer_type == "einsum":
                layer = EinsumLayer(
                    num_features=in_features,
                    num_sums_in=_num_sums_in,
                    num_sums_out=_num_sums_out,
                    num_repetitions=self.config.num_repetitions,
                    dropout=self.config.dropout,
                )
            elif self.config.layer_type == "linsum":
                layer = LinsumLayer(
                    num_features=in_features,
                    num_sums_in=_num_sums_in,
                    num_sums_out=_num_sums_out,
                    num_repetitions=self.config.num_repetitions,
                    dropout=self.config.dropout,
                )
            else:
                raise ValueError(f"Unknown layer type {self.config.layer_type}")

            intermediate_layers.append(layer)

        if self.config.depth == 0:
            # Create a single sum layer
            layer = SumLayer(
                num_sums_in=self.config.num_leaves,
                num_features=1,
                num_sums_out=self.config.num_classes,
                num_repetitions=self.config.num_repetitions,
                dropout=self.config.dropout,
            )
            intermediate_layers.append(layer)

        # Construct leaf
        leaf_num_features_out = intermediate_layers[-1].num_features
        self.leaf = self._build_input_distribution(num_features_out=leaf_num_features_out)

        # List layers in a bottom-to-top fashion
        self.layers: List[Union[EinsumLayer, LinsumLayer]] = nn.ModuleList(reversed(intermediate_layers))

        # If model has multiple reptitions, add repetition mixing layer
        if self.config.num_repetitions > 1:
            self.mixing = MixingLayer(
                num_features=1,
                num_sums_in=self.config.num_repetitions,
                num_sums_out=self.config.num_classes,
                dropout=self.config.dropout,
            )

        # Construct sampling root with weights according to priors for sampling
        if self.config.num_classes > 1:
            self._class_sampling_root = SumLayer(
                num_sums_in=self.config.num_classes,
                num_features=1,
                num_sums_out=1,
                num_repetitions=1,
            )
            self._class_sampling_root.weights = nn.Parameter(
                torch.log(
                    torch.ones(size=(1, self.config.num_classes, 1, 1)) * torch.tensor(1 / self.config.num_classes)
                ),
                requires_grad=False,
            )

    def _build_input_distribution(self, num_features_out: int):
        """Construct the input distribution layer."""
        # Cardinality is the size of the region in the last partitions
        base_leaf = self.config.leaf_type(
            num_features=self.config.num_features,
            num_channels=self.config.num_channels,
            num_leaves=self.config.num_leaves,
            num_repetitions=self.config.num_repetitions,
            **self.config.leaf_kwargs,
        )

        return FactorizedLeaf(
            num_features=base_leaf.out_features,
            num_features_out=num_features_out,
            num_repetitions=self.config.num_repetitions,
            base_leaf=base_leaf,
        )

    @property
    def __device(self):
        """Small hack to obtain the current device."""
        return next(self.parameters()).device

    def mpe(
        self,
        evidence: torch.Tensor = None,
        marginalized_scopes: List[int] = None,
        is_differentiable: bool = False,
    ) -> torch.Tensor:
        """
        Perform MPE given some evidence.

        Args:
            evidence: Input evidence. Must contain some NaN values.
        Returns:
            torch.Tensor: Clone of input tensor with NaNs replaced by MPE estimates.
        """
        return self.sample(
            evidence=evidence, is_mpe=True, marginalized_scopes=marginalized_scopes, is_differentiable=is_differentiable
        )

    def sample(
        self,
        num_samples: int = None,
        class_index=None,
        evidence: torch.Tensor = None,
        is_mpe: bool = False,
        mpe_at_leaves: bool = False,
        temperature_leaves: float = 1.0,
        temperature_sums: float = 1.0,
        marginalized_scopes: List[int] = None,
        is_differentiable: bool = False,
        seed: int = None,
    ):
        """
        Sample from the distribution represented by this SPN.

        Possible valid inputs:

        - `num_samples`: Generates `num_samples` samples.
        - `num_samples` and `class_index (int)`: Generates `num_samples` samples from P(X | C = class_index).
        - `class_index (List[int])`: Generates `len(class_index)` samples. Each index `c_i` in `class_index` is mapped
            to a sample from P(X | C = c_i)
        - `evidence`: If evidence is given, samples conditionally and fill NaN values.

        Args:
            num_samples: Number of samples to generate.
            class_index: Class index. Can be either an int in combination with a value for `num_samples` which will result in `num_samples`
                samples from P(X | C = class_index). Or can be a list of ints which will map each index `c_i` in the
                list to a sample from P(X | C = c_i).
            evidence: Evidence that can be provided to condition the samples. If evidence is given, `num_samples` and
                `class_index` must be `None`. Evidence must contain NaN values which will be imputed according to the
                distribution represented by the SPN. The result will contain the evidence and replace all NaNs with the
                sampled values.
            is_mpe: Flag to perform max sampling (MPE).
            mpe_at_leaves: Flag to perform mpe only at leaves.
            marginalized_scopes: List of scopes to marginalize.
            is_differentiable: Flag to enable differentiable sampling.
            seed: Seed for torch.random.

        Returns:
            torch.Tensor: Samples generated according to the distribution specified by the SPN.

        """
        class_is_given = class_index is not None
        evidence_is_given = evidence is not None
        is_multiclass = self.config.num_classes > 1

        assert not (class_is_given and evidence_is_given), "Cannot provide both, evidence and class indices."
        assert (
            num_samples is None or not evidence_is_given
        ), "Cannot provide both, number of samples to generate (num_samples) and evidence."

        if num_samples is not None:
            assert num_samples > 0, "Number of samples must be > 0."

        # if not is_mpe:
        #     assert ((class_index is not None) and (self.config.num_classes > 1)) or (
        #         (class_index is None) and (self.config.num_classes == 1)
        #     ), "Class index must be given if the number of classes is > 1 or must be none if the number of classes is 1."

        if class_is_given:
            assert (
                self.config.num_classes > 1
            ), f"Class indices are only supported when the number of classes for this model is > 1."

        if evidence is not None:
            # Set n to the number of samples in the evidence
            num_samples = evidence.shape[0]
        elif num_samples is None:
            num_samples = 1

        if is_differentiable:
            indices_out = torch.ones(
                size=(num_samples, 1, 1), dtype=torch.float, device=self.__device, requires_grad=True
            )
            indices_repetition = torch.ones(
                size=(num_samples, 1), dtype=torch.float, device=self.__device, requires_grad=True
            )
        else:
            indices_out = torch.zeros(size=(num_samples, 1), dtype=torch.long, device=self.__device)
            indices_repetition = torch.zeros(size=(num_samples,), dtype=torch.long, device=self.__device)

        ctx = SamplingContext(
            num_samples=num_samples,
            is_mpe=is_mpe,
            mpe_at_leaves=mpe_at_leaves,
            temperature_leaves=temperature_leaves,
            temperature_sums=temperature_sums,
            num_repetitions=self.config.num_repetitions,
            evidence=evidence,
            indices_out=indices_out,
            indices_repetition=indices_repetition,
            is_differentiable=is_differentiable,
        )
        with sampling_context(self, evidence, marginalized_scopes, requires_grad=is_differentiable, seed=seed):
            if self.config.num_classes > 1:
                # If class is given, use it as base index
                if class_index is not None:
                    # Construct indices tensor based on given classes
                    if isinstance(class_index, list):
                        # A list of classes was given, one element for each sample
                        indices = torch.tensor(class_index, device=self.__device).view(-1, 1)
                        if is_differentiable:
                            # TODO: Test this
                            # One hot encode
                            indices = torch.zeros(
                                size=(num_samples, self.config.num_classes, 1), dtype=torch.float, device=self.__device
                            ).scatter_(1, indices.unsqueeze(-1), 1)
                            indices.requireds_grad_(True)  # Enable gradients
                        num_samples = indices.shape[0]
                    else:
                        indices = torch.empty(size=(num_samples, 1), dtype=torch.long, device=self.__device)
                        indices.fill_(class_index)
                        if is_differentiable:
                            # TODO: Test this
                            # One hot encode
                            indices = torch.zeros(
                                size=(num_samples, self.config.num_classes, 1), dtype=torch.float, device=self.__device
                            ).scatter_(1, indices.unsqueeze(-1), 1)
                            indices.requires_grad_(True)  # Enable gradients

                    ctx.indices_out = indices
                else:
                    # Sample class
                    ctx = self._class_sampling_root.sample(ctx=ctx)

            # Save parent indices that were sampled from the sampling root
            if self.config.num_repetitions > 1:
                indices_out_pre_root = ctx.indices_out
                ctx = self.mixing.sample(ctx=ctx)

                # Obtain repetition indices
                if is_differentiable:
                    ctx.indices_repetition = ctx.indices_out.view(num_samples, self.config.num_repetitions)
                else:
                    ctx.indices_repetition = ctx.indices_out.view(num_samples)
                ctx.indices_out = indices_out_pre_root

            # Sample inner layers in reverse order (starting from topmost)
            for layer in reversed(self.layers):
                ctx = layer.sample(ctx=ctx)

            # Sample leaf
            samples = self.leaf.sample(ctx=ctx)

            if evidence is not None:
                # First make a copy such that the original object is not changed
                evidence = evidence.clone()
                shape_evidence = evidence.shape
                evidence = evidence.view_as(samples)
                evidence[:, :, marginalized_scopes] = samples[:, :, marginalized_scopes].to(evidence.dtype)
                evidence = evidence.view(shape_evidence)
                return evidence
            else:
                return samples
            
    def range_sample(
        self,
        num_samples: int = None,
        interval: torch.tensor = None,
        class_index=None,
        evidence: torch.Tensor = None,
        is_mpe: bool = False,
        mpe_at_leaves: bool = False,
        temperature_leaves: float = 1.0,
        temperature_sums: float = 1.0,
        marginalized_scopes: List[int] = None,
        is_differentiable: bool = False,
        seed: int = None,
    ):
        """
        Sample from the distribution represented by this SPN.

        Possible valid inputs:

        - `num_samples`: Generates `num_samples` samples.
        - `num_samples` and `class_index (int)`: Generates `num_samples` samples from P(X | C = class_index).
        - `class_index (List[int])`: Generates `len(class_index)` samples. Each index `c_i` in `class_index` is mapped
            to a sample from P(X | C = c_i)
        - `evidence`: If evidence is given, samples conditionally and fill NaN values.

        Args:
            num_samples: Number of samples to generate.
            class_index: Class index. Can be either an int in combination with a value for `num_samples` which will result in `num_samples`
                samples from P(X | C = class_index). Or can be a list of ints which will map each index `c_i` in the
                list to a sample from P(X | C = c_i).
            evidence: Evidence that can be provided to condition the samples. If evidence is given, `num_samples` and
                `class_index` must be `None`. Evidence must contain NaN values which will be imputed according to the
                distribution represented by the SPN. The result will contain the evidence and replace all NaNs with the
                sampled values.
            is_mpe: Flag to perform max sampling (MPE).
            mpe_at_leaves: Flag to perform mpe only at leaves.
            marginalized_scopes: List of scopes to marginalize.
            is_differentiable: Flag to enable differentiable sampling.
            seed: Seed for torch.random.

        Returns:
            torch.Tensor: Samples generated according to the distribution specified by the SPN.

        """
        class_is_given = class_index is not None
        evidence_is_given = evidence is not None
        is_multiclass = self.config.num_classes > 1

        assert not (class_is_given and evidence_is_given), "Cannot provide both, evidence and class indices."
        assert (
            num_samples is None or not evidence_is_given
        ), "Cannot provide both, number of samples to generate (num_samples) and evidence."

        if num_samples is not None:
            assert num_samples > 0, "Number of samples must be > 0."

        assert interval is not None, "Sampling interval must be given."

        # Add channel dimension if not present
        if interval.dim() == 3:  # [N, D, 2]
            interval = interval.unsqueeze(1)

        if interval.dim() == 5:  # [N, C, H, W, 2]
            interval = interval.view(interval.shape[0], self.config.num_channels, interval.shape[2] * interval.shape[3], 2)


        if marginalized_scopes is not None:
            raise NotImplementedError

        assert interval.dim() == 4
        assert (
            interval.shape[1] == self.config.num_channels
        ), f"Number of channels in input ({interval.shape[1]}) does not match number of channels specified in config ({self.config.num_channels})."
        assert (
                interval.shape[2] == self.config.num_features
        ), f"Number of features in input ({interval.shape[0]}) does not match number of features specified in config ({self.config.num_features})."
        assert (
                interval.shape[3] == 2
        ), f"The bounds of each interval should be exactly 2."
        assert interval.shape[0] == 1, "Unsurported input."

        interval = interval.repeat(num_samples, 1, 1, 1)

        # if not is_mpe:
        #     assert ((class_index is not None) and (self.config.num_classes > 1)) or (
        #         (class_index is None) and (self.config.num_classes == 1)
        #     ), "Class index must be given if the number of classes is > 1 or must be none if the number of classes is 1."

        if class_is_given:
            assert (
                self.config.num_classes > 1
            ), f"Class indices are only supported when the number of classes for this model is > 1."

        if evidence is not None:
            # Set n to the number of samples in the evidence
            num_samples = evidence.shape[0]
        elif num_samples is None:
            num_samples = 1

        if is_differentiable:
            indices_out = torch.ones(
                size=(num_samples, 1, 1), dtype=torch.float, device=self.__device, requires_grad=True
            )
            indices_repetition = torch.ones(
                size=(num_samples, 1), dtype=torch.float, device=self.__device, requires_grad=True
            )
        else:
            indices_out = torch.zeros(size=(num_samples, 1), dtype=torch.long, device=self.__device)
            indices_repetition = torch.zeros(size=(num_samples,), dtype=torch.long, device=self.__device)

        ctx = SamplingContext(
            num_samples=num_samples,
            is_mpe=is_mpe,
            mpe_at_leaves=mpe_at_leaves,
            temperature_leaves=temperature_leaves,
            temperature_sums=temperature_sums,
            num_repetitions=self.config.num_repetitions,
            evidence=evidence,
            indices_out=indices_out,
            indices_repetition=indices_repetition,
            is_differentiable=is_differentiable,
        )
        with sampling_context(self, evidence, marginalized_scopes, requires_grad=is_differentiable, seed=seed):
            if self.config.num_classes > 1:
                # If class is given, use it as base index
                if class_index is not None:
                    # Construct indices tensor based on given classes
                    if isinstance(class_index, list):
                        # A list of classes was given, one element for each sample
                        indices = torch.tensor(class_index, device=self.__device).view(-1, 1)
                        if is_differentiable:
                            # TODO: Test this
                            # One hot encode
                            indices = torch.zeros(
                                size=(num_samples, self.config.num_classes, 1), dtype=torch.float, device=self.__device
                            ).scatter_(1, indices.unsqueeze(-1), 1)
                            indices.requireds_grad_(True)  # Enable gradients
                        num_samples = indices.shape[0]
                    else:
                        indices = torch.empty(size=(num_samples, 1), dtype=torch.long, device=self.__device)
                        indices.fill_(class_index)
                        if is_differentiable:
                            # TODO: Test this
                            # One hot encode
                            indices = torch.zeros(
                                size=(num_samples, self.config.num_classes, 1), dtype=torch.float, device=self.__device
                            ).scatter_(1, indices.unsqueeze(-1), 1)
                            indices.requires_grad_(True)  # Enable gradients

                    ctx.indices_out = indices
                else:
                    # Sample class
                    ctx = self._class_sampling_root.sample(ctx=ctx)

            # Save parent indices that were sampled from the sampling root
            if self.config.num_repetitions > 1:
                indices_out_pre_root = ctx.indices_out
                ctx = self.mixing.sample(ctx=ctx)

                # Obtain repetition indices
                if is_differentiable:
                    ctx.indices_repetition = ctx.indices_out.view(num_samples, self.config.num_repetitions)
                else:
                    ctx.indices_repetition = ctx.indices_out.view(num_samples)
                ctx.indices_out = indices_out_pre_root

            # Sample inner layers in reverse order (starting from topmost)
            for layer in reversed(self.layers):
                ctx = layer.sample(ctx=ctx)

            # Sample leaf
            samples = self.leaf.range_sample(ctx=ctx, interval=interval)

            if evidence is not None:
                # First make a copy such that the original object is not changed
                evidence = evidence.clone()
                shape_evidence = evidence.shape
                evidence = evidence.view_as(samples)
                evidence[:, :, marginalized_scopes] = samples[:, :, marginalized_scopes].to(evidence.dtype)
                evidence = evidence.view(shape_evidence)
                return evidence
            else:
                return samples

    def extra_repr(self) -> str:
        return f"{self.config}"


def posterior(ll_x_g_y: torch.Tensor, num_classes) -> torch.Tensor:
    """
    Compute the posterior probability logp(y | x) of the data.

    Args:
        x: Data input.

    Returns:
        Posterior logp(y | x).
    """
    # logp(y | x) = logp(x, y) - logp(x)
    #             = logp(x | y) + logp(y) - logp(x)
    #             = logp(x | y) + logp(y) - logsumexp(logp(x,y), dim=y)
    ll_y = np.log(1.0 / num_classes)
    ll_x_and_y = ll_x_g_y + ll_y
    ll_x = torch.logsumexp(ll_x_and_y, dim=1, keepdim=True)
    ll_y_g_x = ll_x_g_y + ll_y - ll_x
    return ll_y_g_x

