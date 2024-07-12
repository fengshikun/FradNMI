from abc import abstractmethod, ABCMeta
from torch import nn


class BaseWrapper(nn.Module, metaclass=ABCMeta):
    r"""Base class for model wrappers.

    Children of this class should implement the `forward` method,
    which calls `self.model(z, pos, batch=batch)` at some point.
    Wrappers that are applied before the REDUCE operation should return
    the model's output, `z`, `pos`, `batch` and potentially vector
    features`v`. Wrappers that are applied after REDUCE should only
    return the model's output.
    """

    def __init__(self, model):
        super(BaseWrapper, self).__init__()
        self.model = model

    def reset_parameters(self):
        self.model.reset_parameters()

    @abstractmethod
    def forward(self, z, pos, batch=None):
        return


class AtomFilter(BaseWrapper):
    def __init__(self, model, remove_threshold):
        """
        Wrapper module that filters atoms based on a threshold value.

        Args:
            model (nn.Module): PyTorch model to wrap and filter atoms.
            remove_threshold (float): Threshold value for filtering atoms based on their properties.

        Inputs:
            z (torch.Tensor): Atom properties tensor.
            pos (torch.Tensor): Atom positions tensor.
            batch (torch.Tensor, optional): Batch tensor for grouping atoms. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - x (torch.Tensor): Output tensor from the wrapped model.
                - v (torch.Tensor, optional): Edge feature tensor, if available.
                - z (torch.Tensor): Filtered atom properties tensor.
                - pos (torch.Tensor): Filtered atom positions tensor.
                - batch (torch.Tensor, optional): Filtered batch tensor.
                
        Raises:
            AssertionError: If all atoms in any sample are filtered out, ensuring at least one atom per
                sample exists with Z > remove_threshold.
        """
        super(AtomFilter, self).__init__(model)
        self.remove_threshold = remove_threshold

    def forward(self, z, pos, batch=None):
        x, v, z, pos, batch = self.model(z, pos, batch=batch)

        n_samples = len(batch.unique())

        # drop atoms according to the filter
        atom_mask = z > self.remove_threshold
        x = x[atom_mask]
        if v is not None:
            v = v[atom_mask]
        z = z[atom_mask]
        pos = pos[atom_mask]
        batch = batch[atom_mask]

        assert len(batch.unique()) == n_samples, (
            "Some samples were completely filtered out by the atom filter. "
            f"Make sure that at least one atom per sample exists with Z > {self.remove_threshold}."
        )
        return x, v, z, pos, batch
