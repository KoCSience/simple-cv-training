from tqdm import tqdm


class TqdmLossTopK(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.postfix_str = ""

    def set_postfix_str_loss_topk(
        self,
        global_step: int,
        loss: float,
        topk_values: tuple[float, ...],
        topk: tuple[int, ...] = (1, 5),
    ) -> str:
        """Build a progress-bar postfix from one value per configured top-k.

        Raises:
            ValueError: If ``topk`` and ``topk_values`` have different lengths.
                Validation happens before resetting or extending the postfix so
                callers never observe a partially updated status string.
        """
        self._validate_topk_pairing(topk_values, topk)

        self.init_postfix_str()
        self.add_step_to_postfix_str(global_step)
        self.add_loss_to_postfix_str(loss)
        self.add_topk_to_postfix_str(topk_values, topk)

        self.set_postfix_str(self.postfix_str)
        return self.postfix_str  # type: ignore[no-any-return]

    def init_postfix_str(self) -> None:
        self.postfix_str = ""

    def add_step_to_postfix_str(self, step: int) -> None:
        self.postfix_str += f"step={step:d}, "

    def add_loss_to_postfix_str(self, loss: float) -> None:
        self.postfix_str += f"loss={loss:6.4e} "

    def add_topk_to_postfix_str(self, topk_values: tuple[float, ...], topk: tuple[int, ...]) -> None:
        """Append top-k values after validating their one-to-one mapping."""
        self._validate_topk_pairing(topk_values, topk)
        for k, value in zip(topk, topk_values, strict=True):
            self.postfix_str += f"top{k}={value:6.2f} "

    @staticmethod
    def _validate_topk_pairing(
        topk_values: tuple[float, ...],
        topk: tuple[int, ...],
    ) -> None:
        """Reject mismatched inputs before a progress string is mutated."""
        if len(topk_values) != len(topk):
            raise ValueError(
                f"topk_values must contain {len(topk)} values to match topk={topk}; got {len(topk_values)}"
            )
