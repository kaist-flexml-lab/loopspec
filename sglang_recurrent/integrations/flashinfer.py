"""Use FlashInfer TinyGEMM whenever a recurrent linear call supports it."""

import logging

import torch


logger = logging.getLogger(__name__)

_installed = False
_tinygemm_bf16 = None


def _eligible(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> bool:
    """Return whether these linear operands satisfy TinyGEMM's tensor constraints.

    Expected shapes are input ``(*batch, in_features)``, weight
    ``(out_features, in_features)``, and optional bias ``(out_features,)``.
    CUDA capability and kernel availability are checked during installation.
    """

    if (
        input_.ndim == 0
        or input_.shape[-1] == 0
        or weight.ndim != 2
    ):
        return False
    # Flatten all leading dimensions to count the rows passed to the kernel.
    rows = input_.numel() // input_.shape[-1]
    output_features, input_features = weight.shape
    return (
        input_.shape[-1] == input_features
        and input_.device.type == "cuda"
        and weight.device == input_.device
        and input_.dtype == weight.dtype == torch.bfloat16
        and input_.is_contiguous()
        and weight.is_contiguous()
        and 1 <= rows <= 16
        and input_features % 64 == 0
        and output_features % 16 == 0
        and (
            bias is None
            or (
                bias.device == input_.device
                and bias.dtype == torch.bfloat16
                and bias.is_contiguous()
                and bias.shape == (output_features,)
            )
        )
    )


def _tinygemm_linear(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Compute ``input_ @ weight.T + bias`` with an installed TinyGEMM kernel.

    Requires operands accepted by ``_eligible``: input ``(*batch, in_features)``,
    weight ``(out_features, in_features)``, and optional bias ``(out_features,)``.
    Returns ``(*batch, out_features)`` with the input's dtype and device.
    """

    rows = input_.numel() // input_.shape[-1]
    output = torch.empty(
        (rows, weight.shape[0]),
        dtype=input_.dtype,
        device=input_.device,
    )
    # The kernel writes into output and accepts a two-dimensional input.
    _tinygemm_bf16(
        input_.reshape(rows, input_.shape[-1]), weight, output, bias
    )
    # Restore the leading dimensions expected by the linear layer's caller.
    return output.view(*input_.shape[:-1], weight.shape[0])


def install_linear_backend(
    *,
    device: int | str | torch.device | None = None,
) -> dict[str, str | tuple[int, int]]:
    """Install TinyGEMM when supported, otherwise retain SGLang's backend.

    ``device`` selects the CUDA device by index, string, or torch.device;
    ``None`` uses the current CUDA device. Installation patches SGLang's
    unquantized linear method once for the entire process.

    Returns a status dict with ``active`` ("tinygemm" or "torch"), ``reason``,
    and, when queried, ``device_capability`` as a ``(major, minor)`` tuple.
    The "torch" status means SGLang's existing backend is retained.
    """
    global _installed, _tinygemm_bf16

    if _installed:
        return {
            "active": "tinygemm",
            "reason": "already installed process-wide",
        }
    if not torch.cuda.is_available():
        return {
            "active": "torch",
            "reason": "CUDA is unavailable",
        }

    device = torch.cuda.current_device() if device is None else device
    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(device)
    try:
        from flashinfer.gemm import tinygemm_bf16
    except Exception as error:
        reason = f"TinyGEMM unavailable: {error}"
        logger.warning(reason)
        return {
            "active": "torch",
            "reason": reason,
            "device_capability": capability,
        }
    # FlashInfer expects an SM number, e.g. (9, 0) becomes 90.
    capability_number = capability[0] * 10 + capability[1]
    if not tinygemm_bf16.is_compute_capability_supported(capability_number):
        return {
            "active": "torch",
            "reason": (
                "FlashInfer TinyGEMM does not support "
                f"sm{capability[0]}{capability[1]}"
            ),
            "device_capability": capability,
        }

    # Probe a supported shape before patching SGLang. Synchronizing also exposes
    # asynchronous CUDA failures here, so a failed probe keeps the old backend.
    try:
        _tinygemm_bf16 = tinygemm_bf16
        input_ = torch.zeros((2, 64), dtype=torch.bfloat16, device=device)
        weight = torch.zeros((16, 64), dtype=torch.bfloat16, device=device)
        output = torch.empty((2, 16), dtype=torch.bfloat16, device=device)
        tinygemm_bf16(input_, weight, output)
        torch.cuda.synchronize(device)
    except Exception as error:
        _tinygemm_bf16 = None
        reason = f"TinyGEMM unavailable: {error}"
        logger.warning(reason)
        return {
            "active": "torch",
            "reason": reason,
            "device_capability": capability,
        }

    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    original_apply = UnquantizedLinearMethod.apply

    def apply(
        self: UnquantizedLinearMethod,
        layer: torch.nn.Module,
        input_: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply a linear layer through TinyGEMM or the saved SGLang method.

        ``layer.weight`` has shape ``(out_features, in_features)``, ``input_``
        has shape ``(*batch, in_features)``, and optional ``bias`` has shape
        ``(out_features,)``. Returns a tensor of shape ``(*batch, out_features)``.
        """

        if _eligible(input_, layer.weight, bias):
            return _tinygemm_linear(input_, layer.weight, bias)
        return original_apply(self, layer, input_, bias)

    # Save the original method above so unsupported operands still use it.
    UnquantizedLinearMethod.apply = apply
    _installed = True
    return {
        "active": "tinygemm",
        "reason": "FlashInfer tinygemm_bf16 is available",
        "device_capability": capability,
    }


__all__ = ["install_linear_backend"]
