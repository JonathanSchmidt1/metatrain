from .model import ELECTRAFY
from .trainer import Trainer


def _register_electrafy_losses() -> None:
    """Wire :class:`~.modules.loss.TensorMapNMAELoss` into the metatrain loss
    factory under the type key ``"nmae"``.

    Upstream's :class:`metatrain.utils.loss.LossType` is a closed Enum so we
    cannot extend it after import. Instead we wrap
    :func:`metatrain.utils.loss.create_loss` to intercept the ``"nmae"`` key
    before it reaches ``LossType.from_key``. Idempotent: re-imports of this
    module won't stack wrappers.
    """
    import metatrain.utils.loss as _mtloss

    from .modules.loss import TensorMapNMAELoss

    registry = {"nmae": TensorMapNMAELoss}

    if getattr(_mtloss.create_loss, "_electrafy_wrapped", False):
        return

    _orig = _mtloss.create_loss

    def create_loss(loss_type, **kwargs):  # type: ignore[no-untyped-def]
        if loss_type in registry:
            return registry[loss_type](**kwargs)
        return _orig(loss_type, **kwargs)

    create_loss._electrafy_wrapped = True  # type: ignore[attr-defined]
    _mtloss.create_loss = create_loss


_register_electrafy_losses()


__model__ = ELECTRAFY
__trainer__ = Trainer

__authors__ = [
    ("Jonathan Schmidt", "@jonschmi"),
]

__maintainers__ = [
    ("Jonathan Schmidt", "@jonschmi"),
]
