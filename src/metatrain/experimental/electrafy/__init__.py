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


def _patch_metrics_global_keys_for_nccl() -> None:
    """Monkey-patch :func:`metatrain.utils.metrics._get_global_keys` to use a
    NCCL-compatible object gather.

    Upstream's implementation calls ``torch.distributed.all_gather_object``,
    which constructs CPU byte tensors. Newer PyTorch's strict-mode NCCL
    rejects CPU collectives bound to a CUDA-backed process group with
    ``Attempt to perform collective on tensor not on device passed to
    init_process_group`` -- which crashes the trainer at the end of every
    epoch when it tries to aggregate per-rank metric keys.

    We replace the helper with one that pickles the local list, pushes the
    bytes through a CUDA ``uint8`` tensor, calls plain ``all_gather`` (which
    NCCL handles natively), and unpickles on each rank. Falls back to the
    original ``all_gather_object`` path when the active backend is not
    NCCL (CPU + gloo continue to work).

    Idempotent.
    """
    import pickle

    import torch
    import torch.distributed as dist

    import metatrain.utils.metrics as _mtmetrics

    if getattr(_mtmetrics._get_global_keys, "_electrafy_nccl_patched", False):
        return

    def _get_global_keys_nccl(keys):
        import os

        local_keys = list(keys)
        world_size = dist.get_world_size()

        if dist.get_backend() != "nccl":
            gathered = [None] * world_size
            dist.all_gather_object(gathered, local_keys)
            union: set = set()
            for r in gathered:
                if r:
                    union.update(r)
            return sorted(union)

        # Strict-mode NCCL requires collectives on the exact device passed to
        # init_process_group. The trainer binds it to cuda:{local_rank} but
        # does not call torch.cuda.set_device, so torch.cuda.current_device()
        # still returns 0. Read LOCAL_RANK from env (set by metatrain's
        # DistributedEnvironment._setup_distr_env from SLURM_LOCALID) and
        # honour that.
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_idx = local_rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{device_idx}")
        payload = pickle.dumps(local_keys)
        local_size = torch.tensor([len(payload)], dtype=torch.int64, device=device)
        sizes = [
            torch.zeros(1, dtype=torch.int64, device=device)
            for _ in range(world_size)
        ]
        dist.all_gather(sizes, local_size)
        max_size = int(max(s.item() for s in sizes))

        local_buf = torch.zeros(max_size, dtype=torch.uint8, device=device)
        if len(payload) > 0:
            local_buf[: len(payload)] = torch.frombuffer(
                bytearray(payload), dtype=torch.uint8
            ).to(device)
        bufs = [
            torch.empty(max_size, dtype=torch.uint8, device=device)
            for _ in range(world_size)
        ]
        dist.all_gather(bufs, local_buf)

        union = set()
        for buf, sz in zip(bufs, sizes):
            n = int(sz.item())
            if n == 0:
                continue
            rank_keys = pickle.loads(bytes(buf[:n].cpu().numpy()))
            if rank_keys:
                union.update(rank_keys)
        return sorted(union)

    _get_global_keys_nccl._electrafy_nccl_patched = True  # type: ignore[attr-defined]
    _mtmetrics._get_global_keys = _get_global_keys_nccl


_patch_metrics_global_keys_for_nccl()


__model__ = ELECTRAFY
__trainer__ = Trainer

__authors__ = [
    ("Jonathan Schmidt", "@jonschmi"),
]

__maintainers__ = [
    ("Jonathan Schmidt", "@jonschmi"),
]
