from models.badminton.strokenet_v1 import MambaSkeletonActionModelCOCO17
from models.two_player_interaction import TwoPlayerInteractWrapper


def _split_interact_kwargs(kwargs):
    return {
        k[len("interact_") :]: kwargs.pop(k)
        for k in list(kwargs)
        if k.startswith("interact_")
    }


def StrokeNetV1_M2_Interact_(**kwargs):
    head_kwargs = _split_interact_kwargs(kwargs)
    num_classes = int(kwargs.get("num_classes", 25))
    in_channels = int(kwargs.get("in_channels", 2))
    use_pid = bool(kwargs.pop("use_player_id_embedding", True))
    kwargs.pop("num_joints", None)
    inner = MambaSkeletonActionModelCOCO17(**kwargs)
    return TwoPlayerInteractWrapper(
        inner,
        in_channels=in_channels,
        num_classes=num_classes,
        use_player_id_embedding=use_pid,
        head_kwargs=head_kwargs,
    )
