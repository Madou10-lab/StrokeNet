from models.badminton.strokenet_v2 import StrokeNet2D
from models.two_player_interaction import TwoPlayerInteractWrapper


def _split_interact_kwargs(kwargs):
    head_kwargs = {
        k[len("interact_") :]: kwargs.pop(k)
        for k in list(kwargs)
        if k.startswith("interact_")
    }
    return head_kwargs


def StrokeNet2D_M2_Interact_(**kwargs):
    head_kwargs = _split_interact_kwargs(kwargs)
    num_classes = int(kwargs.get("num_classes", 25))
    in_channels = int(kwargs.get("in_channels", 2))
    use_pid = bool(kwargs.pop("use_player_id_embedding", True))
    num_joints = int(kwargs.pop("num_joints", 17))
    inner = StrokeNet2D(num_joints=num_joints, **kwargs)
    return TwoPlayerInteractWrapper(
        inner,
        in_channels=in_channels,
        num_classes=num_classes,
        use_player_id_embedding=use_pid,
        head_kwargs=head_kwargs,
    )
