from torch.utils.tensorboard import SummaryWriter

def get_writer(comment: str | None = None):
    return SummaryWriter(comment=comment or '')

def log_scalar(writer, tag: str, value: float, step: int):
    if writer is not None:
        writer.add_scalar(tag, value, step)
