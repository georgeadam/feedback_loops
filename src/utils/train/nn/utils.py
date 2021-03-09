def compute_loss(model, x, y, criterion):
    out = model(x)

    return criterion(out, y)


def log_regular_losses(writer, writer_prefix, train_loss, val_loss, epoch):
    writer.add_scalar(writer_prefix.format(name="train_loss"), train_loss, epoch)
    writer.add_scalar(writer_prefix.format(name="val_loss"), val_loss, epoch)


def log_lre_losses(writer, writer_prefix, train_loss, val_loss_reg, val_loss_lre, epoch):
    writer.add_scalar(writer_prefix.format(name="train_loss"), train_loss, epoch)
    writer.add_scalar(writer_prefix.format(name="val_loss_reg"), val_loss_reg, epoch)
    writer.add_scalar(writer_prefix.format(name="val_loss_lre"), val_loss_lre, epoch)