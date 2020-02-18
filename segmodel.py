import segmentation_models as sm


# define model

def denseunet():
    model = sm.Unet('densenet121', encoder_weights='imagenet')
    model.compile(
        'Adam',
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )
    return model