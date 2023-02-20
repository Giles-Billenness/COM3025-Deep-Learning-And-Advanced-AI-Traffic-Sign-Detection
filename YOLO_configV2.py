_base_ = './yolov3_d53_mstrain-608_273e_coco.py'
# dataset settings
model=dict(bbox_head=dict(num_classes=15))
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(320, 320), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320, 320),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
classes=("yield_sign", "pedestrians_crossing_sign", "no_entry_sign", "chevron_left_sign", "chevron_right_sign", "parking_sign",
            "stop_sign", "60mph_sign", "30mph_sign", "one_way_sign", "speed_bump_sign", "no_left_turn_sign", 
            "no_heavy_goods_vehicles_sign", "no_right_turn_sign", "children_sign")
data = dict(
    samples_per_gpu=24,
    train=dict(
        classes=classes,
        pipeline=train_pipeline,
        type='CocoDataset',
        data_root="/content/competitionKaggle/",
        ann_file="/content/competitionKaggle/RevisedAnnotationsTrainWithCrowd.json",
        img_prefix= "/content/competitionKaggle/" + "RevisedImages/"
        ),
    val=dict(
        classes=classes,
        pipeline=test_pipeline,
        type='CocoDataset',
        data_root="/content/competitionKaggle/",
        ann_file="/content/competitionKaggle/RevisedAnnotationsValWithCrowd.json",
        img_prefix= "/content/competitionKaggle/" + "RevisedImages/"
        ),
    test=dict(
        classes=classes,
        pipeline=test_pipeline,
        type='CocoDataset',
        data_root="/content/competitionKaggle/",
        ann_file="/content/competitionKaggle/RevisedAnnotationsValWithCrowd.json",
        img_prefix= "/content/competitionKaggle/" + "RevisedImages/"
        ),
    )
dataset_type='CocoDataset'
data_root="/content/competitionKaggle/"
load_from='/content/mmdetection/checkpoints/yolov3_d53_320_273e_coco-421362b6.pth'
work_dir="/content/drive/MyDrive/Group Coursework (80%)/Models/YOLO"
runner=dict(max_epochs=15)
seed=0
gpu_ids=range(1)
log_config=dict(
    hooks = [
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
        ]
    )