_base_ = './faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=15)),
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')))
# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
classes = ("yield_sign", "pedestrians_crossing_sign", "no_entry_sign", "chevron_left_sign", "chevron_right_sign", "parking_sign",
            "stop_sign", "60mph_sign", "30mph_sign", "one_way_sign", "speed_bump_sign", "no_left_turn_sign",
            "no_heavy_goods_vehicles_sign", "no_right_turn_sign", "children_sign")
data = dict(
    samples_per_gpu=4,
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
load_from='/content/mmdetection/checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'
work_dir="/content/drive/MyDrive/Group Coursework (80%)/Models/Faster RCNN"
seed=0
gpu_ids=range(1)
runner=dict(max_epochs=15)
log_config=dict(
    hooks = [
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
        ]
    )