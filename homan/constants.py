import os.path as osp

EXTERNAL_DIRECTORY = "./external"

# Configurations for PointRend.
POINTREND_PATH = osp.join(EXTERNAL_DIRECTORY, "detectron2/projects/PointRend")
POINTREND_CONFIG = osp.join(
    POINTREND_PATH,
    "configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
POINTREND_MODEL_WEIGHTS = (
    "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/"
    "164955410/model_final_3c3198.pkl")

# Configurations for BodyMocap.
BODY_MOCAP_PATH = osp.join(EXTERNAL_DIRECTORY, "frankmocap")
BODY_MOCAP_REGRESSOR_CKPT = osp.join(
    # BODY_MOCAP_PATH,
    "extra_data/body_module/pretrained_weights",
    "2020_05_31-00_50_43-best-51.749683916568756.pt",
)
HAND_MOCAP_REGRESSOR_CKPT = osp.join(
    # HAND_MOCAP_PATH,
    "extra_data/hand_module/pretrained_weights",
    "pose_shape_best.pth",
)
BODY_MOCAP_SMPL_PATH = osp.join(
    # BODY_MOCAP_PATH,
    "extra_data/smpl")

# Configurations for PHOSA
FOCAL_LENGTH = 1.0
REND_SIZE = 256  # Size of target masks for silhouette loss.
BBOX_EXPANSION_FACTOR = 0.3  # Amount to pad the target masks for silhouette loss.
SMPL_FACES_PATH = "models/smpl_faces.npy"
MANO_OBJ_PATH = "extra_data/mano/closed_mano.obj"

# Dict[class_name: List[Tuple(path_to_parts_json, interaction_pairs_dict)]].
PART_LABELS = {
    "default": [(
        None,
        {
            "all": ["lhand", "rhand"]
        },
    )],
}
INTERACTION_MAPPING = {
    "default": ["lhand", "rhand"],
}
BBOX_EXPANSION = {
    "default": 0.3,
}
BBOX_EXPANSION_PARTS = {
    "default": 0.3,
}
INTERACTION_THRESHOLD = {
    "default": 5,
}
