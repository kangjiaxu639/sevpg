import numpy as np
# constants
scenes = {}
scenes["train"] = [
    'Allensville',
    'Beechwood',
    'Benevolence',
    'Coffeen',
    'Cosmos',
    'Forkland',
    'Hanson',
    'Hiteman',
    'Klickitat',
    'Lakeville',
    'Leonardo',
    'Lindenwood',
    'Marstons',
    'Merom',
    'Mifflinburg',
    'Newfields',
    'Onaga',
    'Pinesdale',
    'Pomaria',
    'Ranchester',
    'Shelbyville',
    'Stockman',
    'Tolstoy',
    'Wainscott',
    'Woodbine',
]
# scenes["train"] = ["1LXtFkjw3qL"]
scenes["val"] = [
    'Collierville',
    'Corozal',
    'Darden',
    'Markleeville',
    'Wiconisco',
]

master_scene_dir = "/home/ros/kjx/alp/data/scene_datasets/gibson_semantic/"

coco_categories = {
    "chair": 0,
    "couch": 1,
    "potted plant": 2,
    "bed": 3,
    "toilet": 4,
    "tv": 5,
    "dining-table": 6,
    "oven": 7,
    "sink": 8,
    "refrigerator": 9,
    "book": 10,
    "clock": 11,
    "vase": 12,
    "cup": 13,
    "bottle": 14
}

coco_categories_mapping = {
    56: 0,  # chair
    57: 1,  # couch
    58: 2,  # potted plant
    59: 3,  # bed
    61: 4,  # toilet
    62: 5,  # tv
    60: 6,  # dining-table
    69: 7,  # oven
    71: 8,  # sink
    72: 9,  # refrigerator
    73: 10,  # book
    74: 11,  # clock
    75: 12,  # vase
    41: 13,  # cup
    39: 14,  # bottle
}

action_mapping = {
    "move_forward": 0,
    "turn_left": 1,
    "turn_right": 2,
}
action_decode = {
    0: "move_forward",
    1: "turn_left",
    2: "turn_right",
}


color_palette = [
    1.0, 1.0, 1.0,
    0.6, 0.6, 0.6,
    0.95, 0.95, 0.95,
    0.96, 0.36, 0.26,
    0.12156862745098039, 0.47058823529411764, 0.7058823529411765,
    0.9400000000000001, 0.7818, 0.66,
    0.9400000000000001, 0.8868, 0.66,
    0.8882000000000001, 0.9400000000000001, 0.66,
    0.7832000000000001, 0.9400000000000001, 0.66,
    0.6782000000000001, 0.9400000000000001, 0.66,
    0.66, 0.9400000000000001, 0.7468000000000001,
    0.66, 0.9400000000000001, 0.8518000000000001,
    0.66, 0.9232, 0.9400000000000001,
    0.66, 0.8182, 0.9400000000000001,
    0.66, 0.7132, 0.9400000000000001,
    0.7117999999999999, 0.66, 0.9400000000000001,
    0.8168, 0.66, 0.9400000000000001,
    0.9218, 0.66, 0.9400000000000001,
    0.9400000000000001, 0.66, 0.8531999999999998,
    0.9400000000000001, 0.66, 0.748199999999999]

color_palette_array = np.asarray([
    1.0, 1.0, 1.0,
    0.6, 0.6, 0.6,
    0.95, 0.95, 0.95,
    0.96, 0.36, 0.26,
    0.12156862745098039, 0.47058823529411764, 0.7058823529411765,
    0.9400000000000001, 0.7818, 0.66,
    0.9400000000000001, 0.8868, 0.66,
    0.8882000000000001, 0.9400000000000001, 0.66,
    0.7832000000000001, 0.9400000000000001, 0.66,
    0.6782000000000001, 0.9400000000000001, 0.66,
    0.66, 0.9400000000000001, 0.7468000000000001,
    0.66, 0.9400000000000001, 0.8518000000000001,
    0.66, 0.9232, 0.9400000000000001,
    0.66, 0.8182, 0.9400000000000001,
    0.66, 0.7132, 0.9400000000000001,
    0.7117999999999999, 0.66, 0.9400000000000001,
    0.8168, 0.66, 0.9400000000000001,
    0.9218, 0.66, 0.9400000000000001,
    0.9400000000000001, 0.66, 0.8531999999999998,
    0.9400000000000001, 0.66, 0.748199999999999])

mpcat40_labels = [
    # '', # -1
    #'void', # 0
    'wall',
    'floor',
    'chair',
    'door',
    'table', # 5
    'picture',
    'cabinet',
    'cushion',
    'window',
    'sofa', # 10
    'bed',
    'curtain',
    'chest_of_drawers',
    'plant',
    'sink',
    'stairs',
    'ceiling',
    'toilet',
    'stool',
    'towel', # 20
    'mirror',
    'tv_monitor',
    'shower',
    'column',
    'bathtub',
    'counter',
    'fireplace',
    'lighting',
    'beam',
    'railing',
    'shelving',
    'blinds',
    'gym_equipment', # 33
    'seating',
    'board_panel',
    'furniture',
    'appliances',
    'clothes',
    'objects',
    'misc',
    'unlabeled' # 41
]




mp3d_habitat_labels = {
            'chair': 0, #g
            'table': 1, #g
            'picture':2, #b
            'cabinet':3, # in resnet
            'cushion':4, # in resnet
            'sofa':5, #g
            'bed':6, #g
            'chest_of_drawers':7, #b in resnet
            'plant':8, #g
            'sink':9, #g
            'toilet':10, #g
            'stool':11, #b
            'towel':12, #b in resnet
            'tv_monitor':13, #g
            'shower':14, #b
            'bathtub':15, #b in resnet
            'counter':16, #b isn't this table?
            'fireplace':17,
            'gym_equipment':18,
            'seating':19,
            'clothes':20, # in resnet
            'background': 21
}

hm3d_habitat_labels = {
            # 'background': 0,
            'chair': 0, #g
            'bed': 1, #g
            'plant':2, #b
            'toilet':3, # in resnet
            'tv_monitor':4, # in resnet
            'sofa':5,
            'background':6, #background
}



def get_habitat_labels(data_name):
    if data_name =="hm3d":
        return hm3d_habitat_labels
    elif data_name =="mp3d":
        return mp3d_habitat_labels

def get_fourty_dict(data_name):
    fourty2_dict = {}

    for i in range(len(mpcat40_labels)):
        lb = mpcat40_labels[i]
        if data_name =="hm3d":
            if lb in hm3d_habitat_labels.keys():
                fourty2_dict[i] = hm3d_habitat_labels[lb]
        elif data_name =="mp3d":
            if lb in mp3d_habitat_labels.keys():
                fourty2_dict[i] = mp3d_habitat_labels[lb]

    fourty2 = copy.deepcopy(fourty2_dict)
    return fourty2