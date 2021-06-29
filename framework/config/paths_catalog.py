"""Centralized catalog of paths."""
import os
from os.path import exists

class DatasetCatalog(object):
    DATA_DIR = "./data"
    if not exists(DATA_DIR):
        print("Please specify the folder of your datasets!")
        exit()

    DATASETS = {
        "activitynet_train":{
            "video_dir": "ActivityNet",
            "data_dir" : "./data",
            "ann_file": "ActivityNet/train.json",
            "feat_file": "ActivityNet/activitynet_train_feat.pickle",
        },
        "activitynet_val":{
            "video_dir": "ActivityNet",
            "data_dir" : "./data",
            "ann_file": "ActivityNet/val.json",
            "feat_file": "ActivityNet/activitynet_val_feat.pickle",
        },
        "activitynet_test":{
            "video_dir": "ActivityNet",
            "data_dir" : "./data",
            "ann_file": "ActivityNet/test.json",
            "feat_file": "ActivityNet/activitynet_test_feat.pickle",
        },
        "activitynet_ood_test_30":{
            "video_dir": "ActivityNet",
            "data_dir" : "./data",
            "ann_file": "ActivityNet/test.json",
            "feat_file": "ActivityNet/activitynet_test_feat.pickle",
        },        
        "activitynet_ood_test_60":{
            "video_dir": "ActivityNet",
            "data_dir" : "./data",
            "ann_file": "ActivityNet/test.json",
            "feat_file": "ActivityNet/activitynet_test_feat.pickle",
        },           
        "charades_I3D_train":{
            "video_dir": "Charades-STA",
            "data_dir" : "./data",
            "ann_file": "Charades-STA/charades_sta_train.txt",
            "feat_file": "Charades-STA/charades_i3d/i3d_finetuned",
        },
        "charades_I3D_test":{
            "video_dir": "Charades-STA",
            "data_dir" : "./data",
            "ann_file": "Charades-STA/charades_sta_test.txt",
            "feat_file": "Charades-STA/charades_i3d/i3d_finetuned",
        },  
        "charades_I3D_ood_test_10":{
            "video_dir": "Charades-STA",
            "data_dir" : "./data",
            "ann_file": "Charades-STA/charades_sta_test.txt",
            "feat_file": "Charades-STA/charades_i3d/i3d_finetuned",
        },  
        "charades_I3D_ood_test_15":{
            "video_dir": "Charades-STA",
            "data_dir" : "./data",
            "ann_file": "Charades-STA/charades_sta_test.txt",
            "feat_file": "Charades-STA/charades_i3d/i3d_finetuned",
        },                         
        "charades_VGG_train":{
            "video_dir": "Charades-STA",
            "data_dir" : "./data",
            "ann_file": "Charades-STA/charades_sta_train.txt",
            "feat_file": "Charades-STA/charades_vgg/VGG",
        },
        "charades_VGG_test":{
            "video_dir": "Charades-STA",
            "data_dir" : "./data",
            "ann_file": "Charades-STA/charades_sta_test.txt",
            "feat_file": "Charades-STA/charades_vgg/VGG",
        }, 
        "charades_VGG_ood_test_10":{
            "video_dir": "Charades-STA",
            "data_dir" : "./data",
            "ann_file": "Charades-STA/charades_sta_test.txt",
            "feat_file": "Charades-STA/charades_vgg/VGG",
        },  
        "charades_VGG_ood_test_15":{
            "video_dir": "Charades-STA",
            "data_dir" : "./data",
            "ann_file": "Charades-STA/charades_sta_test.txt",
            "feat_file": "Charades-STA/charades_vgg/VGG",
        },     
        "didemo_train":{
            "video_dir": "didemo",
            "data_dir" : "./data",
            "ann_file": "didemo/train_data.json",
            "feat_file": "didemo/average_global_flow.h5",
        },
        "didemo_val":{
            "video_dir": "didemo",
            "data_dir" : "./data",
            "ann_file": "didemo/val_data.json",
            "feat_file": "didemo/average_global_flow.h5",
        },
        "didemo_test":{
            "video_dir": "didemo",
            "data_dir" : "./data",
            "ann_file": "didemo/test_data.json",
            "feat_file": "didemo/average_global_flow.h5",
        },
        "didemo_ood_test_3":{
            "video_dir": "didemo",
            "data_dir" : "./data",
            "ann_file": "didemo/test_data.json",
            "feat_file": "didemo/average_global_flow.h5",
        },             
        "didemo_ood_test_6":{
            "video_dir": "didemo",
            "data_dir" : "./data",
            "ann_file": "didemo/test_data.json",
            "feat_file": "didemo/average_global_flow.h5",
        },    
        "charades_C3D_train":{
            "video_dir": "Charades-STA",
            "data_dir" : "./data",
            "ann_file": "Charades-STA/charades_sta_train.txt",
            "feat_file": "Charades-STA/charades_c3d/charades_features_raw",
        },
        "charades_C3D_test":{
            "video_dir": "Charades-STA",
            "data_dir" : "./data",
            "ann_file": "Charades-STA/charades_sta_test.txt",
            "feat_file": "Charades-STA/charades_c3d/charades_features_raw",
        }, 
        "charades_C3D_ood_test_10":{
            "video_dir": "Charades-STA",
            "data_dir" : "./data",
            "ann_file": "Charades-STA/charades_sta_test.txt",
            "feat_file": "Charades-STA/charades_c3d/charades_features_raw",
        },  
        "charades_C3D_ood_test_15":{
            "video_dir": "Charades-STA",
            "data_dir" : "./data",
            "ann_file": "Charades-STA/charades_sta_test.txt",
            "feat_file": "Charades-STA/charades_c3d/charades_features_raw",
        },              

    }

    @staticmethod
    def get(name):
        attrs = DatasetCatalog.DATASETS[name]
        
        args = dict(
            dataset_name = name,
            root=os.path.join(attrs["data_dir"], attrs["video_dir"]),
            ann_file=os.path.join(attrs["data_dir"], attrs["ann_file"]),
            feat_file=os.path.join(attrs["data_dir"], attrs["feat_file"]),
        )
        if "didemo" in name:
            return dict(
                factory = "DiDeMoDataset",
                args = args
            )            
        elif "activitynet" in name:
            return dict(
                factory = "ActivityNetDataset",
                args = args
            )
        elif "charades" in name:
            return dict(
                factory = "CharadesDataset",
                args = args
            )            
        raise RuntimeError("Dataset not available: {}".format(name))
