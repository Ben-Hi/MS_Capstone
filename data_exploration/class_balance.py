import os, sys
sys.path.append("/home/hillenb/Desktop/MS_Capstone/Mask_RCNN_Detectron2")
import settings
from dataset_converters import get_branch_train_dicts, get_branch_test_dicts


def _get_class_balance(data):
    class_counts = [0, 0, 0, 0, 0]
    for img in data:
        for annotation in img['annotations']:
            class_counts[annotation['category_id']] += 1
            
    return class_counts


if __name__ == "__main__":
    print("This module shows information about the cherry branch dataset.")
    
    print(settings.DATA_DIR)
    
    # Get training data
    train_data = get_branch_train_dicts()
    train_class_counts = _get_class_balance(train_data)
    total_train_counts = sum(train_class_counts)
    
    # Get testing data
    test_data = get_branch_test_dicts()
    test_class_counts = _get_class_balance(test_data)
    total_test_counts = sum(test_class_counts)
    
    print("\nTraining Data Statistics...")
    print("Number of imgs: {}".format(len(train_data)))
    print("                     Class balance")
    print ("leader     nonbranch     other     sidebranch     spur")
    print(" {}         {}          {}         {}        {}".format(train_class_counts[0], train_class_counts[1], train_class_counts[2], train_class_counts[3], train_class_counts[4]))
    print(" {:.1%}      {:.1%}        {:.1%}        {:.1%}       {:.1%}".format(train_class_counts[0] / total_train_counts, train_class_counts[1] / total_train_counts, train_class_counts[2] / total_train_counts, train_class_counts[3] / total_train_counts, train_class_counts[4] / total_train_counts))
    
    print("\nTest Data Statistics...")
    print("Number of imgs: {}".format(len(test_data)))
    print("                     Class balance")
    print("leader     nonbranch     other     sidebranch     spur")
    print(" {}         {}          {}          {}         {}".format(test_class_counts[0], test_class_counts[1], test_class_counts[2], test_class_counts[3], test_class_counts[4]))
    print(" {:.1%}      {:.1%}        {:.1%}         {:.1%}       {:.1%}".format(test_class_counts[0] / total_test_counts, test_class_counts[1] / total_test_counts, test_class_counts[2] / total_test_counts, test_class_counts[3] / total_test_counts, test_class_counts[4] / total_test_counts))