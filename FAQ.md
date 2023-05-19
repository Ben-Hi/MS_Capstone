### What are the classes in the dataset? ###
The UFORoza dataset contains 5 classes:
    -leader
    -nonbranch
    -other
    -sidebranch
    -spur


### What format is the data stored in? ###
The data was exported from Labelme to COCO Segmentation using roboflow.ai.
The labels.json for each split stores data as follows:

{
    # Information about when the export occured and the label categories in the dataset
    "info":{
        "year": "2023",
        ...
        "images":
        [
            {
                "id": ...,
                "license": ...,
                "file_name": "...",
                "height": 480,
                "width": 640,
                "date_captured": "...",
            },
            ...
        ]

        "annotations":
        [
            {
                "id": ...,
                "image_id": ...,
                "category_id": ...,
                "bbox": [..., ..., ..., ...],
                "area": ...,
                "segmentation": [
                    []
                ]
                "iscrowd": ...
            },
            ...
        ]
    }
}