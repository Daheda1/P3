from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
results = model("DatasetExDark/ExDark_images/Bicycle/2015_00001.png", classes=[1,8,39,5,2,15,56,41,16,3,0,60])  # return a list of Results objects
#! Table = dining table
#! Person = person

print(results[0].boxes.cls)
    
def get_class_name(class_id):
    return {
        1: "Bicycle",
        8: "Boat",
        39: "Bottle",
        5: "Bus",
        2: "Car",
        15: "Cat",
        56: "Chair",
        41: "Cup",
        16: "Dog",
        3: "Motorbike",
        0: "People",
        60: "Table"
    }.get(class_id, "Unknown Class")

# Example usage:
print(get_class_name(1)) 