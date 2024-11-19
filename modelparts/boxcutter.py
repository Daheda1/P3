# modelparts/boxcutter.py
def crop_bounding_boxes(image, boxes):
    def box_intersection_area(box1, box2):
        l1, t1, w1, h1 = box1[1:5]
        l2, t2, w2, h2 = box2[1:5]

        # Calculate the coordinates of the intersection rectangle
        x1 = max(l1, l2)
        y1 = max(t1, t2)
        x2 = min(l1 + w1, l2 + w2)
        y2 = min(t1 + h1, t2 + h2)

        # If there is no intersection
        if x1 >= x2 or y1 >= y2:
            return 0

        # Calculate the area of the intersection
        return (x2 - x1) * (y2 - y1)

    def box_area(box):
        _, l, t, w, h = box
        return w * h

    cropped_images = []
    objects = []

    for i, box1 in enumerate(boxes):
        if len(box1) != 5:
            raise ValueError(f"Expected 5 elements in box1, but got {len(box1)}: {box1}")
        object1, l, t, w, h = box1
        cropped_image = image[t:t+h, l:l+w]
        overlapping_objects = [object1]  # Start with the current box's object

        for j, box2 in enumerate(boxes):
            if i != j:  # Skip comparing the same box
                intersection = box_intersection_area(box1, box2)
                smaller_box_area = min(box_area(box1), box_area(box2))
                if intersection > 0.5 * smaller_box_area:
                    overlapping_objects.append(box2[0])  # Add the overlapping object's label

        cropped_images.append(cropped_image)
        objects.append(overlapping_objects)

    return cropped_images, objects