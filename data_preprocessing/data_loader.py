import xml.etree.ElementTree as ET

# define some constants
IMG_SIZE = 224  # size of the image that will be input to the network, same as Imagenet

class Bbox():
    def __init__(self, id, obj_type, xc, yc, width, height):
        self.id = id
        self.object_type = obj_type
        self.xc = xc
        self.yc = yc
        self.width = width
        self.height = height


    def compute_IoU(self, box2):
        box1 = self
        # print box1.__dict__
        # print box2.__dict__
        if box1.xc > box2.xc:
            box1, box2 = box2, box1

        box1_xmax = min(box1.xc+box1.width/2.0, IMG_SIZE)
        box2_xmin = max(box2.xc-box2.width/2.0, 0)

        intersection_width = max(box1_xmax - box2_xmin, 0)
        intersection_width = min(intersection_width, box1.width, box2.width)
        print "intersection_width", intersection_width

        if box1.yc > box2.yc:
            box1, box2 = box2, box1

        box1_ymax = min(box1.yc + box1.height / 2.0, IMG_SIZE)
        box2_ymin = max(box2.yc - box2.height / 2.0, 0)

        intersection_height = max(box1_ymax - box2_ymin, 0)
        intersection_height = min(intersection_height, box1.height, box2.height)
        print "intersection_height", intersection_height

        intersection_area = intersection_width*intersection_height
        print intersection_area
        # union_area = box1.area + box2.area - intersection_area
        union_area = (box1.width*box1.height) + (box2.width*box2.height) - intersection_area
        print union_area
        # print intersection_area/float(union_area)
        return intersection_area/union_area


def read_labels_from_xml(xml_file):
    '''
    
    :param xml_file: str 
        path to the xml file
    :return: 
    '''

    root = ET.parse(xml_file)
    image_id = xml_file.split('/')[-1]
    objects = root.findall('object')

    # size of the current image
    img_width = int(root.find('size/width').text)
    img_height = int(root.find('size/height').text)
    labels = []
    for object in objects:
        type = object.find('name').text
        xmin = int(object.find('bndbox/xmin').text)*IMG_SIZE/img_width  # scale to image size
        ymin = int(object.find('bndbox/ymin').text)*IMG_SIZE/img_height
        xmax = int(object.find('bndbox/xmax').text)*IMG_SIZE/img_width
        ymax = int(object.find('bndbox/ymax').text)*IMG_SIZE/img_height
        label = Bbox(image_id, type, (xmin+xmax)/2.0, (ymin+ymax)/2.0, xmax-xmin, ymax-ymin)

        labels.append(label)

    return labels


if __name__ == "__main__":
    test_path = '../data/VOCdevkit/VOC2012/Annotations/2007_000027.xml'
    # tree = ET.parse(test_path)
    # root = tree.getroot()
    #
    # print root.tag
    # print root.attrib
    #
    # for child in root:
    #     print child.tag, child.attrib
    #
    # objects = root.findall('object')
    # print objects[0].find('name').text

    labels = read_labels_from_xml(test_path)
    print labels[0].__dict__









