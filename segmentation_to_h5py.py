import h5py
import sys
polygon_sides = [8,16,24,32]

images_path = sys.argv[1]
masks_path = sys.argv[2]
all_images = glob('{}/*'.format(images_path))
image_pairs = {}
for image_path in all_images:
    mask_path = '{}/{}'.format(
        masks_path,image_path.split(os.sep)[-1])
    if os.path.isfile(mask_path):
        image_pairs[image_path] = mask_path

hf = h5py.File(sys.argv[3], 'w')

all_ls = {
    x:np.linspace(0, 2*np.pi, num=x,endpoint=False) for x in polygon_sides}

for image_path in list(image_pairs.keys()):
    print(image_path)
    image = read_image(image_path)
    mask = read_image(image_pairs[image_path])[:,:,0][:,:,np.newaxis]
    mask = np.where(mask > 0,1,0)
    edge_image = np.uint8(cv2.Laplacian(mask,cv2.CV_64F))
    num_labels, labels_im = cv2.connectedComponents(mask)
    bounding_boxes = []
    bounding_polygons = {x:[] for x in all_ls}
    centers = []

    g = hf.create_group(image_path.split(os.sep)[-1])
    edge_group = g.create_group('edges')

    for c in range(1,num_labels):
        edge_x,edge_y = np.where((labels_im == c) & (edge_image > 0))
        object_edges = np.stack([edge_y,edge_x])
        edge_group.create_dataset(str(c),
                                  shape=object_edges.shape,
                                  dtype=np.int32,
                                  data=object_edges)
        x,y = np.where(labels_im == c)
        if len(x) < 10:
            print('\tbigmad')
        bbox = [np.min(x),np.min(y),
                np.max(x),np.max(y)]
        bbox_center = [int((bbox[0] + bbox[2])/2),
                       int((bbox[1] + bbox[3])/2)]
        center_x,center_y = bbox_center
        bounding_boxes.append(bbox)

        for polygon_side in all_ls:
            ls = all_ls[polygon_side]
            bounding_polygon = edges_to_polygon(edge_x-center_x,
                                                edge_y-center_y,
                                                polygon_side)
            bounding_polygon[0] += center_y
            bounding_polygon[1] += center_x
            bounding_polygons[polygon_side].append(bounding_polygon)
        centers.append([center_y,center_x])

    bounding_boxes = np.array(bounding_boxes)
    bounding_polygons = {
        x:np.array(bounding_polygons[x]) for x in bounding_polygons}
    centers = np.array(centers)
    #weight_map = get_weights(mask)[:,:,np.newaxis]

    g.create_dataset('image', shape=image.shape, dtype=np.uint8, data=image)
    g.create_dataset('mask', shape=mask.shape, dtype=np.uint8, data=mask)
    g.create_dataset('bounding_boxes',shape=bounding_boxes.shape,
                     dtype=np.int16,data=bounding_boxes)
    g.create_dataset('weight_map',shape=weight_map.shape,
                     dtype=np.float32,data=weight_map)
    bp = g.create_group('bounding_polygons')
    for polygon_side in bounding_polygons:
        bp.create_dataset(
            str(polygon_side),
            shape=bounding_polygons[polygon_side].shape,
            dtype=np.int32,
            data=np.int16(bounding_polygons[polygon_side]))
    g.create_dataset('centers',shape=centers.shape,
                     dtype=np.int32,data=np.int16(centers))

hf.close()
