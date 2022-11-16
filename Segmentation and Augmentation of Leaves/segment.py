# Importing necessary packages

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.close()


# Trying different color spaces to find best color channel.
def preprocess_input(path):
    image = cv2.imread(path)

    if image is None:
        return -1

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #     plt.imshow(img_rgb)
    #     plt.show()

    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    #     plt.imshow(img_lab)
    #     plt.show()

    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #     plt.imshow(img_hsv)
    #     plt.show()

    R = img_rgb[:, :, 0]
    G = img_rgb[:, :, 1]
    B = img_rgb[:, :, 2]

    L = img_lab[:, :, 0]
    a = img_lab[:, :, 1]
    b = img_lab[:, :, 2]

    H = img_hsv[:, :, 0]
    S = img_hsv[:, :, 1]
    V = img_hsv[:, :, 2]

    pixel_vals = b.flatten()
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Since we are interested in only actual leaf pixels, we choose 2 clusters
    # one cluster for actual leaf pixels and other for unwanted background pixels.
    K = 2
    centroids = np.array([[55], [175]]).astype(np.int32)
    centroid_labels = np.random.randint(K, size=pixel_vals.shape, dtype=np.int32)

    retval, labels, centers = cv2.kmeans(data=pixel_vals,
                                         K=K,
                                         bestLabels=centroid_labels,
                                         criteria=criteria,
                                         attempts=10,
                                         flags=cv2.KMEANS_USE_INITIAL_LABELS,
                                         centers=centroids)

    #     retval, labels, centers = cv2.kmeans(pixel_vals, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    centers = np.uint8(centers)

    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((b.shape))
    pixel_labels = labels.reshape(img_lab.shape[0], img_lab.shape[1])
    # displaying segmented image
    #     segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2RGB)
    # print("Pixel Labels", pixel_labels)
    count_nonzero = np.count_nonzero(pixel_labels)
    count_zero = pixel_labels.size - count_nonzero
    # print("Non-Zero", count_nonzero)
    # print("Zero", count_zero)
    plt.imshow(pixel_labels)
    plt.show()

    # Doing this, some unwanted pixels that are clustered in main cluster can be avoided.
    # Ref - https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#gac2718a64ade63475425558aa669a943a
    pixel_labels = np.uint8(pixel_labels)
    ret, components = cv2.connectedComponents(pixel_labels, connectivity=8)
    #     plt.imshow(components, cmap='gray')
    #     plt.show()

    indices = []
    for i in range(1, ret):
        row, col = np.where(components == i)
        indices.append(max(len(row), len(col)))
    component = np.argmax(np.array(indices))
    main_component = component + 1  # indexing starts from 0, so we increment by 1 to get actual component index
    # creating a mask and extracting pixels corresponding to cluster to which leaf belongs.
    # 1 for actual leaf pixels and 0 for other pixels

    # mask = np.where(components == main_component, 1, 0)
    if count_nonzero < count_zero:
        mask = np.where(components == main_component, 1, 0)
    else:
        mask = np.where(components == main_component, 0, 1)

    B = image[:, :, 0]
    G = image[:, :, 1]
    R = image[:, :, 2]
    # Extract only masked pixels
    r = R * mask
    g = G * mask
    b = B * mask
    final_img = np.dstack((r, g, b))

    return final_img
