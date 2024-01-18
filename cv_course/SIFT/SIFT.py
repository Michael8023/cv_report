import cv2
import matplotlib.pyplot as plt

def create_sift_descriptor(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors
if __name__=='__main__':
    image_path = './image.jpg'
    keypoints, descriptors = create_sift_descriptor(image_path)
    image = cv2.imread(image_path)
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

    # 显示图像
    plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title('Image with SIFT Keypoints')
    plt.savefig('SIFT_image.jpg')

    # 打印关键点和描述符的数量
    print(f'Number of keypoints: {len(keypoints)}')
    print(f'Descriptor shape: {descriptors.shape}')
