from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import matplotlib.pyplot as plt
import numpy as np
 
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))
        
image = cv2.imread(r"/data/lzq/GPP/Datas/GPP/Data/NWPU/images/0005.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

sam_checkpoint = "/data/lzq/segment-anything-main/sam_vit_h_4b8939.pth"
model_type = "default"
device = "cpu"
 
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)


mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)
 
plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.savefig("test_1.jpg")