from __init__ import * 
import segmenteverygrain as segModule

model = seg.Unet()
model.compile(optimizer=Adam(), loss=seg.weighted_crossentropy, metrics=["accuracy"])
model.load_weights('./checkpoints/seg_model')

# the SAM model checkpoints can be downloaded from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
sam = sam_model_registry["default"](checkpoint="model/sam_vit_h_4b8939.pth")


big_im = np.array(load_img('1_2_image.png'))
big_im_pred = segModule.predict_big_image(big_im, model, I=256)
# decreasing the 'dbs_max_dist' parameter results in more SAM prompts (and longer processing times):
labels, grains, coords = segModule.label_grains(big_im, big_im_pred, dbs_max_dist=10.0)
all_grains, labels, mask_all, grain_data, fig, ax = segModule.sam_segmentation(sam, big_im, big_im_pred, coords, labels, min_area=50.0)

print(grain_data)