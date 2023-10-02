
from __init__ import segModule
from __init__ import *

@st.cache_resource()
def seg_mask_generate():
    sam_checkpoint = "model/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry["default"](checkpoint=sam_checkpoint)
    sam.to(device)

    return sam

@st.cache_resource()
def unetModel():
    model = segModule.Unet()
    model.compile(optimizer=Adam(), loss=segModule.weighted_crossentropy, metrics=["accuracy"])
    model.load_weights('./checkpoints/seg_model')
    return model

from math import pi,sqrt
def calc_ri(a):
    r = sqrt(a/pi)
    ri=2*pi*r
    return ri

def get_total_count(grain_data):
    return len(grain_data)

def update_cmap_with_shape(fig,shapes):
    pass


def update_grain_data(grain_data, scale):

    def calculate_shape_from_size(size):
        ranges_in_mm = [0.002, 0.063, 2.0, 63, 200]
        sorted_particles = {}
        if size<ranges_in_mm[0]:
            return 'clay'
        elif (size < ranges_in_mm[1]) & (size > ranges_in_mm[0]):
            return 'silt'
        elif (size < ranges_in_mm[2]) & (size > ranges_in_mm[1]):
            return 'sand'
        elif (size < ranges_in_mm[3]) & (size > ranges_in_mm[2]):
            return 'gravel'
        elif (size < ranges_in_mm[4]) & (size > ranges_in_mm[3]):
            return 'cobbles'
        elif (size > ranges_in_mm[4]):
            return 'boulders'

    grain_data['major_axis_length(mm)'] = grain_data['major_axis_length'].values*scale
    grain_data['minor_axis_length(mm)'] = grain_data['minor_axis_length'].values*scale
    grain_data['perimeter(mm)'] = grain_data['perimeter'].values*scale
    grain_data['area(mm)'] = grain_data['area'].values*scale**2
    grain_data['Roundness Index'] = grain_data['area'].apply(calc_ri)
    grain_data['Elongation Index'] = grain_data.apply(lambda row: (row['major_axis_length'] - row['minor_axis_length']) / row['major_axis_length'], axis=1)
    grain_data['shape'] = grain_data.apply(lambda row: calculate_shape_from_size(row['area(mm)']),axis=1)
    return grain_data

def plot_result(big_im,all_grains,labels):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(big_im)
    segModule.plot_image_w_colorful_grains(big_im, all_grains, ax, cmap='Paired')
    segModule.plot_grain_axes_and_centroids(all_grains, labels, ax, linewidth=1, markersize=10)
    return fig

@st.cache_data()
def onSeg(big_im, _sam, _model):
    big_im_pred = segModule.predict_big_image(big_im, _model, I=256)
    # decreasing the 'dbs_max_dist' parameter results in more SAM prompts (and longer processing times):
    labels, grains, coords = segModule.label_grains(big_im, big_im_pred, dbs_max_dist=10.0)
    all_grains, labels, mask_all, grain_data, big_im = segModule.sam_segmentation(_sam, big_im, big_im_pred, coords, labels, min_area=50.0 )
    return all_grains, labels, mask_all, grain_data, big_im

def mo(big_im,all_grains,labels):
    # plot results again if necessary
    fig, ax = plt.subplots(figsize=(15,10))
    ax.imshow(big_im)
    segModule.plot_image_w_colorful_grains(big_im, all_grains, ax, cmap='Paired')
    segModule.plot_grain_axes_and_centroids(all_grains, labels, ax, linewidth=1, markersize=10)