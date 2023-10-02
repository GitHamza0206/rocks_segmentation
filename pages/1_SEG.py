from __init__ import * 
import time 
from streamlit_image_coordinates import streamlit_image_coordinates 
from streamlit_extras.dataframe_explorer import dataframe_explorer

st.title("✨ Linsy vision  🏜")
st.info(' Let me help generate segments for any of your images. 😉')

@st.cache_resource()
def mask_generate():
    sam_checkpoint = "model/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    return mask_generator 




def get_vb_from_state(vb_name):
    if vb_name in st.session_state.keys():
        return st.session_state[vb_name]
    return None
def set_vb_in_state(vb_name,vb_value):
    if vb_name not in st.session_state.keys():
        st.session_state[vb_name]=vb_value


def graphs(tab2):
    # Create subplots with 3 rows and 3 columns
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))

    # Plot the distribution of 'Roundness Index' in the first subplot
    axs[0, 0].hist(grain_data['Roundness Index'], bins=10, edgecolor='black')
    axs[0, 0].set_xlabel('Roundness Index')
    axs[0, 0].set_ylabel('Frequency')
    axs[0, 0].set_title('Distribution of Roundness Index')

    # Plot the distribution of 'Elongation Index' in the second subplot
    axs[0, 1].hist(grain_data['Elongation Index'], bins=10, edgecolor='black')
    axs[0, 1].set_xlabel('Elongation Index')
    axs[0, 1].set_ylabel('Frequency')
    axs[0, 1].set_title('Distribution of Elongation Index')

    # Repeat the process for the remaining subplots
    axs[0, 2].hist(grain_data['area'], bins=10, edgecolor='black')
    axs[0, 2].set_xlabel('area')
    axs[0, 2].set_ylabel('Frequency')
    axs[0, 2].set_title('Distribution of area')

    axs[1, 0].hist(grain_data['perimeter'], bins=10, edgecolor='black')
    axs[1, 0].set_xlabel('Perimeter')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].set_title('Distribution of perimeter')

    axs[1, 1].hist(grain_data['orientation'], bins=10, edgecolor='black')
    axs[1, 1].set_xlabel('Orientation')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].set_title('Distribution of orientation')

    # Adjust the layout and spacing
    plt.tight_layout()

    # Display the plots
    tab2.pyplot(fig)


seg_sam=seg_mask_generate()
unet=unetModel()

params_form = st.sidebar.form(key='options')
params_form.header('Params')
params_form.multiselect('Select image from database', ['1','2','3'])
params_form.form_submit_button('Execute')

image_container = st.container()
image_path = image_container.file_uploader("Upload Image 🚀", type=["png","jpg","bmp","jpeg"])  
oicol, sicol = image_container.columns(2)

if image_path is not None:
    image_form = image_container.form(key='segmentation')
    image = read_image(image_path)
    oicol.image(image)
    image = resize_image(image)
    params_form.text_input('Rename your file', placeholder='Renamed file')
    save_button  = params_form.form_submit_button('Save')
    
    generate_seg_button = image_form.form_submit_button('Generate masks with SEG')

    tab1, tab2 = image_form.tabs(["Graphs ", "Details"])
    col11,col12 = tab1.columns(2)
    
    if generate_seg_button :
        all_grains, labels, mask_all, grain_data, fig, ax = onSeg(image, seg_sam, unet)
        grain_data = update_grain_data(grain_data)
        sicol.pyplot(fig)
        tab1.dataframe(grain_data)
        graphs(tab2)


else :
    image_container.error('Upload photo')




