from __init__ import * 
import time 
from streamlit_image_coordinates import streamlit_image_coordinates 
from streamlit_extras.dataframe_explorer import dataframe_explorer

st.set_page_config(
    page_title="Linsy Rock Vision",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

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
    with tab2:
        col11, col12,col13 = st.columns(3)
        with col11 :
            PSD(size_values)
        with col12:
            ASD(area)
        with col13: 
            if cm is not None:  
                PCD(cm)


sam = mask_generate()

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
    
    generate_masks_button = image_form.form_submit_button('Generate masks with SAM')

    tab1, tab2, tab3 = image_form.tabs(["Single Rock viewer", "Graphs ", "Details"])
    col11,col12 = tab1.columns(2)


    if generate_masks_button :
        masks = sam.generate(image)
        size_values=calculate_size_from_multiple_masks(masks)
        
        area = [r['area'] // RATIO_PER_PIXEL  for r in masks]
        cm=generate_color_mask(size_values)
        transparency=0.80
        fig = show_image_plt(image,masks,cm,transparency,title='Rock segmentation by rock size')
        sicol.pyplot(fig)
        with tab2:
            graphs(tab2)
        
        #with col11:
        #    coords = get_vb_from_state('coords')
        #    st.write(coords)
        #    if coords:
        #        i,draw_image= draw_rectangle(image, masks, coords['x'],coords['y'])
        #        coords= streamlit_image_coordinates(
        #                draw_image,
        #                key="numpy",
        #            )
        #        set_vb_in_state('coords', coords)
        #        with col12:
        #            shape= calculate_shape_from_size(size_values[i])
        #            df = single_detail_rock(size_values[i], shape)
        #            col12.dataframe(df)
        #    else:
        #        coords= streamlit_image_coordinates(
        #            image,
        #            key="numpy",
        #        )
        #        set_vb_in_state('coords',coords)
        #        print(coords)

    


else :
    image_container.error('Upload photo')




