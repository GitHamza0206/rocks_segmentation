from __init__ import * 
import time 
from streamlit_image_coordinates import streamlit_image_coordinates 
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_image_select import image_select

st.title("✨ Drilling CPSD  🏜")
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
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

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
    axs[1, 0].hist(grain_data['area(mm)'], bins=10, edgecolor='black')
    axs[1, 0].set_xlabel('area')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].set_title('Distribution of area (mm)^2')

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



# Chemin du dossier contenant les photos
folder_path = "images"
path = os.path.join(os.getcwd(), folder_path)
# Liste des fichiers d'images dans le dossier
image_files = [path+'/'+file for file in os.listdir(path) if file.endswith((".jpg", ".jpeg", ".png"))]











params_ct = st.sidebar.container() 
params_ct.selectbox('Select Project : ', ['Project1','Project2', 'Project3'])
new_project = params_ct.button('Create new Project')
#reset_button = btn_ctnr.button('Reset')


if new_project:
    project_form = st.form(key='pjct_form')
    project_expander = project_form.expander('Project Information')
    pc1,pc2 = project_form.columns(2)
    project_name = pc1.text_input('Project :')
    _d1,_d2 = pc1.columns(2)
    depth = _d1.text_input('Select Depth ' )
    depth_unit = _d2.selectbox('Metric', ['m','ft'])

    client_name = pc2.text_input('Client :')
    well_name = pc2.text_input('Well name :')
    deviation = pc2.text_input('Deviation (deg): ')
    drilling= pc2.checkbox('drilling')
    circulate = pc2.checkbox('circulate')
    _h1,_h2=pc2.columns(2)
    hole_size = _h1.number_input('hole size')
    hz_metric = _h2.selectbox('Metric', ['mm','inch'])
    formation = pc2.text_input('Formation: ')
    bit_type= pc2.radio('Bit_type', ['pdc', 'roller cone'])
    save_project_button  = project_form.form_submit_button('Save')
    save_to_session_state('save_button', save_project_button)

    save_project_button = read_from_session_state('save_button')
    if save_project_button:
        print('yes')
        project_data = {
            "Project": project_name,
            "Depth": depth,
            "Client name" : client_name,
            "Well name": well_name,
            "Deviation" : deviation,
            "Bit Type ": bit_type
        }
        append_to_json("./storage/project1.json", project_data)
        st.write(project_data)

if 'previous_example_index' not in st.session_state.keys():
    st.session_state["previous_example_index"] = 0

index_selected = image_select(
    "",
    images=image_files,
    index=0,
    return_value="index",
    use_container_width=False
)



image_form = st.form(key="img_form", clear_on_submit=True)
ic1,ic2,ic3,ic4 = image_form.columns([3,0.5,1,0.5])

image_path = ic1.file_uploader(
    "Upload Image 🚀",
    type=["png","jpg","bmp","jpeg"])  

scale = ic3.slider(
    "Scale( px : mm)", 
    min_value=0, 
    max_value=100,
    step=1,
    key="scale"
    )



scale_expander = image_form.expander("Customize Scale parameters")
sc1,sc2,sc3 = scale_expander.columns(3)

mmscale = sc1.number_input('Image mm :')
pxscale = sc2.number_input('Image pixels : ')

sc3.number_input('Camera distance mm :')

filename_expander = image_form.expander("Customize Storage parameters")
fn1, fn2, fn3 = filename_expander.columns([2, 2,1])

renamed_file = fn1.text_input(
    'Renamed file',
    'Defaultname.jpg'
)


generate_seg_button = image_form.form_submit_button('Submit')



image_container = st.container()

data_container = st.container()
dc1,dc2 = data_container.columns(2)

graphs_container = st.container() 

oicol, sicol = image_container.columns(2)

image = read_image(image_files[index_selected])


if image_path is not None:
    image = read_image(image_path)



if image is not None:
    oicol.image(image)
    #image = resize_image(image, size= (600,600))

    tab1, tab2 = graphs_container.tabs(["Graphs ", "Data"])
    col11,col12 = tab1.columns(2)
    col21,col22 = tab2.columns(2)


    if generate_seg_button :
        scale_value = int(scale)/100
        if mmscale and pxscale >0:
            scale_value = mmscale/pxscale
        all_grains, labels, mask_all, grain_data, big_im = onSeg(image, seg_sam, unet)
        #fig.set_size_inches(5,5)
        #fig = plot_result(image,all_grains,labels)
       
        grain_data = update_grain_data(grain_data,scale_value)

        #--------PSD GRAPH-------#
        PSD(oicol, grain_data['area(mm)'].values)
        #--------shapes df -------#
        shapes = count_number_of_particles_by_size_range(grain_data['area(mm)'].values)
        shapes_df = pd.DataFrame.from_dict(shapes)
        dc1.dataframe(shapes_df)
        #-------PSD Table-------#
        PSD_table(grain_data['area(mm)'].values, dc2)
        #-------p50-------#
        p50 = np.median(grain_data['area(mm)'].values)
        data_container.text(f"P50 size :  {p50.round(1)}")
        data_container.text(f"Total count :  {len(grain_data)}")
        tab2.dataframe(grain_data)
        graphs(tab1)

        fig, ax = plt.subplots(figsize=(15,10))
        ax.imshow(big_im)
        segModule.plot_image_w_shape_color(image, all_grains, ax, grain_data, cmap='viridis')
        #segModule.plot_image_w_colorful_grains(big_im, all_grains, ax, cmap='Paired')
        segModule.plot_grain_axes_and_centroids(all_grains, labels, ax, linewidth=1, markersize=10)
        plt.xticks([])
        plt.yticks([])
        plt.xlim([0, np.shape(big_im)[1]])
        plt.ylim([np.shape(big_im)[0], 0])
        plt.tight_layout()
        
        sicol.pyplot(fig, use_container_width=True,pad_inches=0, bbox_inches="tight", transparent=True, dpi=300)



else :
    image_container.error('Upload photo')




