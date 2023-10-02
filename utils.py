
from __init__ import * 

def load_data(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return []

# Function to append new data to JSON file
def append_to_json(filename, data):
    existing_data = load_data(filename)
    existing_data.append(data)
    with open(filename, 'w') as f:
        json.dump(existing_data, f)
    

def sv_annotation(image, masks):
    mask_annotator = sv.MaskAnnotator()
    detections = sv.Detections.from_sam(masks)
    return mask_annotator.annotate(image, detections)

#@st.cache_data
def show_anns(anns, ax):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['²mentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

def save_to_session_state(key, value):
    """
    Save a value to st.session_state under the given key.
    
    Parameters:
        key (str): The key to use for storing the value.
        value: The value to save.
    """
    st.session_state[key] = value

def read_from_session_state(key, default=None):
    """
    Read a value from st.session_state using the given key.
    
    Parameters:
        key (str): The key to use for retrieving the value.
        default: The default value to return if the key is not found.
        
    Returns:
        The stored value or the default value if the key is not found.
    """
    return getattr(st.session_state, key, default)

def read_image(image):
    pil_image = Image.open(image) 
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return cv_image  

def get_image_dimensions(image):
    with Image.open(image) as img:
        width, height = img.size
    return width, height

def calculate_ratio_per_pixel():
  return 153.6

def annotate_image(image_c, annot, cords):
    x, y = cords
    annotate = f'size(mm){annot}'
    cv2.putText(image_c, annotate, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def draw_contour(image_c, mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_c, contours, -1, (255, 0, 0), 2)

def crop_image(image_c, crop_box):
    x, y, w, h = crop_box[0], crop_box[1], crop_box[2], crop_box[3]
    return image_c[y:y + h, x:x + w]

def verify_size(size):
    return True

def calculate_size(area):
    RATIO_PER_PIXEL = calculate_ratio_per_pixel()
    size = area /  RATIO_PER_PIXEL**2 * 100
    size = round(size, 2)
    return size

def get_masks(image, mask_generator):
    return mask_generator.generate(image)

def show_image(image_c, masks=None, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.imshow(image_c)
    if masks is not None:
        show_anns(masks)
    plt.axis('off')
    plt.show()

def treat_single_mask(image, single_mask):
    image_c = image.copy()
    # Obtemos a área e o retângulo delimitador da máscara
    area, bbox = single_mask['area'], single_mask['bbox']
    # Calculamos o tamanho da partícula
    size = calculate_size(area)
    if verify_size(size):
        # Anotamos a imagem escrevendo o tamanho
        x, y, w, h = bbox
        annotate_image(image_c, f'{size}mm', (x, y))
        # Desenhamos o contorno em vez do retângulo
        mask = single_mask['segmentation'].astype(np.uint8)
        draw_contour(image_c, mask)
    return image_c

def treat_multiple_masks(image, masks):
    image_c = image.copy()
    for mask in masks:
        image_c = treat_single_mask(image_c, mask)
    return image_c

def calculate_size_from_single_mask(single_mask):
    area = single_mask['area']
    size = calculate_size(area)
    return size

def calculate_size_from_multiple_masks(masks):
    sizes = []
    for mask in masks:
        size = calculate_size_from_single_mask(mask)
        if verify_size(size):
            sizes.append(size)
    return np.array(sizes)

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

def show_anns_plt(anns,cm,transparancy):
    if len(anns) == 0:
        return
    sorted_anns = anns
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    c = iter(cm)
    for ann in sorted_anns:
        m = ann['segmentation']
        cmi=next(c)
        color_mask = np.concatenate([cmi, [transparancy]])
        img[m] = color_mask
    ax.imshow(img)
    plt.colorbar()

def draw_rectangle(image,masks,x,y):
    bbox_array = [r['bbox'] for r in masks]
    for i,bbox in enumerate(bbox_array):
        bx, by, bw, bh = bbox
        if x >= bx and x <= bx + bw and y >= by and y <= by + bh:
            #mask = np.zeros_like(image)
            #mask[by:by+bh, bx:bx+bw, :] = image[by:by+bh, bx:bx+bw, :]
            cv2.rectangle(image, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
            #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            return (i,image)
  
def show_image_plt(image_c, masks=None,cm=None,transparency=0.8,title=None):
    figure = plt.figure()
    plt.imshow(image_c)
    if cm is None:
        cm = np.random.random(3)
    if transparency is None :
        transparency=0.8
    if masks is not None:
        show_anns_plt(masks,cm,transparency)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    return figure
    
def generate_color_mask(size_values):
    # Normalize the size values to the range [0, 1]
    normalized_sizes = (size_values - np.min(size_values)) / (np.max(size_values) - np.min(size_values))

    # Define the colormap from blue to red
    cmap = cm.get_cmap('viridis')

    # Assign colors based on the normalized size values
    color_mask = (cmap(normalized_sizes)[:, :3] ).astype(np.float32)  # Multiply by 255 and convert to uint8
    return color_mask


def resize_image(image, size=(480, 640)):
    image = cv2.resize(image, size)
    return image

def clean_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_image(image)
    return image

def ASD(areas):
    asd = plt.figure(figsize=(8,8))
    plt.hist(areas, bins=30, edgecolor='black')
    plt.xlabel('Particle area (mm)2')
    plt.ylabel('Frequency')
    plt.title('Particle Area Distribution')
    plt.grid(True)
    st.pyplot(asd)
    st.success('ASD Graph')

def PCD(colors):
    pcd = plt.figure(figsize=(8,8))
    plt.hist(colors, bins=30, edgecolor='black')
    plt.xlabel('Particle color intensity')
    plt.ylabel('Frequency')
    plt.title('Particle color Distribution')
    plt.grid(True)
    st.pyplot(pcd)
    st.success('PCD Graph')

def PSD(cont , sizes):
    psd = plt.figure()
    plt.hist(sizes, bins=50, density=False,  edgecolor='black')
    plt.xlabel('Particle Size(mm)')
    plt.ylabel('Frequency')
    plt.title('Particle Shape Distribution')
    plt.grid(True)
    cont.pyplot(psd)


def image_details_df(masks,original_image):
  rs =[]
  for mask in masks:
    size=calculate_size_from_single_mask(mask)
    r = {
            "image": st.image(crop_image(original_image, mask["crop_box"])),
            "size" : size,
            "shape": calculate_shape_from_size(size),
            "score": mask['stability_score']
            }
    rs.append(r)
    break
  df = pd.DataFrame(rs, columns=["image", "size", "shape","score"])
  return df 

def shape_summary(masks):
    shapes = count_number_of_particles_by_size_range(masks)
    shapes_df_temp = {}
    for k,v in shapes.items() :
        shapes_df_temp[k] = [v]
    df_shapes = pd.DataFrame(shapes_df_temp)
    return df_shapes


def count_number_of_particles(masks):
    return len(calculate_size_from_multiple_masks(masks))

def group_by_particles_size_range(masks):
    sizes_of_particles = calculate_size_from_multiple_masks(masks)
    ranges_in_mm = [0.002, 0.063, 2.0, 63, 200]
    sorted_particles = {}
    sorted_particles['clay'] = sizes_of_particles[(sizes_of_particles < ranges_in_mm[0])]
    sorted_particles['silt'] = sizes_of_particles[(sizes_of_particles < ranges_in_mm[1]) & (sizes_of_particles > ranges_in_mm[0])]
    sorted_particles['sand'] = sizes_of_particles[(sizes_of_particles < ranges_in_mm[2]) & (sizes_of_particles > ranges_in_mm[1])]
    sorted_particles['gravel'] = sizes_of_particles[(sizes_of_particles < ranges_in_mm[3]) & (sizes_of_particles > ranges_in_mm[2])]
    sorted_particles['cobbles'] = sizes_of_particles[(sizes_of_particles < ranges_in_mm[4]) & (sizes_of_particles > ranges_in_mm[3])]
    sorted_particles['boulders'] = sizes_of_particles[(sizes_of_particles > ranges_in_mm[4])]

    return sorted_particles

def count_number_of_particles_by_size_range(sizes_of_particles):
    #sizes_of_particles = calculate_size_from_multiple_masks(masks)
    ranges_in_mm = [0.002, 0.063, 2.0, 63, 200]
    sorted_particles = {}
    sorted_particles['clay'] = sizes_of_particles[(sizes_of_particles < ranges_in_mm[0])]
    sorted_particles['silt'] = sizes_of_particles[(sizes_of_particles < ranges_in_mm[1]) & (sizes_of_particles > ranges_in_mm[0])]
    sorted_particles['sand'] = sizes_of_particles[(sizes_of_particles < ranges_in_mm[2]) & (sizes_of_particles > ranges_in_mm[1])]
    sorted_particles['gravel'] = sizes_of_particles[(sizes_of_particles < ranges_in_mm[3]) & (sizes_of_particles > ranges_in_mm[2])]
    sorted_particles['cobbles'] = sizes_of_particles[(sizes_of_particles < ranges_in_mm[4]) & (sizes_of_particles > ranges_in_mm[3])]
    sorted_particles['boulders'] = sizes_of_particles[(sizes_of_particles > ranges_in_mm[4])]

    len_of_sorted_particles = {}
    len_of_sorted_particles = sorted_particles.copy()
    for k,v in len_of_sorted_particles.items():
        len_of_sorted_particles[k] = [len(v)]

    return len_of_sorted_particles



def index_of_images_of_particles_by_size_range(sizes_of_particles):
    ranges_in_mm = [0.002, 0.063, 2.0, 63, 200]
    sorted_particles = {}
    sorted_particles['clay'] = np.where(np.isin(sizes_of_particles , sizes_of_particles[(sizes_of_particles < ranges_in_mm[0])]))[0]
    sorted_particles['silt'] = np.where(np.isin(sizes_of_particles , sizes_of_particles[(sizes_of_particles < ranges_in_mm[1]) & (sizes_of_particles > ranges_in_mm[0])]))[0]
    sorted_particles['sand'] = np.where(np.isin(sizes_of_particles , sizes_of_particles[(sizes_of_particles < ranges_in_mm[2]) & (sizes_of_particles > ranges_in_mm[1])]))[0]
    sorted_particles['gravel'] = np.where(np.isin(sizes_of_particles , sizes_of_particles[(sizes_of_particles < ranges_in_mm[3]) & (sizes_of_particles > ranges_in_mm[2])]))[0]
    sorted_particles['cobbles'] = np.where(np.isin(sizes_of_particles , sizes_of_particles[(sizes_of_particles < ranges_in_mm[4]) & (sizes_of_particles > ranges_in_mm[3])]))[0]
    sorted_particles['boulders'] = np.where(np.isin(sizes_of_particles , sizes_of_particles[(sizes_of_particles > ranges_in_mm[4])]))[0]
    return sorted_particles

def images_of_particles_by_size_range(image, masks, indexes_of_particles):
    roi = np.take(masks, list(indexes_of_particles))
    out = treat_multiple_masks(image, roi)
    return out

def PSD_table(sizes, st):
    table = pd.DataFrame()
    table['sizes'] = sizes
    summary = table.describe()
    summary = summary.round(1)
    st.dataframe(summary.T)

"""# Analyse image

using this cell, you can run the analysis program on a single image that you uplaod inside the images folder. You can change the variable image_path by the image you want
"""

def show_crops(image,masks):
  plt.figure(figsize=(5,5))
  for mask in masks :
    crop = treat_single_mask(image, mask)
    show_image(crop)

def analyse_image(image,generator):
  print('ANALYSING...')
  #image = cv2.imread(image_path)
  image = clean_image(image)
  print('IMAGE CLEANED ')
  masks = run_model(image, generator)
  print('MODEL APPLIED')
  sizes = calculate_size_from_multiple_masks(masks)
  print('SIZES CALCULATED', sizes)
  #show_image(image,masks)
  #print(f'Analysing {image_path} ... ')
  print(f'There are , {count_number_of_particles(masks) } rocks')
  shapes = count_number_of_particles_by_size_range(masks)
  #show_crops(image, masks)
  for k,v in shapes.items() :
    print(f'Number of {k} rocks is {v}')
  PSD(sizes)
  summary = table(sizes)
  print(summary)

"""# Automation of analysis with multiple images

you got to put all your images inside the images folder
"""

import os
def analyse_images_in_folder(folder):
  for image_filename in os.listdir(folder):
    if image_filename.split('.')[-1] in ['jpg','jpeg','png']:
      try:
        image_path = folder+'/'+image_filename
        analyse_image(image_path, mask_generator)
      except Exception as e :
        print(f'error with {image_filename}, continue... ')
        print(e)
        continue
      
def get_image_fn(path):
    print(os.listdir(''))

    return list(os.listdir(path))
   

if __name__ == '__main__':
    #print('hey')
    analyse_images_in_folder('images')

