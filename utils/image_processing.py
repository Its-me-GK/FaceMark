import cv2
import numpy as np

def resize_image(image, width=None, height=None):
    if width is None and height is None:
        return image
    (h, w) = image.shape[:2]
    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def apply_clahe_filter(image):
    """
    Convert the input RGB image to LAB, apply CLAHE on the L-channel,
    and convert back to RGB.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

# def apply_clahe_filter(image):
#     """
#     Convert the input RGB image to LAB, apply CLAHE on the L-channel,
#     and convert back to RGB, with focus on facial areas.
#     """
#     # Convert the input RGB image to LAB
#     lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
#     l, a, b = cv2.split(lab)
    
#     # Create a mask for the facial regions (example coordinates)
#     mask = np.zeros_like(l)
#     h, w = l.shape
#     x, y, face_w, face_h = w // 3, h // 3, w // 3, h // 3
#     mask[y:y+face_h, x:x+face_w] = 255

#     # Apply CLAHE on the L-channel within the masked regions only
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     cl = l.copy()
#     cl[mask == 255] = clahe.apply(l[mask == 255])
    
#     # Merge channels and convert back to RGB
#     limg = cv2.merge((cl, a, b))
#     return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)


def apply_hist_eq_filter(image):
    """
    Convert the input RGB image to grayscale, apply histogram equalization,
    and then convert back to RGB.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    eq = cv2.equalizeHist(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)

def correct_orientation(image):
    """
    Attempt to detect skew and correct image orientation using image moments.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def apply_light_filter(image, gamma=1.2):
    """
    Apply a gentle gamma correction to brighten the image.
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_bluish_filter(image):
    """
    Enhance the blue component by converting the image to LAB and reducing the 'b' channel.
    (In LAB, the b channel represents blue-yellow; subtracting from b adds a bluish tint.)
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    # Subtract a fixed value (e.g. 20) from b channel; adjust this value if needed.
    l = cv2.subtract(l, 20)
    b = cv2.subtract(a, 20)
    # b = cv2.subtract(a, 30)
    lab_bluish = cv2.merge((l, a, b))
    return cv2.cvtColor(lab_bluish, cv2.COLOR_LAB2RGB)

def apply_sharpening_filter(image):
    """
    Apply a sharpening kernel to the image to enhance edges and details.
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def enhance_facial_features(image, amount=1.0, threshold=0):
    """
    Apply a mild unsharp mask to boost facial edges.
    This can help the detector by enhancing important facial details.
    """
    image_float = image.astype(np.float32)
    blurred = cv2.GaussianBlur(image_float, (9, 9), 10.0)
    mask = image_float - blurred
    mask = np.where(np.abs(mask) > threshold, mask, 0)
    sharpened = image_float + amount * mask
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened

def apply_night_vision_filter(image):
    """
    Simulate a night vision effect by converting to grayscale,
    applying histogram equalization, then applying a green colormap.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    eq = cv2.equalizeHist(gray)
    # Apply a greenish colormap. COLORMAP_SUMMER is a good candidate.
    colored = cv2.applyColorMap(eq, cv2.COLORMAP_SUMMER)
    # Convert from BGR to RGB.
    night_vision = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return night_vision


def apply_bluish_filter_v2(image):
    """
    Convert the image to LAB and gently scale the a channel (green–red axis)
    to achieve a subtle bluish tint without overprocessing.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    # Scale the a channel by a factor (e.g. 0.95) to shift color toward blue.
    new_a = np.clip(a * 0.95, 0, 255).astype(np.uint8)
    lab_modified = cv2.merge((l, new_a, b))
    return cv2.cvtColor(lab_modified, cv2.COLOR_LAB2RGB)
