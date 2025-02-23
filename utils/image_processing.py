import cv2

def resize_image(image, width=None, height=None):
    """Resize an image while maintaining the aspect ratio."""
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

def apply_filters(image):
    """Apply CLAHE filtering to enhance low-light images."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


# def apply_filters(image):
#     """
#     Convert the input RGB image to grayscale, apply histogram equalization to enhance contrast,
#     and then convert back to RGB. This can work similarly to a black & white filter that enhances contrast.
#     """
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     # Apply histogram equalization to improve contrast
#     eq = cv2.equalizeHist(gray)
#     # Convert back to RGB by replicating the single channel
#     rgb_eq = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
#     return rgb_eq