from torchvision import transforms

def prepare_image(img):
    h, w, c = img.shape
    s = min(h, w)

    y1 = (h-s)/2
    y2 = y1 + s
    img_crop = img[y1:y2, 640-s:640, :]

    img_resized = cv2.resize(img_crop, (244, 244))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img_resized)
    return img_tensor