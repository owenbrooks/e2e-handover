from torchvision import transforms
import cv2

def prepare_image(img):

    transform = transforms.Compose([
        transforms.ToPILImage(), # from numpy array
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img)
    return img_tensor