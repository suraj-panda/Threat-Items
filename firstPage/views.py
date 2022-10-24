from django.shortcuts import render
from django.http import HttpResponse
from django.contrib import messages
from django.core.files.storage import FileSystemStorage

import torch

#YOLOV5 Requirements
from PIL import Image
import matplotlib.pyplot as p
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.transform import resize
nor_yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/best640x640.pt')
clahe_yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/bestclahe640x640.pt')
#import matplotlib.pyplot as plt


# Create your views here.
def index(request):
    return render(request, 'index.html')

def custom1(request):
    import cv2
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    img_counter = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "{}.jpg".format(img_counter+1)
            print(img_name)
            cv2.imwrite('./basic/' + img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
            import os
            num = next(os.walk('./basic/'))[2]
            image1 = Image.open('./basic/' + num[0])
            image1 = image1.resize((640, 640))
            image1_size = image1.size
            image1.save("./media/" + img_name,"JPEG")
    cam.release()
    cv2.destroyAllWindows()
    return render(request, 'index.html')



def custom2(request):
    import cv2
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    img_counter = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            print("New")
            print(frame)
            img_name = "{}.jpg".format(img_counter+1)
            print(img_name)
            cv2.imwrite('./basic/' + img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
            import os
            num = next(os.walk('./basic/'))[2]
            image1 = Image.open('./basic/' + num[0])
            image1 = image1.resize((640, 640))
            image1_size = image1.size
            image1.save("./media/" + img_name,"JPEG")
    cam.release()
    cv2.destroyAllWindows()
    return render(request, 'index.html')

def predict1(request):
    import cv2
    print(request)
    print(request.POST.dict())
    fileobj = request.FILES['image']
    print(fileobj)
    fs = FileSystemStorage()
    filePathName = fs.save(fileobj.name, fileobj)
    print(filePathName)
    filePathName = fs.url(filePathName)
    print(filePathName)

    name = str(fileobj)
    n = name.split(".")


    import os
    ids = next(os.walk('./media/'))[2]
    l = len(ids)
    print(ids)

    X = np.zeros((2, 640, 640, 3), dtype=np.float32)

    img = load_img('./media/'+n[0]+'.jpg', grayscale=False)
    x_img = img_to_array(img)
    x_img = resize(x_img, (640, 640, 3), mode = 'constant', preserve_range = True)
    X[0] = x_img/255.0
    #img_name = "{}.jpg".format(img_counter+1)
    results = nor_yolo_model(img)
    #print(results)
    #cv2.imwrite('./results/' + img_name, results)
    #results.save(save_dir='./results/' + n[0] + '.jpg')
    results.save(save_dir='./results/' + n[0] + '.jpg')

    num = next(os.walk('./results/' + n[0] + '.jpg/'))[2]
    print(num)
    img2 = load_img('./results/' + n[0] + '.jpg/' + num[0], grayscale=False)
    y_img = img_to_array(img2)
    y_img = resize(y_img, (640, 640, 3), mode = 'constant', preserve_range = True)
    X[1] = y_img/255.0

    input_t = X[0:X.shape[0] // 2]
    input_d = X[X.shape[0] // 2:]
    print(input_t.shape)
    print(input_d.shape)

    num = next(os.walk('./results/' + n[0] + ".jpg/"))[2]
    image1 = Image.open('./media/' + n[0] + '.jpg')
    image2 = Image.open("./results/" + n[0] + ".jpg/" + num[0])
    image1 = image1.resize((640, 640))
    image1_size = image1.size
    image2_size = image2.size
    new_image = Image.new('RGB',(2*image1_size[0], image1_size[1]), (250,250,250))
    new_image.paste(image1,(0,0))
    new_image.paste(image2,(image1_size[0],0))
    new_image.save("./merge/merged_image.jpg","JPEG")
    new_image.show()
    return render(request, 'index.html')

def predict2(request):
    import cv2
    print(request)
    print(request.POST.dict())
    fileobj = request.FILES['image']
    print(fileobj)
    fs = FileSystemStorage()
    filePathName = fs.save(fileobj.name, fileobj)
    print(filePathName)
    filePathName = fs.url(filePathName)
    print(filePathName)

    name = str(fileobj)
    n = name.split(".")


    import os
    ids = next(os.walk('./media/'))[2]
    l = len(ids)
    print(ids)

    X = np.zeros((2, 640, 640, 3), dtype=np.float32)

    img = load_img('./media/'+n[0]+'.jpg', grayscale=False)
    x_img = img_to_array(img)
    x_img = resize(x_img, (640, 640, 3), mode = 'constant', preserve_range = True)
    X[0] = x_img/255.0
    #img_name = "{}.jpg".format(img_counter+1)
    results = clahe_yolo_model(img)
    #print(results)
    #cv2.imwrite('./results/' + img_name, results)
    #results.save(save_dir='./results/' + n[0] + '.jpg')
    results.save(save_dir='./results/' + n[0] + '.jpg')

    num = next(os.walk('./results/' + n[0] + '.jpg/'))[2]
    print(num)
    img2 = load_img('./results/' + n[0] + '.jpg/' + num[0], grayscale=False)
    y_img = img_to_array(img2)
    y_img = resize(y_img, (640, 640, 3), mode = 'constant', preserve_range = True)
    X[1] = y_img/255.0

    input_t = X[0:X.shape[0] // 2]
    input_d = X[X.shape[0] // 2:]
    print(input_t.shape)
    print(input_d.shape)

    num = next(os.walk('./results/' + n[0] + ".jpg/"))[2]
    image1 = Image.open('./media/' + n[0] + '.jpg')
    image2 = Image.open("./results/" + n[0] + ".jpg/" + num[0])
    image1 = image1.resize((640, 640))
    image1_size = image1.size
    image2_size = image2.size
    new_image = Image.new('RGB',(2*image1_size[0], image1_size[1]), (250,250,250))
    new_image.paste(image1,(0,0))
    new_image.paste(image2,(image1_size[0],0))
    new_image.save("./merge/merged_image.jpg","JPEG")
    new_image.show()
    return render(request, 'index.html')