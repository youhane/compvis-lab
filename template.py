from distutils import filelist
import cv2 as cv
import os 
import numpy as np
import math
from matplotlib import pyplot as plt

def get_path_list(root_path):
    '''
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing the names of the sub-directories in the
            root directory
    '''
    path_list = []
    for path in os.listdir(root_path):
        path_list.append(path)
    return path_list

def get_class_id(root_path, train_names):
    '''
        To get a list of train images and a list of image classes id

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories
        Returns
        -------
        list
            List containing all image in the train directories
        list
            List containing all image classes id
    '''
    listImages = []
    listClasses = []

    for root, _, files in os.walk(root_path):
        for file in files:
            if file.endswith('jpg'):
                listImages.append(os.path.join(root, file))
                # listClasses.append(os.path.basename(root))
                listClasses.append(train_names.index(os.path.basename(root)))
        
    return listImages, listClasses




def detect_faces_and_filter(image_list, image_classes_list=None):
    '''
        To detect a face from given image list and filter it if the face on
        the given image is less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list, optional
            List containing all image classes id
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
        list
            List containing all filtered image classes id
    '''  
    filtered_cropped_images = []
    filtered_faces_rects = []
    filtered_image_classes_list = []
    face_cascade = cv.CascadeClassifier('haar_face.xml')
    
    for i, image in enumerate(image_list):
        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 1:
            for (x, y, w, h) in faces:
                filtered_cropped_images.append(gray[y:y+h, x:x+w])
                filtered_faces_rects.append((x, y, w, h))
                if image_classes_list:
                    filtered_image_classes_list.append(image_classes_list[i])
    return filtered_cropped_images, filtered_faces_rects, filtered_image_classes_list


def train(train_face_grays, image_classes_list):
    '''
        To create and train face recognizer object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id
        
        Returns
        -------
        object
            Recognizer object after being trained with cropped face images
    '''
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.train(train_face_grays, np.array(image_classes_list))
    return recognizer

def get_test_images_data(test_root_path):
    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory
        
        Returns
        -------
        list
            List containing all loaded gray test images
    '''
    path_list = []
    root, _, files = next(os.walk(test_root_path))

    for file in files:
        if file.endswith('jpg'):
            path_list.append(os.path.join(root, file))

    return path_list
    
def predict(recognizer, test_faces_gray):
    '''
        To predict the test image with the recognizer

        Parameters
        ----------
        recognizer : object
            Recognizer object after being trained with cropped face images
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''
    predict_results = []
    for face in test_faces_gray:
        predict_results.append(recognizer.predict(face))
    return predict_results
    
def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):
    '''
        To draw prediction results on the given test images and acceptance status

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all test images after being drawn with
            final result
    '''
    for i, image in enumerate(test_image_list):
        if predict_results[i][1] < 50:
            cv.putText(image, train_names[predict_results[i][0]], (test_faces_rects[i][0], test_faces_rects[i][1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv.putText(image, 'Unknown', (test_faces_rects[i][0], test_faces_rects[i][1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return test_image_list

def combine_and_show_result(image_list):
    '''
        To show the final image that already combine into one image

        Parameters
        ----------
        image_list : nparray
            Array containing image data
    '''
    image_list = np.concatenate(image_list, axis=1)
    cv.imshow('Result', image_list)
    cv.waitKey(0)
    cv.destroyAllWindows()

'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''
if __name__ == "__main__":

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_root_path = "dataset/train"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    train_names = get_path_list(train_root_path) #labels_list
    train_image_list, image_classes_list = get_class_id(train_root_path, train_names) #faces, indexes
    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    recognizer = train(train_face_grays, filtered_classes_list)

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_root_path = "dataset/test"
    '''
        -------------------
        End of modifiable
        -------------------
    '''
    test_image_list = get_test_images_data(test_root_path)
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list)
    predict_results = predict(recognizer, test_faces_gray)
    # predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names)
    
    # combine_and_show_result(predicted_test_image_list)