**Image Description Web Application**

**Overview**

The Image Description Web Application allows users to upload images and receive a textual description of the image content. The backend utilizes a pre-trained MobileNetV2 model from TensorFlow Hub to classify images based on the ImageNet dataset. The result is displayed as a description of the image content, alongside the uploaded image.

**Objective**

This application provides the following features:

**Image Upload**: Users can upload an image via a web interface.

**Image Classification**: The application uses the MobileNetV2 model to classify the image.

**Description Generation**: A description based on the image's classification is returned to the user.

**Image Display**: The uploaded image is displayed on the web page.

**API Endpoints**

POST /upload
Description: Handles image upload, processes it, and returns a description.
Request: A POST request containing the image file.
Response: JSON containing the description of the image and the image URL. Example response:

{
  "description": "A dog, a pet animal",
  "image_url": "http://127.0.0.1:5000/uploads/<uploaded_image>"
}

**Deployment**

Cloud Deployment
Deploy the app in render.
Live website link - https://img-descr-arogo.onrender.com/
