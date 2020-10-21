# God-Eye
personal repo of final year project

### Object Distance: ###
  This code fetches the distance between the object on the left and then finds its distance from all other objects and also displays all the objects that are less than 5 inches away.
    
  The live image module file takes a live video frame from a web camera and finds the object on the left and then finds its distance from all other objects and also displays all the objects that are less than 5 inches away. 
 
  ### Child and Fork Detection: ###
 Collect at least 500 images that contain your object — The bare minimum would be about 100, ideally more like 1000 or more, but, the more images you have, the more tedious step 2 will be.
#### Annotate/label the images : #### 
I am personally using LabelImg. This process is basically drawing boxes around your object(s) in an image. The label program automatically will create an XML file that describes the object(s) in the pictures.
pip3 install labelImg
labelImg //To open the annotation tool

#### Generate TFRecord : ####
Split this data into train/test samples. Training data should be around 80% and testing around 20%.
Generate TF Records from these splits.

#### Training Process : ####
Setup a .config file for the model of choice (you could train your own from scratch, but we’ll be using transfer learning).
Train our model.

#### Testing Model : ####
Export inference graph from new trained model.
Detect custom objects in real time.

#### Final Output of Colision : ####
![alt text](https://github.com/hack-e-d/God-Eye/blob/master/colision.jpg)

