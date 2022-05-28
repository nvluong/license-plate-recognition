# License-plate-recognition
Problems include :
  * Define license plate container using yolo v4   
  * Use segmentation to separate each character in the number plate
  * Recognize characters in number plates using cnn
#  Approach
  * prepare license plate data and put it through yolo v4 to detect the area containing license plates in the image
  * then use image processing methods to separate the characters in the number plate area :
      * adaptiveThreshold ,Connected components analysis, GaussianBlur ...
## Required settings
add the file yolov4-custom_1000 to the weights folder :
     [link](https://drive.google.com/file/d/1r09xXltB287xWtOnQFfhZwVd2LcRLMRR/view?usp=sharing)


