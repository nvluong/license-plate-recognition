# License-plate-recognition
Problems include :
  * Define license plate container using yolo v4   
  * Use segmentation to separate each character in the number plate
  * Recognize characters in number plates using cnn
#  Approach
  * prepare license plate data and put it through yolo v4 to detect the area containing license plates in the image
  * then use image processing methods to separate the characters in the number plate area :
      * adaptiveThreshold ,Connected components analysis, GaussianBlur ...
  * use cnn network to recognize characters:
      * Conv2d x2 -> MaxPool -> Dropout -> 
        Conv2d x2 -> MaxPool -> Dropout -> 
        Conv2d x2 -> MaxPool -> Dropout -> 
        Flatten -> Dense -> Dropout -> Softmax(31)
        
# Built With
  * Python
  * Tensorflow
  * Google colab
# Data train yolo
[link](https://thigiacmaytinh.com/tai-nguyen-xu-ly-anh/tong-hop-data-xu-ly-anh/?fbclid=IwAR2tajA5Ku83kIrb09ovhmb_68Zmdwo9KvV_CSNBCTbuIIsiK_FUM4W4Dh8)

# Data digit train CNN
[link](https://www.mediafire.com/file/3l3x7bd7rq91l5r/data_digit.zip/file)

## Required settings
add the file yolov4-custom_1000 to the weights folder :
     [link](https://drive.google.com/file/d/1r09xXltB287xWtOnQFfhZwVd2LcRLMRR/view?usp=sharing)
     
# Result
![image](https://user-images.githubusercontent.com/32773852/170819572-a305c432-51c6-4fca-8767-a7afe5f52cc3.png)


There are some downsides :
  * The image is misrecognized if it is backlit
  * When the input image is angled too much, some characters will be misrecognized, There is a solution is to use a transformer network to rotate the image tilted           towards the straight image, or you can train more data with the tilted image
  * Sometimes misidentified between 8 and B, 0 and D


