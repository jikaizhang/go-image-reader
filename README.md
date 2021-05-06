# go-image-reader

# Procedure:
1. I uploaded 20 images containing a Go board.
2. Because I want to know the accuracy of my model, so I first used a Go app called Sabaki to reproduce the current stone positions, then generate Smart Game Format(sgf) files from it, and read and store the stone information into a 2D array, I will later test my model against expected board infos.
3. To make it easier to understand the output of my model, I implemented a tiny GUI using Pygame to visualize the board from a 2D array.
4. Then it's time to start building the model.
5. The first task is to locate the board and crop it. I have tried to use Canny edge detection, and then apply OpenCV's findContours() function to find the quadrilateral with largest area, but most of the times the model fails to locate the whole board. It does not help much if I blur the image with Gaussian blur or median blur. So I decided to attempt an alternative.
6. The alternative I use is: let the user select the four corners of the game board on the image.
7. Once the corners are selected, I use OpenCV's getPerspectiveTransform() function to flatten the cropped image. This will be the board we are going to analyze,
8. Next is detecting the stones. The first method that I tried is a naive one. Because we know that a Go board is 19 * 19, so we know the expected position of each junction. So I choose some points around the junctions and check if their pixel values are below a threshold (black) or above another threshold (white). With this method, I reach an accuracy of ~86% on detecting whether a junction is a stone or empty, and ~84% on detecting both the existence and the color of the stone.
9. I am dissatisfied with this result, so I try another way. I used OpenCV's HoughCircles() function to find all circles (stones) on the board, and then classify its color. 
10. I also did an optimization with color classification, because the brightness of the images varies, so it does not make sense to hard code a threshold. Instead, I calculate the median pixel value of all the stones on the board, and then compare each stone with this median.
11. With the above approach, the model reached 95% accuracy in stone existence classification, and 94.4% in classifying both the existence and the color.
12. I try to find the similarities amoung the images with low test accuracy, and I figured out the effect of light on the accuracy. My model tends to perform bad when there is bright spots or reflections on the board. The image below is an example of the bright glares:
<img src="https://github.com/jikaizhang/go-image-reader/blob/main/images/11.JPG">
13. So I used CLAHE (Contrast Limited Adaptive Histogram Equalization) to reduce some bright reflections, then the accuracies increased to 96.2% and 95.5%, respectively.
14. Finally, I added some popups to be more user friendly. I give the users options of uploading his/her own board image, or use the images I provided.

# How to use it:

1. Run main.py, it will popup a window, asking you if you want to upload your own Go board image.
<img src="https://github.com/jikaizhang/go-image-reader/blob/main/instructions/1.jpg">
2. If you choose to upload your own, then you will need to type the image path.
<img src="https://github.com/jikaizhang/go-image-reader/blob/main/instructions/2.jpg">
3. Then you need to click on the four corners of the game board.
<img src="https://github.com/jikaizhang/go-image-reader/blob/main/instructions/3.jpg">
4. Then it will show you the cropped game board.
<img src="https://github.com/jikaizhang/go-image-reader/blob/main/instructions/4.jpg">
5. Next, the program analyzes the image, and show you the result of its recognition in a GUI. When you click the close button of the GUI, you can see the original image again and compare the result.
<img src="https://github.com/jikaizhang/go-image-reader/blob/main/instructions/5.jpg">
6. If you choose to use our preprocessed images.
<img src="https://github.com/jikaizhang/go-image-reader/blob/main/instructions/6.jpg">
<img src="https://github.com/jikaizhang/go-image-reader/blob/main/instructions/7.jpg">
7. Then the program will go through all the images and show you the test accuracy.
<img src="https://github.com/jikaizhang/go-image-reader/blob/main/instructions/8.jpg">
8. If you choose to use our original images, then you will need to select the four corners of every image, and the program will show you the overall accuracy, pretty much a combination of the above two options.