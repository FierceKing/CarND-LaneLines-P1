# **Finding Lane Lines on the Road** 

-- ***by Meng Wang***


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps: 
1. Converted the images to grayscale

<div align=center>
![alt text](./example/grayscale.jpg)
</div>

2. Apply Gaussian blur
3. Using Canny edge detection to find edges
4. Apply a polygon to mask unwanted edges
5. Find lines using Hough line detection method
- Find left and right lane lines using the vertices of the detected Hough lines.
6. draw lanes on the original image

In order to draw a single line on the left and right lanes, I modified the hough_lines() function to produce the lane lines by modifying another function process_lines(). the function process_lines() would take the hough lines and return calculated lane lines using a new function find_lane(). The find_lane() function use liner regression of the hough line vertices to do the job.

If you'd like to include images to show how the pipeline works, here is how to include an image: 




### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
