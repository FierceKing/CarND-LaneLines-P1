import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


class LaneDetect:
	def __init__(self, output_level={1, 5}, smooth_level=10, only_image=False):
		self.lane_detected = [[[(0, 0), (0, 0)], [(0, 0), (0, 0)]]]  # storage depend on smooth_level
		self.filtered_lane = []  # average frames to get stable lane lines
		self.validity_marker = [[[0], [0]]]  # if in a frame a lane cannot be detected, it's marker set to 0
		self.processed_image = [0]  # store image in different output level
		self.input_frame = [0]  # frame passed to class
		self.output_frame = [0]  # frame to return
		self.MAX_OUTPUT_LEVEL = {1, 2, 3, 4, 5}  # output complexity
		# 1. original frame
		# 2. Canny edge detected
		# 3. polygon for mask region of interest
		# 4. Hough lines detected
		# 5. Lane lines calculated
		self.output_level = self.MAX_OUTPUT_LEVEL & output_level
		# only_image marker to identify input file type
		if only_image:
			self.smooth = 1
		self.smooth = smooth_level

	# main processing unit
	def frame_process(self, input_frame):
		# copy input frame into class
		self.input_frame = input_frame
		# create empty processed_image
		self.processed_image = np.zeros_like(input_frame)
		# initialize output_frame
		self.output_frame = np.copy(self.input_frame)
		# calculate the lane in current frame
		new_lane = self._lane_calculation()
		# add lane_detected buffer and update validity marker
		self._add_to_buff(new_lane)
		# apply average filter to smooth the output
		self.filtered_lane = self._lane_filter()
		# determine the output frame style according to output_level and smooth_level
		self._determine_output()
		return self.output_frame

	# determine the output frame style according to output_level and smooth_level
	def _determine_output(self):
		if 5 in self.output_level:
			# drawing lane lines need to be done here,
			# due to smoothed lane lines are calculated after _lane_calculation
			lane_image = self._draw_lane_layer()  # create a layer only contains calculated lane lines.
			self.processed_image = cv2.addWeighted(lane_image, 0.8, self.processed_image, 1, 0)
			self.output_frame = np.copy(self.processed_image)
		else:
			self.output_frame = np.copy(self.processed_image)

		return 0

	# average filter
	def _lane_filter(self):

		# convert coordinates to opencv supported format
		def convert_to_tuple(arr):
			r = arr.tolist()
			tp = [[(r[0][0][0], r[0][0][1]), (r[0][1][0], r[0][1][1])],
			      [(r[1][0][0], r[1][0][1]), (r[1][1][0], r[1][1][1])]]
			return tp

		# calculate filtered lane
		if len(self.lane_detected) <= self.smooth - 1:  # when frames are not enough for smooth
			return self.lane_detected[-1]
		# convert to numpy array for matrix calculation, originally there are tuple format data
		lane_array = np.array(self.lane_detected)
		marker_array = np.array(self.validity_marker)

		# average all detected lanes when the data is valid
		filtered_lane = np.sum(lane_array, axis=0) / np.sum(marker_array, axis=0)
		filtered_lane = np.array(filtered_lane, dtype=int)  # convert to int for coordinates
		filtered_lane = convert_to_tuple(filtered_lane)  # convert back to opencv supported format
		return filtered_lane

	# add detected lanes to buffer, and modify validity marker accordingly
	def _add_to_buff(self, new_lane):
		self.lane_detected.append(new_lane)
		new_marker = [[0], [0]]
		for i in range(2):
			if new_lane[i] != [(0, 0), (0, 0)]:
				new_marker[i] = [1]
		self.validity_marker.append(new_marker)
		self.lane_detected = self.lane_detected[-self.smooth:]
		self.validity_marker = self.validity_marker[-self.smooth:]
		return 0

	# plot a layer that only contains calculated lane lines
	def _draw_lane_layer(self):
		lanes = self.filtered_lane  # only to make the variable shorter for readability
		lane_image = np.zeros_like(self.output_frame)
		cv2.line(lane_image, lanes[0][0], lanes[0][1], color=(0, 0, 200), thickness=10)
		cv2.line(lane_image, lanes[1][0], lanes[1][1], color=(0, 0, 200), thickness=10)
		return lane_image

	# method to find lane lines using polyfit
	# lines are hough lines, boarder is the y boundary for the polygon mask
	@staticmethod
	def _find_lane(lines, boarder):
		# sort the coordinates for easy access
		def sort_coordinates(lines):
			transposed_lines = np.transpose(lines)
			x = transposed_lines[0][0]
			x = np.append(x, transposed_lines[2][0])
			y = transposed_lines[1][0]
			y = np.append(y, transposed_lines[3][0])
			return x, y

		if len(lines) == 0:  # if no Hough lines detected, return 0 coordinates
			return [(0, 0), (0, 0)]
		sx, sy = sort_coordinates(lines)

		z = np.polyfit(sy, sx, 1)  # fitting the Hough lines vertices to 1 order liner regression
		p1 = np.poly1d(z)
		xs = [0, 0]
		for i in range(2):
			xs[i] = np.uint32(p1(boarder[i]))  # calculate x given y
		return [(xs[0], boarder[0]), (xs[1], boarder[1])]  # return a line

	# the core function for computer vision
	# update self.processed_image according to output level
	def _lane_calculation(self):
		original_img = np.copy(self.input_frame)
		if 1 in self.output_level:  # original picture
			self.processed_image = np.copy(original_img)
		stage1_gray_img = cv2.cvtColor(self.input_frame, cv2.COLOR_RGB2GRAY)

		# Define a kernel size and apply Gaussian smoothing
		kernel_size = 5
		stage2_blur_gray_img = cv2.GaussianBlur(stage1_gray_img, (kernel_size, kernel_size), 0)

		# Define our parameters for Canny and apply
		low_threshold = 100
		high_threshold = 200
		stage3_edges_img = cv2.Canny(stage2_blur_gray_img, low_threshold, high_threshold)
		if 2 in self.output_level:  # edge layer
			stage3_3c_edge_img = np.dstack((stage3_edges_img, stage3_edges_img, stage3_edges_img))
			self.processed_image = cv2.addWeighted(stage3_3c_edge_img, 0.8, self.processed_image, 1.0, 0)

		# Next we'll create a masked edges image using cv2.fillPoly()
		patch1_polygon_mask = np.zeros_like(stage3_edges_img)
		ignore_mask_color = 255

		# This time we are defining a four sided polygon to mask
		imshape = self.input_frame.shape
		y_level = 320
		vertices = np.array([[(75, imshape[0]), (imshape[1] - 55, imshape[0]),
		                      (530, y_level), (430, y_level)]],
		                    dtype=np.int32)
		cv2.fillPoly(patch1_polygon_mask, vertices, ignore_mask_color)
		patch2_valid_mask = cv2.bitwise_and(stage3_edges_img, patch1_polygon_mask)
		if 3 in self.output_level:  # polygon layer
			cv2.polylines(self.processed_image, vertices, isClosed=True, color=(0, 255, 0), thickness=2)

		# Define the Hough transform parameters
		# Make a blank the same size as our image to draw on
		rho = 1  # distance resolution in pixels of the Hough grid
		theta = np.pi / 180  # angular resolution in radians of the Hough grid
		threshold = 10  # minimum number of votes (intersections in Hough grid cell)
		min_line_length = 5  # minimum number of pixels making up a line
		max_line_gap = 3  # maximum gap in pixels between connectable line segments

		# Run Hough on edge detected image
		# Output "lines" is an array containing endpoints of detected line segments
		hough_lines = cv2.HoughLinesP(patch2_valid_mask, rho, theta, threshold, np.array([]),
		                              min_line_length, max_line_gap)

		# divide lines into 2 categories, left & right
		left_lines = []
		right_lines = []
		patch3_hough_line_mask = np.zeros_like(self.input_frame)  # creating a blank to draw lines on
		# iterate to plot identified lines on blank image,
		# and divide all lines into left & right categories
		for line in hough_lines:
			for x1, y1, x2, y2 in line:
				cv2.line(patch3_hough_line_mask, (x1, y1), (x2, y2), (255, 0, 0), 2)
				# ignore lines that are horizontal
				if abs(x1 - x2) < abs(y1 - y2):
					continue
				# these are lines mostly on the left
				elif 480 - x1 + 480 - x2 > 0:
					left_lines.append(line)
				# these are lines mostly on the right
				elif x1 > 480 or x2 > 480:
					right_lines.append(line)
				# if there're lines that cannot be categorized, print message, for debug
				else:
					print("This line is not categorized: ", line)
				# generate patch that only contains identified lines

		if 4 in self.output_level:  # hough lines layer
			self.processed_image = cv2.addWeighted(patch3_hough_line_mask, 1.0, self.processed_image, 1.0, 0)

		# find fit using hough line's coordinates
		left_lane = self._find_lane(left_lines, boarder=[y_level, imshape[0]])
		right_lane = self._find_lane(right_lines, boarder=[y_level, imshape[0]])

		return [left_lane, right_lane]


def main():
	# initialize LaneDetect object, giving output level and smooth level
	detector = LaneDetect(output_level={1, 3, 4, 5}, smooth_level=10)
	video_file = cv2.VideoCapture('./test_videos/solidYellowLeft.mp4')  # open video file
	fps = video_file.get(cv2.CAP_PROP_FPS)  # get fps
	size = (int(video_file.get(cv2.CAP_PROP_FRAME_WIDTH)),
	        int(video_file.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # get video resolution
	# create video writer object
	video_writer = cv2.VideoWriter('MyOutputVid.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

	# read in 1 frame, and then loop until video is finished
	success, frame = video_file.read()
	while success:
		processed_frame = detector.frame_process(frame)  # pass the frame to process
		cv2.imshow('processed_frame', processed_frame)  # show processed frame window
		video_writer.write(processed_frame)  # write it to file
		success, frame = video_file.read()
		if cv2.waitKey(1) & 0xFF == ord('q'):  # process can be interrupted by pressing 'q'
			break

	video_file.release()  # release handles
	video_writer.release()
	cv2.destroyAllWindows()


def main2():
	detector = LaneDetect(only_image=True)  # initialize LaneDetect object with only_image option
	image = mpimg.imread('./test_images/solidWhiteRight.jpg')  # read in picture
	processed_image = detector.frame_process(image)  # pass the picture for process
	plt.imshow(processed_image)  # show the processed picture
	plt.show()


if __name__ == '__main__':
	main()  # entrance for video process
	# main2()  # entrance for picture process
