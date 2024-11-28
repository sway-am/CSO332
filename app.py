import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
from easyocr import Reader
reader = easyocr.Reader(['en'])


def detect_angle_rpm(path, rpm_or_speed):
  image = path
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  brightness_value = 100
  bright_image = cv2.convertScaleAbs(gray_image, alpha=1, beta=brightness_value)
  _, thresholded_image = cv2.threshold(bright_image, 254, 255, cv2.THRESH_BINARY)
  edges = cv2.Canny(thresholded_image, 50, 150, apertureSize=3)


  lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)


  max_length = 0
  best_line = None

  if lines is not None:
      for rho, theta in lines[:, 0]:

          a = np.cos(theta)
          b = np.sin(theta)
          x0 = a * rho
          y0 = b * rho
          x1 = int(x0 + 1000 * (-b))
          y1 = int(y0 + 1000 * (a))
          x2 = int(x0 - 1000 * (-b))
          y2 = int(y0 - 1000 * (a))


          cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


          length = np.sqrt((x2 - x1)*2 + (y2 - y1)*2)
          if length > max_length:
              max_length = length
              best_line = (rho, theta)

  
  if best_line:
      _, theta = best_line
      angle = (theta * 180 / np.pi) - 90
      if angle < -20:
        angle = -20
      st.image( image)
      # st.write(f"The angle of the speedometer needle is {angle:.2f} degrees from the horizontal.")
      if rpm_or_speed == "speed":
        st.write(f"The speed of the speedometer is {angle + 20:.2f} kmph")
      else:
        st.write(f"The RPM of the engine is {(angle  + 22)/27.68 :.2f} X 1000 RPM")
      
  else:
    alt_rpm(path, rpm_or_speed)

def detect_angle_speed(path, rpm_or_speed):
  image = path
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  brightness_value = 70
  bright_image = cv2.convertScaleAbs(gray_image, alpha=1, beta=brightness_value)
  _, thresholded_image = cv2.threshold(bright_image, 254, 255, cv2.THRESH_BINARY)
  edges = cv2.Canny(thresholded_image, 50, 150, apertureSize=3)


  lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)


  max_length = 0
  best_line = None

  if lines is not None:
      for rho, theta in lines[:, 0]:

          a = np.cos(theta)
          b = np.sin(theta)
          x0 = a * rho
          y0 = b * rho
          x1 = int(x0 + 1000 * (-b))
          y1 = int(y0 + 1000 * (a))
          x2 = int(x0 - 1000 * (-b))
          y2 = int(y0 - 1000 * (a))


          cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


          length = np.sqrt((x2 - x1)*2 + (y2 - y1)*2)
          if length > max_length:
              max_length = length
              best_line = (rho, theta)

  
  if best_line:
      _, theta = best_line
      angle = (theta * 180 / np.pi) - 90
      if angle < -20:
        angle = -20
      
      # st.write(f"The angle of the speedometer needle is {angle:.2f} degrees from the horizontal.")
      st.image( image)
      if rpm_or_speed == "Speedometer":
        st.write(f"The speed of the speedometer is {angle + 20:.2f} kmph")
      else:
        st.write(f"The RPM of the engine is {(angle  + 22)/27.68 :.2f} X 1000 RPM")

  else:
    alt_speed(path, rpm_or_speed)




def alt_rpm(path, speed):
  # Load the image in grayscale

  image = path
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  brightness_value = 100
  bright_image = cv2.convertScaleAbs(image, alpha=1, beta=brightness_value)
  _, thresholded_image = cv2.threshold(bright_image, 254, 255, cv2.THRESH_BINARY)


  # Check if the image was loaded correctly
  if image is None:
      raise ValueError("Image not loaded. Check the file path.")

  # Apply adaptive thresholding to highlight the needle
  thresh = thresholded_image
  # st.image(thresh)

  # Find contours
  contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Filter contours and find the needle contour (assuming it's the largest)
  needle_contour = max(contours, key=cv2.contourArea)

  # Fit a line to the needle contour
  [vx, vy, x, y] = cv2.fitLine(needle_contour, cv2.DIST_L2, 0, 0.01, 0.01)
  vx, vy, x, y = vx.item(), vy.item(), x.item(), y.item()  # Extract scalar values

  angle_radians = np.arctan2(vy, vx)
  angle_degrees = np.degrees(angle_radians)

  # st.write the angle
  if speed == "speed":
    st.write(f"The speed of the speedometer is {angle_degrees + 20:.2f} kmph")
  else:
    st.write(f"The RPM of the engine is {(angle_degrees  + 22)/27.68 :.2f} X 1000 RPM")


  # Optional: Draw the line for visualization
  rows, cols = image.shape
  lefty = int((-x * vy / vx) + y)
  righty = int(((cols - x) * vy / vx) + y)
  result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
  cv2.line(result_image, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
  st.image(result_image)



def alt_speed(path, speed):
  # Load the image in grayscale

  image = path
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  brightness_value = 70
  bright_image = cv2.convertScaleAbs(image, alpha=1, beta=brightness_value)
  _, thresholded_image = cv2.threshold(bright_image, 254, 255, cv2.THRESH_BINARY)


  # Check if the image was loaded correctly
  if image is None:
      raise ValueError("Image not loaded. Check the file path.")

  # Apply adaptive thresholding to highlight the needle
  thresh = thresholded_image
  # st.image(thresh)

  # Find contours
  contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Filter contours and find the needle contour (assuming it's the largest)
  needle_contour = max(contours, key=cv2.contourArea)

  # Fit a line to the needle contour
  [vx, vy, x, y] = cv2.fitLine(needle_contour, cv2.DIST_L2, 0, 0.01, 0.01)
  vx, vy, x, y = vx.item(), vy.item(), x.item(), y.item()  # Extract scalar values

  angle_radians = np.arctan2(vy, vx)
  angle_degrees = np.degrees(angle_radians)

  # st.write the angle
  if speed == "Speedometer":
    st.write(f"The speed of the speedometer is {angle_degrees + 20:.2f} kmph")
  else:
    st.write(f"The RPM of the engine is {(angle_degrees  + 22)/27.68 :.2f} X 1000 RPM")


  # Optional: Draw the line for visualization
  rows, cols = image.shape
  lefty = int((-x * vy / vx) + y)
  righty = int(((cols - x) * vy / vx) + y)
  result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
  cv2.line(result_image, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
  st.image(result_image)



def detect_angle_temp_and_fuel(path, fuel):
  # Load the image in grayscale

  image = path
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  brightness_value = 100
  bright_image = cv2.convertScaleAbs(image, alpha=1, beta=brightness_value)
  _, thresholded_image = cv2.threshold(bright_image, 254, 255, cv2.THRESH_BINARY)


  # Check if the image was loaded correctly
  if image is None:
      raise ValueError("Image not loaded. Check the file path.")

  # Apply adaptive thresholding to highlight the needle
  thresh = thresholded_image
  # st.image(thresh)

  # Find contours
  contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Filter contours and find the needle contour (assuming it's the largest)
  needle_contour = max(contours, key=cv2.contourArea)

  # Fit a line to the needle contour
  [vx, vy, x, y] = cv2.fitLine(needle_contour, cv2.DIST_L2, 0, 0.01, 0.01)
  vx, vy, x, y = vx.item(), vy.item(), x.item(), y.item()  # Extract scalar values

  angle_radians = np.arctan2(vy, vx)
  angle_degrees = np.degrees(angle_radians)

  height, width = image.shape[:2]
  point1 = (int(x - vx * width), int(y - vy * width))
  point2 = (int(x + vx * width), int(y + vy * width))
  output_image = image.copy()
  cv2.line(output_image, point1, point2, (0, 255, 0), 2)
  st.image(output_image)

  # st.write the angle
  if fuel == "fuel":
    st.write(f"Angle of the fuel gauge needle: {angle_degrees:.2f} degrees")
  else:
    st.write(f"Angle of the temperature gauge needle: {angle_degrees:.2f} degrees")
  


roi = [[(494, 12), (827, 230), 'Dial', 'Speedometer'], [(600, 199), (708, 285), 'Dial', 'Fuel Guage'], [(9, 11), (328, 241), 'Dial', 'RPM Gauge'], [(115, 203), (225, 282), 'Dial', 'Engine Temperature'], [(388, 159), (484, 186), 'text', 'Average Fuel Economy'], [(407, 199), (477, 224), 'text', 'Range'], [(425, 245), (485, 264), 'text', 'Distance Travelled'], [(219, 238), (259, 264), 'Illumination', 'Upper dipper']]

def main():
    st.title("Image Processing with AKAZE and EasyOCR")

    uploaded_query_image = st.file_uploader("Upload Query Image", type=["jpg", "png", "jpeg"])
    uploaded_images = st.file_uploader("Upload Images to Compare", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_query_image and uploaded_images:
        # Load query image
        imgQ = np.array(Image.open(uploaded_query_image))
        st.image(imgQ, caption="Query Image", use_column_width=True)

        # Initialize AKAZE detector
        akaze = cv2.AKAZE_create(threshold=0.001)
        kp1, des1 = akaze.detectAndCompute(imgQ, None)

        for uploaded_image in uploaded_images:
            img = np.array(Image.open(uploaded_image))
            kp2, des2 = akaze.detectAndCompute(img, None)

            # Match descriptors using BFMatcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des2, des1)
            matches = sorted(matches, key=lambda x: x.distance)

            # Filter good matches
            good = [m for m in matches if m.distance < 60]

            if len(good) > 10:
                srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)

                if M is not None:
                    imgScan = cv2.warpPerspective(img, M, (imgQ.shape[1], imgQ.shape[0]))
                    st.image(imgScan, caption=f"Warped Perspective: {uploaded_image.name}", use_column_width=True)

                    # Add regions of interest (ROI)
                    # roi = [((50, 50), (150, 100), "text", "Speedometer"),  # Example ROI
                    #        ((200, 200), (300, 250), "Illumination", "Dipper")]  # Example ROI

                    for r in roi:
                        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
                        #st.image(imgCrop, caption=f"Cropped ROI: {r[3]}", use_column_width=True)

                        if r[2] == "text":
                            results = reader.readtext(imgCrop)
                            extracted_text = " ".join([result[1] for result in results])
                            st.write(f"Extracted Text from {r[3]}: {extracted_text}")

                        elif r[2] == "Dial":
                            if r[3] == "Speedometer":
                                detect_angle_speed(imgCrop, "Speedometer")
                            elif r[3] == "Fuel Guage":
                                detect_angle_temp_and_fuel(imgCrop, "Fuel")
                            elif r[3] == "Engine Temperature":
                                detect_angle_temp_and_fuel(imgCrop, "Temperature")
                            else:
                                detect_angle_rpm(imgCrop, "RPM")

                        elif r[2] == "Illumination":
                            st.image(imgCrop, caption=f"Cropped ROI: {r[3]}", use_column_width=True)
                            imgHSV = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
                            lower_green = np.array([40, 50, 50])
                            upper_green = np.array([80, 255, 255])
                            maskGreen = cv2.inRange(imgHSV, lower_green, upper_green)
                            greenPixels = cv2.countNonZero(maskGreen)
                            totalPixels = maskGreen.size
                            dipperStatus = 1 if greenPixels > (totalPixels / 4) else 0
                            st.write(f"Dipper Status ({r[3]}): {dipperStatus}")

                else:
                    st.write(f"Homography could not be computed for {uploaded_image.name}.")
            else:
                st.write(f"Not enough good matches for {uploaded_image.name}.")

if __name__ == "__main__":
    main()
