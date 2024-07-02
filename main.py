#importing required libaraies
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd

attendance_csv = r'C:\Users\rhyth\OneDrive\Documents\py[1]\attendanceeee\attendance.csv'


def clear_attendance_file():
    with open(r'C:\Users\rhyth\OneDrive\Documents\py[1]\attendanceeee\attendance.csv', 'w') as file:
    

        #  it reduces the file size to 0 bytes, effectively clearing all contents of the file.
        # he truncate() method is used to resize the file. If the specified size is larger than the
        # current file size, the file is extended, and the extended part contains null bytes. If the 
        # specified size is smaller, the file is truncated, and the data beyond the specified size is discarded.
        file.truncate(0)  #

clear_attendance_file()


# set the path to the folder containing the images to be used for recognition
image_folder_path = 'C:\\Users\\rhyth\\OneDrive\\Documents\\py[1]\\attendanceeee\\images'


# create empty lists to store the images and their corresponding class names
images_list = []
class_names = []


# get a list of all the files in the specified folder
file_list = os.listdir(image_folder_path)
print(file_list)


# loop through each file in the folder
for file_name in file_list:
    # read the image file using OpenCV and add it to the images list
    # it returns an image object, which is essentially a NumPy array representing the image.
    current_image = cv2.imread(f'{image_folder_path}/{file_name}')
    images_list.append(current_image)
    # extract the class name from the file name and add it to the class_names list\

    # This function os.path.splitext(path) takes a file path (or just a file name) and splits it into a tuple
    # containing two parts: the root name and the extension
    class_names.append(os.path.splitext(file_name)[0])


# print the list of images and their class names
#print(images)
#print(classNames)


# define a function to encode the face features in a given image
def find_encodings(images):
    # create an empty list to store the face encodings
    encoding_list = []
    # loop through each image in the images list
    for image in images:
        # convert the image from BGR (OpenCV's default color format) to RGB (the format expected by the face_recognition library)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # use the face_recognition library to generate a list of face encodings in the image (we assume there is only one face per image)
        # The face_recognition.face_encodings function detects faces in an image and
        # generates a unique numerical representation (encoding) for each detected face. 
        # These encodings can be used to compare and recognize faces.
        # The face_recognition.face_encodings(image) function returns a list of face encodings. 
        encoding = face_recognition.face_encodings(image)[0]
        # add the encoding to the encoding_list
        encoding_list.append(encoding)
    # return the list of face encodings
    return encoding_list


# define a function to mark attendance for a given person
def mark_attendance(name):
    # open the attendance file in read-write mode
    with open(r'C:\Users\rhyth\OneDrive\Documents\py[1]\attendanceeee\attendance.csv', 'r+') as file:
        # read all the lines from the file and store them in a list
        data_list = file.readlines()
        # create an empty list to store the names already present in the file
        name_list = []

        # loop through each line in the file
        for line in data_list:
            # split the line by comma and extract the first element (the name) and add it to the name_list
            entry = line.split(',')
            name_list.append(entry[0])

        # if the name is not already in the name_list
        if name not in name_list:
            # get the current date and time
            # 2024-07-01 12:34:56.789012 it results  in
            now = datetime.now()
            date_time_string = now.strftime("%m/%d/%Y, %H:%M:%S")
            # write the name and the current date/time to the file (in CSV format)
            file.writelines(f'\n{name},{date_time_string}')




# encode the face features in the images
encoded_faces = find_encodings(images_list)
# print a message to indicate that the encoding is complete
print('encoding complete')


# create a new video capture object using the default camera
video_capture = cv2.VideoCapture(0)


# start an infinite loop to process each frame of the video stream
while True:
    # read a frame from the video stream
    #read will return tuples of two elements ..firs is the boolearn,second is the fram
    success, frame = video_capture.read()
    # resize the image to a smaller size (to speed up processing)
    # The scale factor in the context of image resizing refers to the ratio by which the dimensions of the
    # image are multiplied to achieve the desired size. It determines how much the image will be scaled up or
    # down along the respective axis (horizontal or vertical).

# Resize the frame to a smaller size to speed up processing
# frame: The source image to be resized
# (0, 0): The desired size (width, height). Using (0, 0) allows the function to use the scaling factors instead
# None: The optional interpolation method. None defaults to the default interpolation
# 0.25: The scale factor along the horizontal axis (width)
# 0.25: The scale factor along the vertical axis (height)

#  interpolation is used when resizing or transforming images to determine how new pixel values are calculated from existing ones


    small_frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    # convert the image from BGR to RGB
    small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all faces in the current frame

    # face_recognition.face_locations(img): This function detects the locations of all 
    # faces in the image imgS. It returns a list of tuples, with each tuple representing the
    # coordinates of a face in the format (top, right, bottom, left).

    # top: The y-coordinate of the top edge of the face bounding box.
    # right: The x-coordinate of the right edge of the face bounding box.
    # bottom: The y-coordinate of the bottom edge of the face bounding box.
    # left: The x-coordinate of the left edge of the face bounding box.
    face_locations = face_recognition.face_locations(small_frame)
    # Encode the faces in the current frame
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    # Loop through each face and try to recognize it
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare the current face to the known faces and get a list of matches



        matches = face_recognition.compare_faces(encoded_faces, face_encoding)

        # Calculate the distance between the current face and each known face
        # When we talk about "face distance," we're actually measuring how similar or different two face 
        # fingerprints are. This similarity is calculated using a mathematical formula that checks how far apart 
        # these fingerprints are in a 128-dimensional space.The smaller the face distance, the more alike two faces are
        #  according to their fingerprints.
        #Imagine you have two face encodings:
        # encoding_1 = [0.1, 0.2, ..., 0.9] (128 numbers)
        # encoding_2 = [0.2, 0.3, ..., 0.8] (128 numbers)
        # Each number in these arrays represents a specific feature or characteristic of the face in the 128-dimensional space. 
        # The values reflect the importance or strength of each feature extracted from the face


        face_distances = face_recognition.face_distance(encoded_faces, face_encoding)
        # print(face_distances)
        # Find the index of the known face with the smallest distance to the current face
        best_match_index = np.argmin(face_distances)
        
        # By combining both methods (compare_faces and face distance), facial recognition systems can achieve higher accuracy
        # and reliability in identifying individuals based on their facial features.

        # If there is a match, mark attendance and display the name on the screen
        if matches[best_match_index]:
            # get the name of the matched person 
            name = class_names[best_match_index]
            print(name)
            # get the coordinates of the detected face in the current frame
            top, right, bottom, left = face_location
            # scale the coordinates up by a factor of 4 to match the original frame size
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            # draw a green rectangle around the detected face
            cv2.rectangle(frame, (left, top), (right, bottom), (153,153,7), 2)
            # draw a bluish rectangle at the bottom of the face rectangle to act as a background for the name text
            # cv2.FILLED: Thickness parameter specifying that the rectangle should be filled with the specified color
            #  rather than drawn as an outline.
            cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (106, 45, 87), cv2.FILLED)


            # write the name of the detected person on the frame
            # The bottom-left corner of the text string in the image. This is specified as a tuple (x, y) of two integers
            # .# The image on which to draw the text
            # The bottom-left corner of the text string in the image (slightly offset)
            # The font type
            # The font scale factor
            # The text string to be drawn
            # The color of the text in BGR format
            # The thickness of the lines used to draw the text

            
            cv2.putText(frame, name, (left + 4, bottom - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (195,229,232), 2)

            # mark the attendance of the detected person in the CSV file
            mark_attendance(name)

    # show the current frame with detected faces and names
    cv2.imshow('webcam', frame)

    # wait for a key press and check if it's the 'q' key to exit the loop
    # The expression (cv2.waitKey(1) & 0xFF) waits for a key press for 1 millisecond and 
    # ensures the returned key code is an 8-bit ASCII value by performing a bitwise AND with 0xFF
    # 0xFF: This is a hexadecimal representation of the number 255 (which is 11111111 in binary). Performing a
    # bitwise AND with 0xFF ensures that only the lower 8 bits of the value are considered. This is useful for compa
    # tibility across different systems and versions of OpenCV.
    if (cv2.waitKey(1) & 0xFF )== ord('q'):
        # exit the loop if the 'q' key is pressed
        break
        # Check if the window has been closed

    # Check if the window has been closed
    # 'webcam': The name of the window to check the property of
    # cv2.WND_PROP_VISIBLE: The property to check (whether the window is visible)
    # < 1: Condition to break the loop if the window is not visible (i.e., has been closed)
    if cv2.getWindowProperty('webcam', cv2.WND_PROP_VISIBLE) < 1:
        break


# Read CSV file with specified column names
# df = pd.read_csv('attendance.csv', names=['Name', 'Date', 'Time'])

# # Generate Excel file name with current date
# excel_file_name = f"attendance_{datetime.now().strftime('%Y-%m-%d')}.xlsx"

# # Convert DataFrame to Excel file with custom column names and dynamic file name
# df.to_excel(excel_file_name, index=False)
# del df


# Read CSV file with specified column names
new_df = pd.read_csv(attendance_csv, names=['Name', 'Date', 'Time'])

# Directory where the Excel file will be saved
directory = r'C:\Users\rhyth\OneDrive\Documents\py[1]\attendanceeee'

# Generate Excel file name with current date
excel_file_name = os.path.join(directory, f"attendance_{datetime.now().strftime('%Y-%m-%d')}.xlsx")


if os.path.exists(excel_file_name):
    # Read existing Excel file into DataFrame
    existing_df = pd.read_excel(excel_file_name)

    # Concatenate existing DataFrame and new DataFrame
    # Using ignore_index=True tells pd.concat() to ignore the original indices and 
    # create a new default integer index for the concatenated DataFrame.
    #default is ignore_index=False which keeps index as it is
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    #it will remove duplicated rows based on name column
    combined_df.drop_duplicates(subset=['Name'], keep='first', inplace=True)
else:
    # If Excel file doesn't exist, use just the new DataFrame
    combined_df = new_df

# Write combined DataFrame to Excel file
combined_df.to_excel(excel_file_name, index=False)




