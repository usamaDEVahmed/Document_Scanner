import cv2
import numpy as np
from PIL import Image

class DocumentScanner():

    def __init__(self, path, IMG_FLAG=False, VID_FLAG=False, SAVE_PDF=False, path_to_save=None):
        self.capture = None
        self.IMG_FLAG = IMG_FLAG
        self.VID_FLAG = VID_FLAG
        self.SAVE_PDF = SAVE_PDF
        self.path_to_save = path_to_save

        if self.IMG_FLAG and self.VID_FLAG:
            print('[-] Error: Image and Video Flag cannot be True at the same time.')
        elif self.IMG_FLAG:
            self.capture = cv2.imread(path)
        elif self.VID_FLAG:
            self.capture = cv2.VideoCapture(path)
    
    def execute(self):
        if self.IMG_FLAG and self.capture:
            self.execute_image()
        elif self.VID_FLAG and self.capture:
            self.execute_video()
        else:
            print('[-] Error: Source not detected. Remember to use a valid Flag with the source.')

    def execute_image(self):
        preprocessed = self.preprocess(self.capture)
        corner_points = self.get_cotours(preprocessed)
        reordered_corners = self.reorder_points(corner_points)
        warpped = self.warp_perspective(reordered_corners, self.capture)
        result = self.crop_sides(warpped)

        if self.SAVE_PDF:
            self.save_as_pdf(result)

        cv2.imshow('Result', result)
        cv2.waitKey(0)

    def execute_video(self):

        while True:
            return_flag, frame = self.capture.read()
            if return_flag:
                preprocessed = self.preprocess(frame)
                corner_points = self.get_cotours(preprocessed)
                reordered_corners = self.reorder_points(corner_points)
                warpped = self.warp_perspective(reordered_corners, frame)
                result = self.crop_sides(warpped)
                cv2.imshow('Result', result)

            else:
                print('[-] Error: Could not read Video.')
                break
 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if self.SAVE_PDF:
            self.save_as_pdf(result)
        self.capture.release()
        cv2.destroyAllWindows()


    def preprocess(self, image):
        final = None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 1)
        canny = cv2.Canny(blur, 50, 200)
        kernel = np.ones((5, 5), dtype=np.uint8)
        dilated = cv2.dilate(canny, kernel, iterations=2)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        # cv2.imshow('Final', self.capture)
        # cv2.waitKey(0)
        return eroded
    
    def get_cotours(self, image):
        required_corners = np.array([])

        # image = self.preprocess()
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for each in contours:
            area = cv2.contourArea(each)

            if area > 5000:
                capture_copy = np.copy(self.capture)
                perimeter = cv2.arcLength(each, True)
                
                corner_pts = cv2.approxPolyDP(each, 0.02*perimeter, True)
                total_corners = len(corner_pts)

                if total_corners == 4:
                    required_corners = corner_pts

                    # cv2.drawContours(self.capture, required_corners, -1, (0, 255, 0), thickness=20)
                    # cv2.imshow('asdf', self.capture)
                    # cv2.waitKey(0)
            
        return required_corners


    def warp_perspective(self, corners, image):
        # corners = self.get_cotours()
        # corners = self.reorder_points(corners)

        width = 480
        height = 640

        pts_1 = np.float32(corners)
        pts_2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

        matrix = cv2.getPerspectiveTransform(pts_1, pts_2)
        output = cv2.warpPerspective(image, matrix, (width, height))
        resized = cv2.resize(output, (480, 640))
        
        return resized


    def reorder_points(self, points):
        arranged_points = np.zeros_like(points, dtype=np.int32)
        points = np.reshape(points, (4, 2))
        addition = points.sum(axis=1)
        # print(points)
        arranged_points[0] = points[np.argmin(addition)]
        arranged_points[-1] = points[np.argmax(addition)]

        difference = np.diff(points, axis=1)
        # print(difference)
        arranged_points[1] = points[np.argmin(difference)]
        arranged_points[2] = points[np.argmax(difference)]
        
        return arranged_points

    def crop_sides(self, image):
        return image[20:image.shape[0]-20, 20:image.shape[1]-20]

    def save_as_pdf(self, image):
        picture = Image.fromarray(image)
        picture = picture.convert('RGB')
        try:
            if self.path_to_save:
                picture.save(str(path))
            else:
                picture.save('****.pdf')
        except ValueError:
            print('[-] Error. Remember to use a valid path and extension with filename.')

    


if __name__ == "__main__":
    doc = DocumentScanner('vid.mp4', IMG_FLAG=True, SAVE_PDF=True)
    doc.execute()
