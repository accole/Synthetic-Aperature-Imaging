import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage import data
from skimage.feature import match_template


#function to read in video and convert to gray frames
def readVideo():
    #import the video capture
    vid = cv2.VideoCapture("video.mp4")

    Grayframes = []
    frames = []
    while(True):
        #capture frame by frame
        ret, frame = vid.read()
        
        if ret == True:
            #convert the frame from RGB to gray
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #print("Gray:", np.shape(gray))
            frames.append(frame)
            Grayframes.append(gray)
        else:
            break

    #when done, release the capture
    vid.release()
    cv2.destroyAllWindows()

    #save the first gray frame for the hw
    cv2.imwrite('AC_frame1_grayscale.png', Grayframes[0])

    return (frames, Grayframes)



def plotPart3():
    delta = 1
    f = 1
    z2 = 1
    z1 = 1
    factor = z2 - z1
    #numerator = f * factor
    denominator = z2 * z1
    x = []
    y = []
    factor_list = [1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]
    for new_factor in factor_list:
        x.append(new_factor)
        numerator = f * new_factor
        #print(numerator)
        z2 = new_factor + 1
        denominator = z2
        denominator = float(denominator)
        #print(denominator)
        w = numerator / denominator
        print(w)
        y.append(w)
    _,ax3 = plt.subplots(1)
    ax3.set_ylabel('Width of the Blur Kernel')
    ax3.set_xlabel('Value of |z2 - z1|')
    ax3.set_title('Width versus Difference of Depth Planes')
    ax3.plot(x,y)
    plt.savefig('3-3.png')

    x = []
    y = []
    for new_f in factor_list:
        x.append(new_f)
        numerator = new_f
        z2 = 2
        denominator = 2
        denominator = float(denominator)
        w = numerator / denominator
        y.append(w)
    _,ax3 = plt.subplots(1)
    ax3.set_ylabel('Width of the Blur Kernel')
    ax3.set_xlabel('Value of f')
    ax3.set_title('Width versus Difference of Depth Planes')
    ax3.plot(x,y)
    plt.savefig('3-4.png')

    
    
def templateFrames(gframes, frames):
    #crop the first frame to include only the template
    #import the .png file to search all other frames for the template
    template = cv2.imread('Bear Template.png')
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    #define template matching method
    method = cv2.TM_CCOEFF_NORMED

    #get the dimensions of the template
    #shape is 2D since it is black and white
    height, width = template.shape

    #create a list to store the template locations for each frame
    template_locs = []

    #search for the template in all other frames
    i = 0
    for f in range(0, len(gframes)):
        #apply the matching template
        match = cv2.matchTemplate(gframes[f], template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)

        #find the corners of the template rectangle
        top_left = max_loc
        bottom_right = (top_left[0] + width, top_left[1] + height)
        #store the template location
        template_locs.append([top_left, bottom_right])

        #print out the tracked template for the first frame
        if i == 0:
            #plot frame 1 with template rectangle
            _,ax = plt.subplots(1)
            rect = plt.Rectangle((top_left[0], top_left[1]), width, height, edgecolor='r', facecolor='none')
            frame1 = frames[0]
            #changes it from RGB to BGR
            ax.imshow(frame1[:,:,::-1])
            ax.set_axis_off()
            ax.add_patch(rect)
            plt.savefig('frame1_with_template.png')
            #plot cross normalization with pixel axis'
            _,ax2 = plt.subplots(1)
            ax2.imshow(match, cmap='gray')
            ax2.set_title('Cross Normalization')
            ax2.xaxis.set_ticks_position('top')
            ax2.set_ylabel('Pixels')
            plt.savefig('Cross_Normalization.png')
            #add one so it doesn't save a figure for every frame
            i = i + 1

    return template_locs


def pixelShift(template_locs):
    #run through the template pixel locations and plot them to see pixel shift
    x = []
    y = []
    for location in template_locs:
        x.append(location[0][0])
        y.append(location[0][1])

    _,ax3 = plt.subplots(1)
    ax3.set_ylabel('Y Pixel Shift')
    ax3.set_xlabel('X Pixel Shift')
    ax3.set_title('Pixel Shift - Top Left Pixel of Template')
    ax3.plot(x,y)
    plt.savefig('Pixel_Shift.png')



def synthesizePixels(frames, template_locs):
    #store the location of the template from the first image
    frame1 = template_locs[0]
    x_1 = frame1[0][0]
    y_1 = frame1[0][1]

    #calculate difference vector for each frame
    diff_vecs = []

    for loc in template_locs:
        x = loc[0][0]
        y = loc[0][1]
        diff_vecs.append([x - x_1, y - y_1])
   
    #for each frame, apply the pixel shift to each pixel
    i = 0
    red_image_list = []
    green_image_list = []
    blue_image_list = []
    
    for f in frames:
        width = np.shape(f)[1]
        height = np.shape(f)[0]
        red = np.zeros((height,width), dtype=np.uint8)
        blue = np.zeros((height,width), dtype=np.uint8)
        green = np.zeros((height,width), dtype=np.uint8)
        
        row = 0
        for RGBframe in f:
            #print(np.shape(RGBframe)) 
            column = 0
            for RGB in RGBframe:
                red[row][column] = RGB[0]
                blue[row][column] = RGB[1]
                green[row][column] = RGB[2] 
                column = column + 1
            row = row + 1
        
        #create affine transformation matrix
        M = np.matrix('1 0 0 ;0 1 0')
        diff_vec = diff_vecs[i]
        M[0,2] = (-1 * diff_vec[0])
        M[1,2] = (-1 * diff_vec[1])
        M = M.astype(np.float32)
        
        dsize = (width, height)
        
        transformed_image = []
        
        red = cv2.warpAffine(red,M,dsize)
        blue = cv2.warpAffine(blue,M,dsize) 
        green = cv2.warpAffine(green,M,dsize)
        
        red = np.array(red, dtype=np.int32)
        blue = np.array(blue, dtype=np.int32)
        green = np.array(green, dtype=np.int32)
        
        red_image_list.append(red)
        green_image_list.append(green)
        blue_image_list.append(blue)
        
        #print(i)
        i = i + 1
    
    #initialize an empty frame to fill with final image
    length = i
    final = np.zeros((height,width,3), dtype=np.int32)
    final_red = np.zeros((height,width), dtype=np.int32)
    final_blue = np.zeros((height,width), dtype=np.int32)
    final_green = np.zeros((height,width), dtype=np.int32)
    
    #each frame is 1920 x 1020 pixels
    for width_pixel in range(0,1920):
        for height_pixel in range(0,1020):
            #sum up the RGB values for every frame at shifted location
            for image in red_image_list:
                final_red[width_pixel][height_pixel] = final_red[width_pixel][height_pixel] + image[width_pixel][height_pixel]
            for image in blue_image_list:
                final_blue[width_pixel][height_pixel] = final_blue[width_pixel][height_pixel] + image[width_pixel][height_pixel]
            for image in green_image_list:
                final_green[width_pixel][height_pixel] = final_green[width_pixel][height_pixel] + image[width_pixel][height_pixel]
            
            #average the values for RGB
            final_red[width_pixel][height_pixel] = final_red[width_pixel][height_pixel]/length
            final_blue[width_pixel][height_pixel] = final_blue[width_pixel][height_pixel]/length
            final_green[width_pixel][height_pixel] = final_green[width_pixel][height_pixel]/length
            
            final_element = np.zeros(3, dtype=np.int32)
            
            #create the final image
            #Red
            final_element[0] = final_red[width_pixel][height_pixel]
            final_element[0] = final_element[0].astype(np.uint8)
            #Blue
            final_element[1] = final_blue[width_pixel][height_pixel]
            final_element[1] = final_element[1].astype(np.uint8)
            #Green
            final_element[2] = final_green[width_pixel][height_pixel]
            final_element[2] = final_element[2].astype(np.uint8)
            
            final[width_pixel][height_pixel][0] = final_element[0]
            final[width_pixel][height_pixel][1] = final_element[1]
            final[width_pixel][height_pixel][2] = final_element[2]

    cv2.imwrite('Final_Image.png', final)
    
    return final



def main():
    #read in the video file and convert to grayscale
    frames, gframes = readVideo()
    plotPart3()
    
    #create a template and locate it in all frames
    #through cross normalization
    template_locs = templateFrames(gframes)
    
    #plot the pixel shift using template locations
    pixelShift(template_locs)
    
    #synthesize the pixel shifts
    final_img = synthesizePixels(frames, template_locs)




if __name__ == '__main__':
    main()
