import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

y_m_per_px = 7.0 / 400 # meters per pixel in y dimension
x_m_per_px = 3.7 / 255 # meters per pixel in x dimension

def get_histogram(image):
    histogram = np.zeros(image.shape[0])
    histogram = np.sum(image, axis = 0)
    return histogram

def getLine(lines, deg):
    fit = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        z1 = np.polyfit((x1, x2), (y1, y2), deg)
        m = z1[0]
        c = z1[1]
        fit.append((m, c))
    avg = np.average(fit, axis = 0)
    # print(avg)
    y1 = 665
    y2 = int(470)
    x1 = int((y1-avg[1])/avg[0])
    x2 = int((y2-avg[1])/avg[0])
    return [x1, y1, x2, y2]

def drawLanes(yellow, white, deg):
        yellow_lines = cv.HoughLinesP(yellow, rho = 6, theta = np.pi/60, threshold = 25, minLineLength = 40, maxLineGap = 150)
        yellow_line = getLine(yellow_lines, deg)

        white_lines = cv.HoughLinesP(white, rho = 6, theta = np.pi/60, threshold = 25, minLineLength = 40, maxLineGap = 150)
        white_line = getLine(white_lines, deg)
        return [yellow_line, white_line]

def getWarpedImage(image_to_be_warped, a, b):
    h = image_to_be_warped.shape[0]
    w = image_to_be_warped.shape[1]
    M = cv.getPerspectiveTransform(a, b)
    warped_image = cv.warpPerspective(image_to_be_warped, M, [w, h], flags = cv.INTER_LINEAR)
    return warped_image

def getLanes(image_with_lanes):

    #################### Get Yellow Lane #####################
    hsv_img = cv.cvtColor(image_with_lanes, cv.COLOR_BGR2HSV)
    yellow_lower = np.array([10, 100, 100])
    yellow_upper = np.array([50, 255, 255])
    mask_yellow = cv.inRange(hsv_img, yellow_lower, yellow_upper)
    
    #################### Get White Lane #####################
    gray = cv.cvtColor(image_with_lanes, cv.COLOR_BGR2GRAY)
    ret, thres = cv.threshold(gray, 210, 255, cv.THRESH_BINARY)

    warped = mask_yellow + thres
    
    return warped

def getBaseWindow(hist):

        left_hist = hist[0:int(hist.shape[0]/2)]
        right_hist = hist[int(hist.shape[0]/2):]
        
        left_lane_base = np.argmax(left_hist)
        right_lane_base = np.argmax(right_hist)+int(hist.shape[0]/2)
        # x = np.arange(0, frame.shape[1])
        # y = np.array(hist)
        # plt.plot(x, y, color ="red")
        # # plt.savefig("results/q3_histogram")
        # plt.show()
        return [left_lane_base, right_lane_base]

def getCurvature(right_plot_y, left_y, left_x, right_y, right_x):
    y_eval = np.max(right_plot_y) 
    left_fit_cr = np.polyfit(left_y * y_m_per_px, left_x * x_m_per_px, 2)
    left_curvem = (((1 + (2*left_fit_cr[0]*y_eval*y_m_per_px + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0]))*100

    right_fit_cr = np.polyfit(right_y * y_m_per_px, right_x * x_m_per_px, 2)
    right_curvem = (((1 + (2*right_fit_cr[0]*y_eval*y_m_per_px + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0]))*100
    average_curvature = (right_curvem+left_curvem)/2

    return left_curvem, right_curvem, average_curvature
            
def getTurn(left_curvem, right_curvem):
    if(right_curvem < 0 or left_curvem < 0):
        turn = "Left"
    elif(right_curvem == 0 or left_curvem == 0):
        turn = "Straight"
    else:
        turn = "Right"
    return turn

def getActiveCells(image, lane_base, all_active, num_win = 15, win_width = 50):
    img = image.copy()
    left_active_cells = []
    right_active_cells = []
    h, w, _ = image.shape
    win_height = int(h / num_win)
    for i in range(num_win):
        win_y_up = h - (i+1)*win_height
        win_y_down = h - (i+0)*win_height
        # Left Sliding Window
        left_win_left_x = lane_base[0] - win_width
        left_win_right_x = lane_base[0] + win_width

        # Right Sliding Window
        right_win_left_x = lane_base[1] - win_width
        right_win_right_x = lane_base[1] + win_width

        left_valid_cells = ((all_active[1] > win_y_up) & (all_active[1] < win_y_down) & (all_active[0] < left_win_right_x) & (all_active[0] > left_win_left_x)).nonzero()[0]
        right_valid_cells = ((all_active[1] > win_y_up) & (all_active[1] < win_y_down) & (all_active[0] < right_win_right_x) & (all_active[0] > right_win_left_x)).nonzero()[0]

        cv.rectangle(img, (left_win_left_x, win_y_up), (left_win_right_x, win_y_down), (255, 255, 255), 1)
        cv.rectangle(img, (right_win_left_x, win_y_up), (right_win_right_x, win_y_down), (255, 255, 255), 1)

        left_active_cells.append(left_valid_cells)
        right_active_cells.append(right_valid_cells)

        if(len(left_valid_cells > 0)):
            lane_base[0] = int(np.mean(all_active[0][left_valid_cells]))
        if(len(right_valid_cells > 0)):
            lane_base[1] = int(np.mean(all_active[0][right_valid_cells]))

    left_active_cells = np.concatenate(left_active_cells)
    right_active_cells = np.concatenate(right_active_cells)
    return img, [left_active_cells, right_active_cells]

def getFittingandCurvature(all_active, active_cells, shape):

    left_x = all_active[0][active_cells[0]]
    left_y = all_active[1][active_cells[0]]
    left_fit = np.polyfit(left_y, left_x, 2)
    left_plot_x = np.linspace(0, shape[0]-1, shape[0])
    left_fit_y = left_fit[0]*left_plot_x**2 + left_fit[1]*left_plot_x + left_fit[2]
    left = np.array(np.vstack([left_fit_y, left_plot_x]).astype(np.int32).T)

    right_x = all_active[0][active_cells[1]]
    right_y = all_active[1][active_cells[1]]
    right_fit = np.polyfit(right_y, right_x, 2)
    right_plot_y = np.linspace(0, shape[0]-1, shape[0])
    right_plot_x = right_fit[0]*right_plot_y**2 + right_fit[1]*right_plot_y + right_fit[2]
    right = np.array(np.vstack([right_plot_x, right_plot_y]).astype(np.int32).T)

    pts_left = np.array([np.transpose(np.vstack([left_fit_y, right_plot_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_plot_x, right_plot_y])))])
    pts = np.hstack((pts_left, pts_right))
    left_curvem, right_curvem, average_curvature = getCurvature(right_plot_y, left_y, left_x, right_y, right_x)
    turn =  getTurn(left_curvem, right_curvem)
    return left, right, pts, turn, [left_curvem, right_curvem, average_curvature]

vid_name = "challenge"
cap = cv.VideoCapture("data/"+vid_name+'.mp4')
if (cap.isOpened()== False):
    print("Error opening video stream or file")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True: 
        h, w,_ = frame.shape

        ##################### WARP ROI ############################
        src = np.float32([[570, 450],
                        [750, 450],
                        [1110, 655],
                        [200, 655]])

        dst = np.float32([[0, 0], 
                        [w, 0], 
                        [w, h], 
                        [0, h]])
    
        vis_warped = getWarpedImage(frame, src, dst)

        #################### LANE THRESHOLDING #####################
        warped = getLanes(vis_warped)

        #################### HISTOGRAM ANALYSIS ####################
        hist = get_histogram(warped)
        lane_base = getBaseWindow(hist)

        #################### SLIDING WINDOW #######################
        all = warped.nonzero()
        all_active = [np.array(all[1]), np.array(all[0])]
        vis_warped, active_cells = getActiveCells(vis_warped, lane_base, all_active)

        #################### LINE FITTING AND CURVATURE ##########################
        left, right, pts, turn, curvatures = getFittingandCurvature(all_active, active_cells, vis_warped.shape)

        # cv.line(frame, (570, 450), (750, 450), (0, 0, 255), 5)
        # cv.line(frame, (1110, 655), (750, 450), (0, 0, 255), 5)
        # cv.line(frame, (1110, 655), (200, 655), (0, 0, 255), 5)
        # cv.line(frame, (570, 450), (200, 655), (0, 0, 255), 5)

        # cv.line(frame, (670, 450), (670, 655), (0, 0, 255), 5)

        ################ Visualization ##########################

        vis_warped = cv.polylines(vis_warped, [left], False, (255, 0, 0), 12)
        vis_warped = cv.polylines(vis_warped, [right], False, (255, 0, 0), 12)

        road = np.zeros_like(frame)
        cv.fillPoly(road, np.int_([pts]), (0,255, 0))

        cv.polylines(road, [left], False, (255, 0, 0), 12)
        cv.polylines(road, [right], False, (255, 0, 0), 12)

        road_warped = getWarpedImage(road, dst, src)
        result = cv.addWeighted(frame, 1.0, road_warped, 1.0, 0.0)

        cv.putText(result,('Expected turn = ' + str(turn)),(50,50), cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv.LINE_AA)
        cv.putText(result,('Left Curvature = ' + str(curvatures[0]) + str("m")),(50,90), cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv.LINE_AA)
        cv.putText(result,('Right Curvature = ' + str(curvatures[1]) + str("m")),(50,130), cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv.LINE_AA)
        cv.putText(result,('Average Curvature = ' + str(curvatures[2]) + str("m")),(50,170), cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv.LINE_AA)
        
        cv.imshow('(2)Warped Image', warped)
        cv.imshow('(3)Detected Points and Curve Fitting', vis_warped)
        cv.imshow('(4)Detected road', road)
        cv.imshow('(1)Result', result)

        if cv.waitKey(25) & 0xFF == ord('q'):
          break
    else: 
        break

cap.release()
cv.destroyAllWindows()