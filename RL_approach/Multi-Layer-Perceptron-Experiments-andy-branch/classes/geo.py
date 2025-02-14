import cv2
import math
import copy
import time
import numpy as np

start_time = time.time()

image_path = '/Users/yanda/Downloads/Plan Planning/test/3.png'
safety_radius = 0.5
alpha = 2
deltas = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # hyperparameter

def draw_dashed_line(img, start, end, color, thickness, dash_length=5):
    # 计算线段的长度
    length = np.linalg.norm(np.array(end) - np.array(start))
    # 计算线段上的单位向量q
    unit_vector = (np.array(end) - np.array(start)) / length
    num_dashes = int(length // dash_length)

    for i in range(num_dashes):
        start_point = start + unit_vector * (i * dash_length)
        end_point = start + unit_vector * ((i + 0.5) * dash_length)
        img = cv2.line(img, tuple(start_point.astype(int)), tuple(end_point.astype(int)), color, thickness)

    return img

def draw_dashed_circle(img, center, radius, color, thickness=1, dash_length=5):
    # 计算圆周的长度
    circumference = 2 * np.pi * radius
    num_dashes = int(circumference // dash_length)

    for i in range(num_dashes):
        start_angle = i * (360 / num_dashes)
        end_angle = (i + 0.5) * (360 / num_dashes)
        img = cv2.ellipse(img, center, (radius, radius), 0, start_angle, end_angle, color, thickness)

    return img

def draw_line_as_points(image, point1, point2, image_height, color=(255, 0, 255), thickness=10):
    """Draw a line as a series of points on the image."""
    x1, y1 = point1
    x2, y2 = point2
    
    # Generate a series of points between point1 and point2
    num_points = max(abs(x2 - x1), abs(y2 - y1))  # Number of points to generate
    x_points = np.linspace(x1, x2, num=int(num_points/30), dtype=int)
    y_points = np.linspace(y1, y2, num=int(num_points/30), dtype=int)

    for x, y in zip(x_points, y_points):
        cv2.circle(image, (x, image_height - y), thickness, color, -1)

def intersection_points(point1, slopes, point2, k):
    x1, y1 = point1
    x2, y2 = point2
    intersections = []

    for slope in slopes:
        if slope != k:
            x_intersect = (slope * x1 - k * x2 + y2 - y1) / (slope - k)
            y_intersect = slope * (x_intersect - x1) + y1
            intersections.append((x_intersect, y_intersect))
        else:
            intersections.append(None)  # Parallel lines don't intersect
    
    return intersections

def distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def line_circle_intersection(A, B, center, radius):
    """Calculate the intersection points of a line segment AB with a circle centered at center with radius."""
    Ax, Ay = A
    Bx, By = B
    Cx, Cy = center

    dx = Bx - Ax
    dy = By - Ay
    fx = Ax - Cx
    fy = Ay - Cy

    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = (fx * fx + fy * fy) - radius * radius

    if a == 0:  # AB is a single point
        if c <= 0:  # The point is on or inside the circle
            return [(Ax, Ay)]
        else:
            return []  # The point is outside the circle

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return []  # No intersection
    elif discriminant == 0:
        t = -b / (2 * a)
        return [(Ax + t * dx, Ay + t * dy)]  # One intersection (tangent)
    else:
        t1 = (-b + math.sqrt(discriminant)) / (2 * a)
        t2 = (-b - math.sqrt(discriminant)) / (2 * a)
        intersection1 = (Ax + t1 * dx, Ay + t1 * dy)
        intersection2 = (Ax + t2 * dx, Ay + t2 * dy)
        return [intersection1, intersection2]  # Two intersections

'''def perpendicular_intersection(waypoint, closest_points):
    """Calculate the intersection of the line through waypoint with slope k and the perpendicular line through closest_points."""
    WP_x, WP_y = waypoint
    CP_x, CP_y = closest_points
    if CP_x == WP_x:
        # closest_points line is vertical, perpendicular line is horizontal
        return (WP_x, CP_y)
    elif CP_y == WP_y:
        # closest_points line is horizontal, perpendicular line is vertical
        return (CP_x, WP_y)
    else:
        # Calculate slopes
        slope_CP = (CP_y - WP_y) / (CP_x - WP_x)
        perp_slope = -1 / slope_CP
        # Calculate intersection point
        x_intersect = (WP_y - CP_y + perp_slope * CP_x - slope_CP * WP_x) / (perp_slope - slope_CP)
        y_intersect = slope_CP * (x_intersect - WP_x) + WP_y
        return (x_intersect, y_intersect)'''

def perpendicular_intersection(waypoint, closest_points):
    x1, y1 = waypoint
    x2, y2 = closest_points
    
    x = (k1 * x1 - y1 - k2 * x2 + y2) / (k1 - k2)
    y = k1 * (x - x1) + y1
    
    return (x, y)

def determine_end_point(waypoint, closest_points, closest_rc, intersections):
    # Step 1: Calculate intersection point
    intersection = perpendicular_intersection(waypoint, closest_points)

    # Step 2: Calculate distance to closest_points
    distance_to_closest_points = distance(intersection, closest_points)
    
    # Step 3: Compare distance to closest_rc and return appropriate point
    if distance_to_closest_points > closest_rc:
        return intersection
    else:
        # Find the intersection point closest to the waypoint
        min_distance = float('inf')
        closest_intersection = None
        
        for point in intersections:
            dist_waypoint_to_point = distance(waypoint, point)
            dist_point_to_closest_points = distance(point, closest_points)
            
            if dist_waypoint_to_point < min_distance and dist_point_to_closest_points > closest_rc:
                min_distance = dist_waypoint_to_point
                closest_intersection = point
                
        return closest_intersection
    
def check_line_through_circles(A, B, circles, func, *func_args):
    for center, radius in circles:
        intersection_points = line_circle_intersection(A, B, center, radius)
        if len(intersection_points) == 2:
            return func(*func_args)
    return B  # One or zero intersections, return B point

# draw same length line
def draw_clipped_lines(image, start_point, end_points, clip_length, color, thickness):
    new_end_points = []

    for end_point in end_points:
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        length = np.sqrt(dx**2 + dy**2)
        scale = clip_length / length if length > 0 else 0
        new_end_x = int(start_point[0] + scale * dx)
        new_end_y = int(start_point[1] + scale * dy)
        new_end_point = (new_end_x, new_end_y)
        cv2.line(image, start_point, new_end_point, color, thickness)
        new_end_points.append(new_end_point)

    return new_end_points

# find next closest points
def closest_two_points(origin, points, R_c, intersections):
    # 创建points、R_c和intersections的副本
    points_copy = copy.deepcopy(points)
    R_c_copy = copy.deepcopy(R_c)
    intersections_copy = copy.deepcopy(intersections)

     # 如果points列表只有一个值，返回updated_points, updated_rc, updated_intersections
    if len(points) == 1:
        return points_copy, R_c_copy, intersections_copy, points_copy, R_c_copy, intersections_copy
    
    # 将点、R_c和intersections值打包在一起
    point_rc_int_pairs = list(zip(points_copy, R_c_copy, intersections_copy))
    
    # 初始化最小距离和最近点坐标
    min_distances = [(float('inf'), None), (float('inf'), None)]  # (distance, (point, rc, intersection))

    # 遍历点集合
    for point, rc, intersection in point_rc_int_pairs:
        # 计算当前点到原点的距离
        distance = math.sqrt((point[0] - origin[0])**2 + (point[1] - origin[1])**2)
        
        # 如果当前距离比已知的最小距离还小，更新最小距离和最近点坐标
        if distance < min_distances[0][0]:
            min_distances[1] = min_distances[0]
            min_distances[0] = (distance, (point, rc, intersection))
        elif distance < min_distances[1][0]:
            min_distances[1] = (distance, (point, rc, intersection))

    # 提取最近的两个点及其对应的R_c和intersections值
    closest_points_rc_int = [min_distances[0][1], min_distances[1][1]]
    closest_points = [closest_points_rc_int[0][0], closest_points_rc_int[1][0]]
    closest_rc = [closest_points_rc_int[0][1], closest_points_rc_int[1][1]]
    closest_intersections = [closest_points_rc_int[0][2], closest_points_rc_int[1][2]]

    # 从副本列表中剔除最近的一个点及其对应的R_c值和intersections值
    points_copy.remove(closest_points[0])
    R_c_copy.remove(closest_rc[0])
    intersections_copy.remove(closest_intersections[0])

    return closest_points, closest_rc, closest_intersections, points_copy, R_c_copy, intersections_copy

# find end point of a line
def end_point(x, y, slopes):
    delta_x = 100 # tangents length
    end_points = []
    for slope in slopes:
        delta_y = slope * delta_x
        end_point = (int(x + delta_x), int(image_height - y - delta_y))
        end_points.append(end_point)
    
    return end_points

# find tangent
def find_m(x1, y1, h, k, r):
    # Calculate the distance from the external point to the center of the circle
    d = np.sqrt((x1 - h)**2 + (y1 - k)**2)
    
    # Calculate the angle of the line connecting the point to the center
    theta = np.arctan2(y1 - k, x1 - h)
    
    # Calculate the angles of the tangent lines
    alpha1 = theta + np.arcsin(r / d)
    alpha2 = theta - np.arcsin(r / d)
    
    # Calculate the slopes of the tangent lines
    m1 = np.tan(alpha1)
    m2 = np.tan(alpha2)

    result = [m1, m2]
    
    return result

# get rid of some obstacles
def pre_process(SP_x, SP_y, EP_x, EP_y, p_cx, p_cy, R_c, alpha):
    m0 = (EP_y - SP_y) / (EP_x - SP_x)
    e0 = SP_y - m0 * SP_x
    
    filtered_intersections = []
    filtered_intercepts = []
    filtered_p_c = []
    filtered_R_c = []

    for i, (x, y) in enumerate(zip(p_cx, p_cy)):
        # 添加限制条件
        if SP_x < EP_x and SP_y < EP_y:
            if x < SP_x or y < SP_y:
                continue
        elif SP_x < EP_x and SP_y > EP_y:
            if x < SP_x or y > SP_y:
                continue
        elif SP_x > EP_x and SP_y > EP_y:
            if x > SP_x or y > SP_y:
                continue
        elif SP_x > EP_x and SP_y < EP_y:
            if x > SP_x or y < SP_y:
                continue
        
        mp = -1 / m0
        e = y - mp * x
        
        x_int = (e - e0) / (m0 - mp)
        y_int = m0 * x_int + e0
        
        distance = math.sqrt((x_int - x) ** 2 + (y_int - y) ** 2)
        
        if alpha * R_c[i] > distance:
            filtered_intersections.append([x_int, y_int])
            filtered_intercepts.append(e)
            filtered_p_c.append((x, y))
            filtered_R_c.append(R_c[i])
    
    return filtered_intersections, filtered_intercepts, filtered_p_c, filtered_R_c, m0, mp

# draw the path
image = cv2.imread(image_path)
image_height, image_width = image.shape[:2]

# Convert image from BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold for color
_, threshold_white = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
red_lower1 = np.array([0, 120, 70])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 120, 70])
red_upper2 = np.array([180, 255, 255])
yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([30, 255, 255])

# Detect color
contours_white, _ = cv2.findContours(threshold_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
mask_red = cv2.bitwise_or(mask_red1, mask_red2)
contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Center of area
centers_radius_white = []
centers_red = []
centers_yellow = []

for contour in contours_white:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 3)
    
    # Calculate center coordinates, adjuste to bottom-left origin
    center_x = x + w / 2
    center_y = image_height - (y + h / 2)
    
    radius = int(safety_radius * np.sqrt(w**2 + h**2))
    
    #cv2.circle(image, (int(center_x), image_height - int(center_y)), radius, (0, 255, 0), 3)
    
    centers_radius_white.append(((center_x, center_y), radius))

for contour in contours_red:
    x, y, w, h = cv2.boundingRect(contour)
    #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    center_x = x + w / 2
    center_y = image_height - (y + h / 2)
    
    centers_red.append((center_x, center_y))

for contour in contours_yellow:
    x, y, w, h = cv2.boundingRect(contour)
    #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)

    center_x = x + w / 2
    center_y = image_height - (y + h / 2)
    
    centers_yellow.append((center_x, center_y))

SP_x, SP_y = centers_red[0][0], centers_red[0][1]
EP_x, EP_y = centers_yellow[0][0], centers_yellow[0][1]
p_cx = [point[0][0] for point in centers_radius_white]
p_cy = [point[0][1] for point in centers_radius_white]
R_c = [point[1] for point in centers_radius_white]

# remove far obstacles
intersections, intercepts, p_c, R_c, m0, mp = pre_process(SP_x, SP_y, EP_x, EP_y, p_cx, p_cy, R_c, alpha)

# pack intersection coordinates and related data into a list of tuples
packed_data = list(zip(intersections, intercepts, p_c, R_c))

# sort the tuple list based on the abscissa of the intersection point
sorted_packed_data = sorted(packed_data, key=lambda x: x[0][0])

# unpack the data and extract the sorted results
sorted_intersections = [item[0] for item in sorted_packed_data]
sorted_intercepts = [item[1] for item in sorted_packed_data]
sorted_p_c = [item[2] for item in sorted_packed_data]
sorted_R_c = [item[3] for item in sorted_packed_data]
 
# selected obstacles' radius including hyperparameter
updated_R_c = [r + d for r, d in zip(sorted_R_c, deltas)]
for (x, y), radius in zip(sorted_p_c, sorted_R_c):
    cv2.circle(image, (int(x), image_height - int(y)), int(radius), (0, 255, 0), 3)
# circle with hyperparameter
for (x, y), radius in zip(sorted_p_c, updated_R_c):
    draw_dashed_circle(image, (int(x), image_height - int(y)), int(radius), (0, 255, 0), 3, 50)

# predefined
k1 = (EP_y - SP_y) / (EP_x - SP_x)
k2 = -1 / k1
waypoint0 = [SP_x, SP_y]
n = len(updated_R_c)

# initialize
updated_points = sorted_p_c
updated_rc = updated_R_c
updated_intersections = sorted_intersections
waypoints = []
waypoints.append(waypoint0)
total_length = 0

for i in range(n):
    waypoint_current = waypoints[-1]  # 获取当前的waypoint
  
    if i == 0:
        circles = list(zip(sorted_p_c, updated_R_c))
    else:
        circles = list(zip(updated_points, updated_rc))

    closest_points, closest_rc, closest_intersections, updated_points, updated_rc, updated_intersections = closest_two_points(
        waypoint_current, updated_points, updated_rc, updated_intersections)

    if len(closest_points) == 1:
        result = find_m(waypoint_current[0], waypoint_current[1], closest_points[0][0], closest_points[0][1], closest_rc[0])
    else:
        result1 = find_m(waypoint_current[0], waypoint_current[1], closest_points[0][0], closest_points[0][1], closest_rc[0])
        result2 = find_m(waypoint_current[0], waypoint_current[1], closest_points[1][0], closest_points[1][1], closest_rc[1])
        result = result1 + result2

    intersections = intersection_points(waypoint_current, result, closest_points[0], k2)
    waypoint_next = check_line_through_circles(waypoint_current, [EP_x, EP_y], circles, determine_end_point, waypoint_current, closest_points[0], closest_rc[0], intersections)
    
    # draw path
    draw_line_as_points(image, waypoint_current, waypoint_next, image_height)

    total_length += distance(waypoint_current, waypoint_next)

    waypoints.append(waypoint_next)  # 添加新的waypoint到列表

    if len(intersections) == 2:
        waypoint_final = [EP_x, EP_y]
        draw_line_as_points(image, waypoint_next, waypoint_final, image_height)
        waypoints.append(waypoint_final)  # 添加最终的waypoint
        break  # 如果达到了终点，可以退出循环

    if waypoint_next == [EP_x, EP_y]:
        break

'''    end_points = end_point(waypoint_current[0], waypoint_current[1], result)
    new_end_points = draw_clipped_lines(image, (int(waypoint_current[0]), image_height - int(waypoint_current[1])), end_points, 500, (255, 255, 255), 2)'''
'''
    if i == 4:
        end_points = end_point(waypoint_current[0], waypoint_current[1], result)
        new_end_points = draw_clipped_lines(image, (int(waypoint_current[0]), image_height - int(waypoint_current[1])), end_points, 500, (255, 255, 255), 2)
'''
'''# selected obstacles' center
for (pcx, pcy) in p_c:
    cv2.circle(image, (int(pcx), image_height - int(pcy)), radius=10, color=(255, 0, 0), thickness=-1)

# selected obstacles' perpendicular lines
for (pc, intersection) in zip(sorted_p_c, sorted_intersections):
    x1, y1 = int(pc[0]), int(pc[1])
    x2, y2 = int(intersection[0]), int(intersection[1])
    draw_dashed_line(image, (x1, image_height - y1), (x2, image_height - y2), (255, 0, 0), 5, 25)

# optimal path
draw_dashed_line(image, (int(SP_x), image_height - int(SP_y)), (int(EP_x), image_height - int(EP_y)), (0, 128, 255), 5, 100)'''

print(f"total length: {total_length:.2f}")

cv2.imshow('geo', image)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Code running time: {elapsed_time:.2f} s")
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite('/Users/yanda/Downloads/Plan Planning/test/result3_2.png', image) 