import numpy as np


class Box(object):
    IS_ABSOLUTE_VALUE_BY_DEFAULT = False

    def __init__(self, x_min, y_min, width, height, is_absolute=IS_ABSOLUTE_VALUE_BY_DEFAULT):
        """
        Coordinates can be absolute or relative
            - Relative refers to between 0 and 1
            - Absolute refers to between 0 and image.shape[axis]
        """
        self.x_min = x_min
        self.y_min = y_min
        self.width = width
        self.height = height
        self.is_absolute = is_absolute

    def __eq__(self, other):
        return self.x_min == other.x_min \
               and self.y_min == other.y_min \
               and self.width == other.width \
               and self.height == other.height \
               and self.is_absolute == other.is_absolute

    def display(self):
        print(self.x_min, self.y_min, self.x_max, self.y_max)

    def copy(self):
        return Box(self.x_min, self.y_min, self.width, self.height, self.is_absolute)

    def shift(self, x, y):
        self.x_min += x
        self.y_min += y

    def assign(self, other):
        self.x_min = other.x_min
        self.y_min = other.y_min
        self.width = other.width
        self.height = other.height
        self.is_absolute = other.is_absolute

    def assign_tf_format(self, y_min, x_min, y_max, x_max):
        self.x_min = x_min
        self.y_min = y_min
        self.width = x_max - x_min
        self.height = y_max - y_min

    def extend_to(self, other):
        new_x_min = min(self.x_min, other.x_min)
        new_y_min = min(self.y_min, other.y_min)
        new_x_max = max(self.x_max, other.x_max)
        new_y_max = max(self.y_max, other.y_max)
        self.assign_tf_format(new_y_min, new_x_min, new_y_max, new_x_max)

    def cut_to_exclude(self, other):
        cut_box = self.get_excluding_box(other)
        self.assign(cut_box)

    def get_excluding_box(self, other):
        intersection_box = self.intersection(other)
        return self.get_cut_box_excluding_included_bordering_box(intersection_box)

    def get_cut_box_excluding_included_bordering_box(self, included_box):
        if included_box.area() == 0.:
            return self
        candidate_cut_boxes = [self.get_cut_box_changing_x_min(included_box),
                               self.get_cut_box_changing_y_min(included_box),
                               self.get_cut_box_changing_x_max(included_box),
                               self.get_cut_box_changing_y_max(included_box)]
        max_area_box = max(candidate_cut_boxes, key=lambda box: box.area())
        return max_area_box

    def get_cut_box_changing_x_min(self, other):
        new_x_min = max(self.x_min, other.x_max)
        return Box.from_tf_format([self.y_min, new_x_min, self.y_max, self.x_max])

    def get_cut_box_changing_y_min(self, other):
        new_y_min = max(self.y_min, other.y_max)
        return Box.from_tf_format([new_y_min, self.x_min, self.y_max, self.x_max])

    def get_cut_box_changing_x_max(self, other):
        new_x_max = min(self.x_max, other.x_min)
        return Box.from_tf_format([self.y_min, self.x_min, self.y_max, new_x_max])

    def get_cut_box_changing_y_max(self, other):
        new_y_max = min(self.y_max, other.y_min)
        return Box.from_tf_format([self.y_min, self.x_min, new_y_max, self.x_max])

    @staticmethod
    def multiple_union(boxes):
        if len(boxes) == 0:
            return Box.empty()
        union_of_boxes = boxes[0]
        for box in boxes[1:]:
            union_of_boxes = union_of_boxes.union(box)
        return union_of_boxes

    def union(self, other):
        assert self.is_absolute == other.is_absolute
        x_min, x_max = self.union_x_axis(other)
        y_min, y_max = self.union_y_axis(other)
        return Box(x_min, y_min, x_max - x_min, y_max - y_min, is_absolute=self.is_absolute)

    def union_x_axis(self, other):
        x_min = min(self.x_min, other.x_min)
        x_max = max(self.x_max, other.x_max)
        return x_min, x_max

    def union_y_axis(self, other):
        y_min = min(self.y_min, other.y_min)
        y_max = max(self.y_max, other.y_max)
        return y_min, y_max

    @staticmethod
    def empty(is_absolute=IS_ABSOLUTE_VALUE_BY_DEFAULT):
        return Box(0., 0., 0., 0., is_absolute)

    def is_empty(self):
        return self.area() == 0.

    def to_tf_format(self):
        return np.array([self.y_min, self.x_min, self.y_max, self.x_max])

    @staticmethod
    def from_tf_format(box, is_absolute=IS_ABSOLUTE_VALUE_BY_DEFAULT):
        [y_min, x_min, y_max, x_max] = box
        width = x_max - x_min
        height = y_max - y_min
        return Box(x_min, y_min, width, height, is_absolute=is_absolute)

    def to_pdfalto_format(self):
        return np.array([self.x_min, self.y_min, self.width, self.height])

    @staticmethod
    def from_tuple(tuple):
        x_min, y_min, width, height = tuple
        return Box(x_min, y_min, width, height)

    @property
    def x_center(self):
        return self.x_min + self.width / 2

    @property
    def y_center(self):
        return self.y_min + self.height / 2

    @property
    def x_max(self):
        return self.x_min + self.width

    @property
    def y_max(self):
        return self.y_min + self.height

    def area(self):
        return self.width * self.height

    def is_visually_after(self, other, tolerance=0.05):
        is_right = self.x_min > other.x_max - tolerance
        x_overlap = (other.x_max > self.x_min and self.x_min > other.x_min) \
                    or (self.x_max > other.x_min and other.x_min > self.x_min)
        is_below = self.y_min > other.y_max - tolerance
        return is_right or (x_overlap and is_below)

    @staticmethod
    def any_inclusion(box1, box2, margin=0.):
        return box1.includes(box2, margin) or box2.includes(box1, margin)

    def is_included_in(self, other, margin=0.):
        return other.includes(self, margin)

    def includes(self, other, margin=0.):
        allowed_outside_ratio = 1. + margin
        return other.area() <= self.intersection(other).area() * allowed_outside_ratio

    def intersection(self, other):
        assert self.is_absolute == other.is_absolute
        x_min_intersection, x_max_intersection = self.x_axis_intersection(other)
        y_min_intersection, y_max_intersection = self.y_axis_intersection(other)
        if x_min_intersection >= x_max_intersection or y_min_intersection >= y_max_intersection:
            return Box.empty(is_absolute=self.is_absolute)
        tf_format_box = [y_min_intersection, x_min_intersection, y_max_intersection, x_max_intersection]
        return Box.from_tf_format(tf_format_box, is_absolute=self.is_absolute)

    def x_axis_intersection(self, other):
        x_min_intersection = max(self.x_min, other.x_min)
        x_max_intersection = min(self.x_max, other.x_max)
        return x_min_intersection, x_max_intersection

    def y_axis_intersection(self, other):
        y_min_intersection = max(self.y_min, other.y_min)
        y_max_intersection = min(self.y_max, other.y_max)
        return y_min_intersection, y_max_intersection

    def any_intersection(self, boxes):
        return any(self.intersects(box) for box in boxes)

    def intersects(self, other):
        return self.x_axis_overlap(other) and self.y_axis_overlap(other)

    def x_axis_overlap(self, other):
        return self.x_min <= other.x_max and self.x_max >= other.x_min

    def y_axis_overlap(self, other):
        return self.y_min <= other.y_max and self.y_max >= other.y_min

    def do_overlap(self, other):
        return self.intersects(other)

    def intersection_over_area(self, other):
        area = self.area()
        return self.intersection(other).area() / area if area > 0. else 0.

    def intersection_over_area_x_axis(self, other):
        x_min_intersection, x_max_intersection = self.x_axis_intersection(other)
        x_axis_overlap_width = max(x_max_intersection - x_min_intersection, 0)
        return x_axis_overlap_width / self.width if self.width > 0. else 0.

    def intersection_over_union(self, other):
        intersection_area = self.intersection(other).area()
        union_area = self.area() + other.area() - intersection_area
        return intersection_area / union_area if union_area > 0. else 0.

    def is_below(self, other, tolerance_margin=0.):
        return (other.y_max - self.y_min) < tolerance_margin

    @staticmethod
    def from_absolute_coordinates(absolute_coordinates_box):
        [y_min, x_min, y_max, x_max] = absolute_coordinates_box
        width = x_max - x_min
        height = y_max - y_min
        return Box(x_min, y_min, width, height, is_absolute=True)

    @staticmethod
    def convert_boxes(boxes):
        return [Box.convert_box(box) for box in boxes]

    @staticmethod
    def convert_box(box):
        if isinstance(box, Box):
            return box
        else:
            return Box.from_tuple(box)

    def to_relative(self, image_width, image_height):
        if self.is_absolute:
            self.is_absolute = False
            self.x_min /= float(image_width)
            self.y_min /= float(image_height)
            self.width /= float(image_width)
            self.height /= float(image_height)

    def to_absolute(self, image_width, image_height):
        if not self.is_absolute:
            self.is_absolute = True
            self.x_min *= float(image_width)
            self.y_min *= float(image_height)
            self.width *= float(image_width)
            self.height *= float(image_height)

    def get_relative_copy(self, image_width, image_height):
        copy = self.copy()
        copy.to_relative(image_width, image_height)
        return copy

    def round(self):
        self.x_min = round(self.x_min)
        self.y_min = round(self.y_min)
        self.width = round(self.width)
        self.height = round(self.height)

    def get_absolute_copy(self, image_width, image_height):
        copy = self.copy()
        copy.to_absolute(image_width, image_height)
        return copy

    def l1_distance(self, other):
        left = other.x_max < self.x_min
        right = self.x_max < other.x_min
        bottom = other.y_max < self.y_min
        top = self.y_max < other.y_min
        if top and left:
            return Box.l1_points_distance(self.x_min, self.y_max, other.x_max, other.y_min)
        elif left and bottom:
            return Box.l1_points_distance(self.x_min, self.y_min, other.x_max, other.y_max)
        elif bottom and right:
            return Box.l1_points_distance(self.x_max, self.y_min, other.x_min, other.y_max)
        elif right and top:
            return Box.l1_points_distance(self.x_max, self.y_max, other.x_min, other.y_min)
        elif left:
            return self.x_min - other.x_max
        elif right:
            return other.x_min - self.x_max
        elif bottom:
            return self.y_min - other.y_max
        elif top:
            return other.y_min - self.y_max
        else:  # rectangles intersect
            return 0.

    @staticmethod
    def l1_points_distance(x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)

    def is_between(self, part_1, part_2):
        box_between_parts = part_1.get_box_in_the_middle(part_2)
        return self.intersects(box_between_parts) and \
               (not self.includes(box_between_parts))

    def get_box_in_the_middle(self, other):
        if self.intersects(other):
            return Box.empty(is_absolute=self.is_absolute)
        elif self.x_axis_overlap(other):
            box_in_the_middle_x_min, box_in_the_middle_x_max = self.x_axis_intersection(other)
            box_in_the_middle_y_min, box_in_the_middle_y_max = self.y_segment_between(other)
        elif self.y_axis_overlap(other):
            box_in_the_middle_x_min, box_in_the_middle_x_max = self.x_segment_between(other)
            box_in_the_middle_y_min, box_in_the_middle_y_max = self.y_axis_intersection(other)
        else:
            box_in_the_middle_x_min, box_in_the_middle_x_max = self.x_segment_between(other)
            box_in_the_middle_y_min, box_in_the_middle_y_max = self.y_segment_between(other)
        box_tf_format = [box_in_the_middle_y_min, box_in_the_middle_x_min,
                         box_in_the_middle_y_max, box_in_the_middle_x_max]
        return Box.from_tf_format(box_tf_format, is_absolute=self.is_absolute)

    def x_segment_between(self, other):
        box_in_the_middle_x_min = min(self.x_max, other.x_max)
        box_in_the_middle_x_max = max(self.x_min, other.x_min)
        return box_in_the_middle_x_min, box_in_the_middle_x_max

    def y_segment_between(self, other):
        box_in_the_middle_y_min = min(self.y_max, other.y_max)
        box_in_the_middle_y_max = max(self.y_min, other.y_min)
        return box_in_the_middle_y_min, box_in_the_middle_y_max

    def margin_based_distance(self, other, use_x):
        if self.intersects(other):
            return 0.
        elif not self.y_axis_overlap(other):
            self_below = self.y_min > other.y_max
            if self_below:
                return self.y_min - other.y_max
            else:  # then other is above
                return other.y_min - self.y_max
        else:  # then x_axis don't overlap
            if not use_x:
                return 1.
            self_right = self.x_min > other.x_max
            if self_right:
                return self.x_min - other.x_max
            else:  # then other is left
                return other.x_min - self.x_max

    def y_distance(self, other):
        if self.y_axis_overlap(other):
            return 0.
        else:
            return min(abs(self.y_min - other.y_max), abs(self.y_max - other.y_min))

    def x_distance(self, other):
        if self.x_axis_overlap(other):
            return 0.
        else:
            return min(abs(self.x_min - other.x_max), abs(self.x_max - other.x_min))