from utility import gaussian_blur, grayscale, canny, region_of_interest, \
  hough_lines, weighted_img
from functools import partial
from copy import deepcopy


class LineDetector():
  def __init__(self,
               kernel_size,
               low_threshold,
               high_threshold,
               mask_poly,
               rho,
               theta,
               threshold,
               min_line_len,
               max_line_gap):

    self.k_size = kernel_size
    self.l_thred = low_threshold
    self.h_thred = high_threshold
    self.vertices = mask_poly
    self.rho = rho
    self.theta = theta
    self.thred = threshold
    self.min_l_len = min_line_len
    self.max_l_gap = max_line_gap

    self.gaussian_blur = partial(
      gaussian_blur,
      kernel_size=self.k_size
    )

    self.canny = partial(
      canny,
      low_threshold=self.l_thred,
      high_threshold=self.h_thred
    )

    self.region_of_interest = partial(
      region_of_interest,
      vertices=self.vertices
    )

    self.hough_lines = partial(
      hough_lines,
      rho=self.rho,
      theta=self.theta,
      threshold=self.thred,
      min_line_len=self.min_l_len,
      max_line_gap=self.max_l_gap
    )

  def get_single_detector(self):
    return lambda img: weighted_img(
      self.hough_lines(
        self.region_of_interest(
          self.canny(
            self.gaussian_blur(
              grayscale(
                deepcopy(
                  img
                )
              )
            )
          )
        )
      ),
      deepcopy(img)
    )

  def __call__(self, imgGen):
    return map(
      lambda imgPair: weighted_img(imgPair[0], imgPair[1]),
      zip(
        map(
          self.hough_lines,
          map(
            self.region_of_interest,
            map(
              self.canny,
              map(
                self.gaussian_blur,
                map(grayscale,
                    deepcopy(imgGen))
              )
            )
          )
        ),
        deepcopy(imgGen)
      )
    )
