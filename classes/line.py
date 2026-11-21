
import numpy as np

class Line():
    '''Receive the characteristics of each line detection'''
    
    def __init__(self):
        
        # Was the line detected in the last iteration?
        self.detected = False
        
        # values of the last n fits of the line
        self.recent_fitted = []
        
        # Average x values of the fitted line over the last n iterations
        self.bestx = None
        
        # Polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        
        # Polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        
        # Radius of curvature of the line in some units
        self.radius_of_curvature = None
        
        # Distance in meters of vehicle center from the line
        self.line_base_pos = None
        
        # Difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        
        # x values for detected line pixels
        self.allx = None
        
        # y values for detected line pixels
        self.ally = None
