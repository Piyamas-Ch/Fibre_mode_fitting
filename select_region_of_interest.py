import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from astropy.io import fits
import tkinter as tk
from tkinter import messagebox

class select_roi_area:
    def __init__(self, fit_file):
        # Center of x and y of region of interest
        self.x_c = 0
        self.y_c = 0
        
        # Load FITS data
        with fits.open(fit_file) as hdul:
            self.data = hdul[0].data.astype(float)
            
    def onselect(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self.coords['rect'] = (min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1))
        plt.close(self.fig)  # close after selection

    def select_area(self):
        # Show the image
        self.fig, ax = plt.subplots()
        ax.imshow(self.data, cmap='gray', origin='upper')
        ax.set_title('Drag to select ROI')

        # Rectangle selector
        self.coords = {}
        selector = RectangleSelector(ax, self.onselect, useblit=True,
                                    button=[1],  # left click only
                                    minspanx=5, minspany=5, spancoords='pixels',
                                    interactive=False)
        plt.show()

        # Crop the image
        if 'rect' in self.coords:
            x, y, w, h = self.coords['rect']
            x_c = int(x + (w/2))
            y_c = int(y + (h/2))
            cropped = self.data[y:y+h, x:x+w]

            # Show cropped image
            print("===========Cropped region===========")
            print(f"x={x}, y={y}, x_c={x_c}, y_c={y_c}, width={w}, height={h}")
            print("====================================")
            plt.figure()
            plt.imshow(cropped, cmap='gray', origin='upper')
            plt.title(f"Cropped region\n(x={x}, y={y}, x_c={x_c}, y_c={y_c}, width={w}, height={h})")
            plt.show(block=False)

            # Update value of center of x and y
            self.x_c = x_c
            self.y_c = y_c

def select_region_of_interest(fit_file):
    # Create object for region of interest selection and load fit file
    roi_obj = select_roi_area(fit_file)

    execute_flag = 1
    while execute_flag == 1:
        # User select region of interest area
        roi_obj.select_area()

        if roi_obj.x_c != 0 and roi_obj.y_c != 0:
            # Create question dialog
            root = tk.Tk() # Create a root window
            root.withdraw() # Hide the main window
            response = messagebox.askquestion("Question", "Do you want to select again?")
            if response == 'yes':
                # If user want to select region of interest again, show image for user to select again
                execute_flag = 1
            else:
                # If user dose not want to select region of interest again, stop selection
                execute_flag = 0
            root.destroy() # Close question dialog
        else:
            execute_flag = 0

        plt.close('all') # Close all plot window

    return roi_obj.x_c, roi_obj.y_c
