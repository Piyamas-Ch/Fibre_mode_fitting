import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from astropy.io import fits
import tkinter as tk
from tkinter import messagebox

class select_roi_area:
    def __init__(self, fit_file):
        # Center of x and y of region of interest
        self.x = 0
        self.y = 0
        
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
        fig, ax = plt.subplots()
        self.fig = fig
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

            # Show cropped image to user to select center of fibre
            fig_crop, ax_crop = plt.subplots()
            self.fig_crop = fig_crop
            self.ax_crop = ax_crop
            ax_crop.imshow(cropped, cmap='gray', origin='upper')
            ax_crop.set_title("Click to select center of fibre")
            plt.show(block=False)

            # Update value of x and y
            self.x = x
            self.y = y
        else:
            self.x = 0
            self.y = 0

    def select_fibre_center(self):
        # initial crosshair lines
        vline = self.ax_crop.axvline(x=0, color='w', linestyle=':', linewidth=1)
        hline = self.ax_crop.axhline(y=0, color='w', linestyle=':', linewidth=1)

        # optional marker at intersection
        marker, = self.ax_crop.plot([0], [0], marker='+', color='r')
        # Optional marker for last clicked position
        click_marker, = self.ax_crop.plot([], [], 'go', markersize=8)  # green dot

        def on_move(event):
            if not event.inaxes:
                return
            
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                return
            
            vline.set_xdata([x, x])
            vline.set_ydata([0, self.data.shape[0]-1])
            hline.set_xdata([0, self.data.shape[1]-1])
            hline.set_ydata([y, y])
            marker.set_data([x], [y])
            self.fig_crop.canvas.draw_idle()

        def on_click(event):
            if not event.inaxes:
                return
            x, y = event.xdata, event.ydata
            click_marker.set_data([x], [y])
            self.fig_crop.canvas.draw_idle()

        self.fig_crop.canvas.mpl_connect('motion_notify_event', on_move)
        self.fig_crop.canvas.mpl_connect('button_press_event', on_click)

        points = plt.ginput(1)  # waits until user clicks once

        return points

def select_region_of_interest(fit_file, r):
    # Create object for region of interest selection and load fit file
    roi_obj = select_roi_area(fit_file)

    execute_flag = 1
    while execute_flag == 1:
        # User select region of interest area
        roi_obj.select_area()

        if roi_obj.x != 0 and roi_obj.y != 0:
            # User select point of fibre center
            points = roi_obj.select_fibre_center()

            if points != []:
                # Crop image again based on point of fibre center and radius
                x_c = roi_obj.x + points[0][0]
                y_c = roi_obj.y + points[0][1]
                x = int(x_c - r - 20)
                y = int(y_c - r - 20)
                w = int((2*r) + 40)
                h = int((2*r) + 40)
                cropped = roi_obj.data[y:y+h, x:x+w]
                # Show cropped image
                plt.close('all') # Close all plot window
                print("===========Cropped region===========")
                print(f"x={x}, y={y}, x_c={int(x_c)}, y_c={int(y_c)}, width={w}, height={h}")
                print("====================================")
                plt.figure()
                plt.imshow(cropped, cmap='gray', origin='upper')
                plt.title(f"Cropped region\n(x={x}, y={y}, x_c={int(x_c)}, y_c={int(y_c)}, width={w}, height={h})")
                plt.show(block=False)

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
                x_c = 0
                y_c = 0
        else:
            execute_flag = 0
            x_c = 0
            y_c = 0

        plt.close('all') # Close all plot window

    return x_c, y_c
