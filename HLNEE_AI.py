import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from HLN_Predict import preprocess_image, prediction
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Define a function to display the prediction results in a new window
def results_window(image):
    results_window_tk = Tk()
    results_window_tk.title('Results')
    results_window_tk.geometry('640x520')

    # Get the predicted class labels and probabilities using the HLN model
    class_labels, top_probs = prediction(image)

    # Create a bar chart of the predicted labels and probabilities using matplotlib
    fig = Figure(figsize=(5, 4), dpi=100, facecolor='black')
    ax = fig.add_subplot(111)
    ax.bar(class_labels, top_probs)

    # Customize the chart's appearance
    ax.tick_params(axis='x', labelcolor='white')
    ax.tick_params(axis='y', labelcolor='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')

    # Display the chart in the new window using tkinter
    canvas = FigureCanvasTkAgg(fig, master=results_window_tk)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Define a function to open an image file, preprocess it, and call the results_window function
def open_image():
    # Open a file explorer window to select an image file
    file_path = filedialog.askopenfilename()

    if file_path:
        # Open the image file using PIL and display it in the main window using tkinter
        image = Image.open(file_path)
        photo = ImageTk.PhotoImage(image)
        main_window.config(image=photo)
        main_window.image = photo

        # Preprocess the image and call the results_window function to display the prediction results
        image = preprocess_image(image)
        results_window(image)

# Create the main tkinter window
master = Tk()
master.title('HLNEE AI')
master.config(bg='black')
master.geometry('640x520')

# Create a canvas widget on the main window for displaying text
canvas = Canvas(master, width=640, height=520, bg='black')
canvas.pack()

# Create a label widget on the main window for displaying the selected image
main_window = Label(master)
main_window.pack()

# Create a button widget on the main window for selecting an image and calling the open_image function
select_image_button = Button(master, text="Select Image for Prediction", command=open_image, width=20, font=('Helvetica', 10))
select_image_button.pack()
select_image_button.place(relx=0.5, rely=0.5, anchor=CENTER)

# Create text widgets on the canvas to display welcome and information messages
canvas.create_text(320, 130, text='Welcome to HLNEE AI, a software for image recognition!', fill='white', font=('Helvetica', 18))
canvas.create_text(320, 480, text='HLNEE AI is an image recognition software based on the HLN model,\n'
                                  '     trained in 100 classes for recognition and prediction of images', fill='white', font=('Helvetica', 10))

# Start the tkinter main event loop to display the GUI and handle user interactions
master.mainloop()
