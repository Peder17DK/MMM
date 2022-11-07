import tkinter as tk
from tkinter import ttk


def Start(): # 
    pass

def Stop():
    pass

def TomaterPlukket():
    pass

def ShowGrayscale():
    pass

def ShowEdges():
    pass

def ShowOriginal():
    pass

def ShowCircles():
    pass

def UpdateValues():
    print(scale1.get(), scale2.get())

master = tk.Tk()
master.title('MMM Gui')
master.geometry("750x750")

image = tk.PhotoImage(file = "C:\\Users\\budde\Documents\\Python projects\\Vision projekt UCL\\tomater.png")
labelImage = tk.Label(master, image= image)
labelImage.place(x=200, y= 100)

labelTomaterPlukket = tk.Label(master, text= "Tomater plukket i alt: " + str(TomaterPlukket()), font=("Arial", 15))
labelTomaterPlukket.place(x = 100, y = 600)

labelTomaterPlukket = tk.Label(master, text= "Tomater plukket i dag: " + str(TomaterPlukket()), font=("Arial", 15))
labelTomaterPlukket.place(x = 100, y = 650)

buttonStart = tk.Button(master, text= "Start", command=Start)
buttonStart.place(x=100, y=100)

buttonStop = tk.Button(master, text= "Stop", command=Stop)
buttonStop.place(x=100, y=200)

buttonSO = tk.Button(master, text= "Original", command=ShowOriginal)
buttonSO.place(x=200, y=50)

buttonSG = tk.Button(master, text= "Grayscale", command=ShowGrayscale)
buttonSG.place(x=275, y=50)

buttonSE = tk.Button(master, text= "Edges", command=ShowEdges)
buttonSE.place(x=350, y=50)

buttonSC = tk.Button(master, text= "Show circles", command=ShowCircles)
buttonSC.place(x=425, y=50)

scale1 = tk.Scale(master, from_= 200, to= 0)
scale1.place(x= 220, y= 450 )

scale2 = tk.Scale(master, from_= 200, to= 0)
scale2.place(x= 270, y= 450 )

buttonNewValues = tk.Button(master, text= "Accept", command=UpdateValues)
buttonNewValues.place(x=400, y= 450)





tk.mainloop()