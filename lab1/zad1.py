#%matplotlib ipympl      # nie wiem czy to cos dalo ale w razie czego : jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-matplotlib

from matplotlib.widgets import SpanSelector, CheckButtons, Button, TextBox
import numpy as np
import matplotlib.pyplot as plt
# wczytywanie pliku
def readFile(filename):
    global content
    try:
        with open(filename, 'r') as file:
            return file.read()

    except FileNotFoundError:
        print(f"Plik {filename} nie został znaleziony.")
    except Exception as e:
        print(f"Wystąpił błąd podczas wczytywania pliku {filename}: {e}")

# Wyświetlanie klikniętego wykresu
def callback(label):
    ln = lines_by_label[label]
    ln.set_visible(not ln.get_visible())
    ln.figure.canvas.draw_idle()

    index = lines.index(ln)
    lines2[index].set_visible(not lines2[index].get_visible())
    lines2[index].figure.canvas.draw_idle()

    onselect(xRange[0], xRange[1])


# Wybieranie obszaru wykresu
def onselect(xmin, xmax):
    global xRange
    xRange = xmin, xmax
    indmin, indmax = np.searchsorted(x, (xmin, xmax))
    indmax = min(len(x) - 1, indmax)

    region_x = x[indmin:indmax]

    if len(region_x) >= 2:
        minY = []
        maxY = []
        for i in range(len(y)):
            region_y = y[i][indmin:indmax]
            lines2[i].set_data(region_x, region_y)
            if lines[i].get_visible():
                maxY.append(max(region_y))
                minY.append(min(region_y))

        if len(minY) != 0:
            axs[1].set_ylim(min(minY), max(maxY))
        axs[1].set_xlim(region_x[0], region_x[-1])
        fig.canvas.draw_idle()

#   wczytywanie pliku
filename = 'ekgNoise'
ekg = readFile(filename)

######################## dane ###################
ekg = np.array([list(map(float, line.split())) for line in ekg.split("\n")])
y = ekg.T
x = np.arange(1, len(ekg) + 1) * 0.001


############# wykres główny #####################

# fig, ax = plt.subplots()
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

lines=[]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78']
colors = colors[:len(y)]

for i in range(len(y)):
    j, = axs[0].plot(x, y[i], visible=False, lw=0.5, color=colors[i], label="ekg "+str(i + 1))
    lines.append(j)

lines_by_label = {l.get_label(): l for l in lines}


rax = axs[1].inset_axes([1.0, 0.0, 0.19, 0.8])
check = CheckButtons(
    ax=rax,
    labels=lines_by_label.keys(),
    actives=[l.get_visible() for l in lines_by_label.values()],
    label_props={'color': colors},
    frame_props={'edgecolor': colors},
    check_props={'facecolor': colors},
)

check.on_clicked(callback)


############# wykres pomocniczy ##############


# fig2, ax2 = plt.subplots()
lines2 = []

for _ in range(len(y)):
    line, = axs[1].plot([], [])
    line.set_visible(False)
    lines2.append(line)

xRange = [0,0]
span = SpanSelector(
    axs[0],
    onselect,
    "horizontal",
    useblit=True,
    props=dict(alpha=0.5, facecolor="tab:blue"),
    interactive=True,
    drag_from_anywhere=True
)
newFileName = "NowyPlik.png"
def savePlot(event):
    global newFileName
    if newFileName.find(".png")==-1 or newFileName.find(".pdf")==-1:
        newFileName+=".png"
    extent = axs[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(newFileName, bbox_inches=extent.expanded(1.4, 1.2))


save_button_ax = plt.axes([0.9, 0.05, 0.1, 0.05])  # Położenie i rozmiar przycisku
save_button = Button(save_button_ax, 'Zapisz')
save_button.on_clicked(savePlot)  # Przypisanie funkcji do przycisku













plt.show()