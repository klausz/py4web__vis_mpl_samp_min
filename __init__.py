from py4web import action
from py4web import response

#index is used without html as I dont need html code
@action("index")
def index():
    return "Hello World"


@action("mpl_barchart")
def mpl_barchart():
#    from py4web import response
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    import numpy as np
    import io
    from py4web import response
    response.headers['Content-Type']='image/png'
    np.random.seed(19680801)
    plt.rcdefaults()
    fig, ax = plt.subplots()
    ## Example data
    people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
    y_pos = np.arange(len(people))
    performance = 3 + 10 * np.random.rand(len(people))
    error = np.random.rand(len(people))
    ax.barh(y_pos, performance, xerr=error, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(people)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Performance')
    ax.set_title('How fast do you want to go today?')
    #plt.show()
    s = io.BytesIO()
    canvas=FigureCanvas(fig)
    canvas.print_png(s)
    return s.getvalue()


@action("mpl_sinusSubPlots")
def mpl_sinusSubPlots():
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    import numpy as np
    import io
#    from py4web import response #needs to be imported, can be done outside
    response.headers['Content-Type']='image/png'
    def f(t):
        s1 = np.cos(2*np.pi*t)
        e1 = np.exp(-t)
        return s1 * e1
    t1 = np.arange(0.0, 5.0, 0.1)
    t2 = np.arange(0.0, 5.0, 0.02)
    t3 = np.arange(0.0, 2.0, 0.01)
    fig, axs = plt.subplots(3, 1, constrained_layout=True) #3 defines the number of plots, prereserved in an array
    fig.suptitle('This is a somewhat long figure title', fontsize=16)
    axs[0].plot(t3, np.cos(2*np.pi*t3), '--')
    axs[0].set_title('subplot 1')
    axs[0].set_xlabel('time (s)')
    axs[0].set_ylabel('Undamped')
    axs[1].plot(t3, np.cos(1*np.pi*t3), '--')
    axs[1].set_title('subplot 2')
    axs[1].set_xlabel('time (s)')
    axs[1].set_ylabel('Undamped')
    axs[2].plot(t1, f(t1), 'o', t2, f(t2), '-')
    axs[2].set_title('subplot 3')
    axs[2].set_xlabel('distance (m)')
    axs[2].set_ylabel('Damped oscillation')
    #plt.show()
    s1 = io.BytesIO()
    canvas=FigureCanvas(fig)
    canvas.print_png(s1)
    return s1.getvalue()


@action("mpl_linesHaV")
def mpl_linesHaV():
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    import numpy as np
    import numpy.random as rnd
    import io
#    from py4web import response #needs to be imported, can be done outside
    response.headers['Content-Type']='image/png'
    def f(t):
        s1 = np.sin(2 * np.pi * t)
        e1 = np.exp(-t)
        return np.absolute((s1 * e1)) + .05
    t = np.arange(0.0, 5.0, 0.1)
    s = f(t)
    nse = rnd.normal(0.0, 0.3, t.shape) * s
    fig = plt.figure(figsize=(12, 6))
    vax = fig.add_subplot(121)
    hax = fig.add_subplot(122)
    vax.plot(t, s + nse, '^')
    vax.vlines(t, [0], s)
    vax.set_xlabel('time (s)')
    vax.set_title('Vertical lines demo')
    hax.plot(s + nse, t, '^')
    hax.hlines(t, [0], s, lw=2)
    hax.set_xlabel('time (s)')
    hax.set_title('Horizontal lines demo')
    #plt.show()
    s = io.BytesIO()
    canvas=FigureCanvas(fig)
    canvas.print_png(s)
    return s.getvalue()


@action("mpl_violinplot")
def mpl_violinplot():
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    import numpy as np
    import io
#    from py4web import response #needs to be imported, can be done outside
    response.headers['Content-Type']='image/png'
    np.random.seed(19680801)
    fs = 10  # fontsize
    pos = [1, 2, 4, 5, 7, 8]
    data = [np.random.normal(0, std, size=100) for std in pos]
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6, 6))
    axes[0, 0].violinplot(data, pos, points=20, widths=0.3,
                          showmeans=True, showextrema=True, showmedians=True)
    axes[0, 0].set_title('Custom violinplot 1', fontsize=fs)
    axes[0, 1].violinplot(data, pos, points=40, widths=0.5,
                          showmeans=True, showextrema=True, showmedians=True,
                          bw_method='silverman')
    axes[0, 1].set_title('Custom violinplot 2', fontsize=fs)
    axes[0, 2].violinplot(data, pos, points=60, widths=0.7, showmeans=True,
                          showextrema=True, showmedians=True, bw_method=0.5)
    axes[0, 2].set_title('Custom violinplot 3', fontsize=fs)
    axes[1, 0].violinplot(data, pos, points=80, vert=False, widths=0.7,
                          showmeans=True, showextrema=True, showmedians=True)
    axes[1, 0].set_title('Custom violinplot 4', fontsize=fs)
    axes[1, 1].violinplot(data, pos, points=100, vert=False, widths=0.9,
                          showmeans=True, showextrema=True, showmedians=True,
                          bw_method='silverman')
    axes[1, 1].set_title('Custom violinplot 5', fontsize=fs)
    axes[1, 2].violinplot(data, pos, points=200, vert=False, widths=1.1,
                          showmeans=True, showextrema=True, showmedians=True,
                          bw_method=0.5)
    axes[1, 2].set_title('Custom violinplot 6', fontsize=fs)
    for ax in axes.flat:
        ax.set_yticklabels([])
    fig.suptitle("Violin Plotting Examples")
    fig.subplots_adjust(hspace=0.4)
    #plt.show()
    s = io.BytesIO()
    canvas=FigureCanvas(fig)
    canvas.print_png(s)
    return s.getvalue()

    
@action("mpl_plot")
def mpl_plot(title='title',xlab='x',ylab='y',mode='plot',
    data={'xxx':[(0,0),(1,1),(1,2),(3,3)],
      'yyy':[(0,0,.2,.2),(2,1,0.2,0.2),(2,2,0.2,0.2),
      (3,3,0.2,0.3)]}):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import io
    response.headers['Content-Type']='image/png'
    fig=Figure()
    fig.set_facecolor('white')
    ax=fig.add_subplot(111)
    if title: ax.set_title(title)
    if xlab: ax.set_xlabel(xlab)
    if ylab: ax.set_ylabel(ylab)
    legend=[]
    keys=sorted(data)
    for key in keys:
        stream = data[key]
        (x,y)=([],[])
    for point in stream:
        x.append(point[0])
        y.append(point[1])
    if mode=='plot':
        ell=ax.plot(x, y)
        legend.append((ell,key))
    if mode=='hist':
        ell=ax.hist(y,20)
    if legend:
        ax.legend([[x for (x,y) in legend], [y for (x,y) in
            legend]],
            'upper right', shadow=True)
    canvas=FigureCanvas(fig)
    stream=io.BytesIO() #stream=cStringIO.StringIO()
    canvas.print_png(stream)
    return stream.getvalue()

@action("mpl_numpytst")
def mpl_numpytst():
    response.headers['Content-Type']='image/png'
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    import numpy as np
    import io
    x = [1, 2, 3, 4, 5]
    y = [1, 4, 9, 16, 25]
    fig=Figure()
    fig = plt.figure()
    fig.add_subplot(111)
    p1 = plt.scatter(x, y)
    s = io.BytesIO()
    canvas=FigureCanvas(fig)
    canvas.print_png(s)
    return s.getvalue()
    
@action("mpl_line_json")
def mpl_line_json():
    import pandas as pd
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    import numpy as np
    import io
    response.headers['Content-Type']='image/png'
    d = {"sell": [
           {
               "Rate": 0.425,
               "Quantity": 0.25
           },
           {
               "Rate": 0.6425,
               "Quantity": 0.40
           },
           {
               "Rate": 0.7025,
               "Quantity": 0.8
           },
           {
               "Rate": 0.93,
               "Quantity": 0.59
           }
        ]}
    df = pd.DataFrame(d['sell'])
    df.plot(x='Quantity', y='Rate')
    fig=Figure()
    fig = plt.figure()
    fig.add_subplot(111)
    p1 = plt.scatter(df.Quantity, df.Rate)
    s = io.BytesIO()
    canvas=FigureCanvas(fig)
    canvas.print_png(s)
    return s.getvalue()