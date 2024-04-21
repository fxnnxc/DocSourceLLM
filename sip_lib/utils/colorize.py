import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def colorize(words, color_array, color_map_version=1, custom_mapping=None, next_line_words=1000000):
    # words is a list of words
    # color_array is an array of numbers between 0 and 1 of length equal to words
    if custom_mapping is not None:
        cmap = custom_mapping
    else:
        if color_map_version =='viridis':
            cmap = matplotlib.cm.get_cmap('viridis')
        elif color_map_version ==1 :
            cmap = custom_colormap1
        elif color_map_version ==2 :
            cmap = custom_colormap2
        else:
            raise ValueError()
    template = '<span style="color: black; background-color: {}">{}</span>'
    colored_strings = ['<p>']
    next_line_count = 0
    for c , (word, color) in enumerate(zip(words, color_array)):
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        colored_strings.append(template.format(color, '&nbsp' + word + '&nbsp'))
        if c% next_line_words == next_line_words -1:
            colored_strings.append("<br>")
    colored_strings.append("</p>")
    colored_strings = "".join(colored_strings)
    return colored_strings

def custom_colormap1(c):
    zero = np.array((1,1,1))
    end = np.array((0,1,111/255))
    return zero * (1-c) + end*c  

def custom_colormap2(c):
    zero = np.array((255/255,46/255,102/255))
    end = np.array((0,1,111/255))
    return zero * (1-c) + end*c  
