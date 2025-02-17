

def add_function_to_plt(plt, func, x_domain):
    return


def add_straight_line_to_plt(plt, point1, point2, color='green', linestyle='--'):
    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    plt.plot(x_values, y_values, color=color, linestyle=linestyle)
    