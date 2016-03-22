"""
Fast and dirty format for re-producable graphs


{
'title':"Title string",
'x-axis':"x-axis label",
"y-axis":"y-axis label",
"plots":{
    "linename":{
        "x":[1,2,3,4],
        "y":[1,2,3,4],
        "style":"b"
    }
    },
"target":"test.png"
}

This should use matplotlib to buld a plot with the properties and a key

"""

import matplotlib.pyplot as plt
import json


def Render(raw_json):
    data = json.loads(raw_json)
    plt.title(data["title"])
    plt.xlabel(data["x-axis"])
    plt.ylabel(data["y-axis"])
    labels = sorted(data["plots"].keys())
    if len(labels)==10:
        tmp = labels[1]
        labels.remove(tmp)
        labels.append(tmp)
    for label in labels:
        plot_info = data["plots"][label]
        plt.plot(plot_info["x"], plot_info["y"], label=label)
    plt.axis([0,.9,0,1])
    plt.legend(loc=2)

    if "target" in data.keys():
        # plt.draw()
        plt.savefig(data["target"])
    else:
        # plt.draw()
        plt.show()
    plt.clf()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        for filename in sys.argv[1:]:
            with open(filename, "r") as fp:
                Render(fp.read())
