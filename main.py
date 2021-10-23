from os import walk
import matplotlib.pyplot as plt
import numpy as np

labels = list(range(1, 25))
labels = [str(x) for x in labels]


def plot_comparison(t_class_name):
    f = []
    for (dirpath, dirnames, filenames) in walk("./data"):
        f.extend(filenames)
        break
    # Sort on fig size label
    f = sorted(f, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    # labels = ['1', '2', '3', '4', '5', '6', '7', '8', '10', '11', '12', '13', '14', '15',
    #         '16', '17', '18', '19', '20']
    means_dict = {
        "GH": [],
        "GW": [],
        "MBQ": []
    }
    for filename in f:
        if "comparison" in filename:
            with open("./data/" + filename) as file:
                for i, line in enumerate(file):
                    if "GW" in line or "GH" in line or "MBQ" in line:
                        alg = line.strip()
                        continue
                    test_class = line.split(":")[0]
                    total_run = int(line.split(":")[-1].split(",")[0])
                    if test_class == t_class_name:
                        means_dict[alg].append(total_run)

    plt.yscale("log")
    plt.plot(labels[:len(means_dict["GH"])], means_dict["GH"], color='red', label='GH', marker="x")
    plt.plot(labels[:len(means_dict["GH"])], means_dict["GW"], color='blue', label='GW', marker="x")
    plt.plot(labels[:len(means_dict["GH"])], means_dict["MBQ"], color='green', label='MBQ', marker="x")

    plt.xlabel('figure size $\mathregular{(2^x = n)}$', fontweight='bold')
    plt.ylabel("ns")
    plt.title("Execution Times (" + t_class_name + ")")
    plt.legend()
    plt.savefig("comparison_" + t_class_name)


def plot_GH_removals(t_class_name):
    f = []
    for (dirpath, dirnames, filenames) in walk("./data"):
        f.extend(filenames)
        break
    # Sort on fig size label
    f = sorted(f, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    means = []
    means_sort = []
    for filename in f:
        if "GH" not in filename:
            continue
        with open("./data/" + filename) as file:
            for line in file:
                test_class = line.split(":")[0]
                removals = int(line.split(":")[-1].split(",")[3])
                if test_class == t_class_name:
                    means.append(removals)

    plt.yscale("log")
    plt.plot(labels[:len(means)], means, color='red', label='UH Removals', marker="x")

    plt.xlabel('figure size $\mathregular{(2^x = n)}$', fontweight='bold')
    plt.ylabel("ns")
    plt.title("Graham Scan List Removals (" + t_class_name + ")")

    plt.legend()
    plt.savefig("GH_removals_" + t_class_name)


def plot_GH_orientations(t_class_name):
    f = []
    for (dirpath, dirnames, filenames) in walk("./data"):
        f.extend(filenames)
        break
    # Sort on fig size label
    f = sorted(f, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    means = []
    means_sort = []
    for filename in f:
        if "GH" not in filename:
            continue
        with open("./data/" + filename) as file:
            for line in file:
                test_class = line.split(":")[0]
                orientations = int(line.split(":")[-1].split(",")[2])
                rem = int(line.split(":")[-1].split(",")[3])

                if test_class == t_class_name:
                    means.append(orientations)
                    means_sort.append(rem)

    plt.yscale("log")
    plt.plot(labels[:len(means)], means, color='red', label='#Orientation Calls', marker="x")
    plt.plot(labels[:len(means)], means_sort, color='blue', label='#Removals', marker="x")

    plt.xlabel('figure size $\mathregular{(2^x = n)}$', fontweight='bold')
    plt.ylabel("#Actions")
    plt.title("GH Sidedness & Removals (" + t_class_name + ")")

    plt.legend()
    plt.savefig("GH_orientation_removals_" + t_class_name)


def plot_GH_sort(t_class_name):
    f = []
    for (dirpath, dirnames, filenames) in walk("./data"):
        f.extend(filenames)
        break
    # Sort on fig size label
    f = sorted(f, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    means = []
    means_sort = []
    for filename in f:
        if "GH" not in filename:
            continue
        with open("./data/" + filename) as file:
            for line in file:
                test_class = line.split(":")[0]
                total_run = int(line.split(":")[-1].split(",")[0])
                sort_time = int(line.split(":")[-1].split(",")[1])
                if test_class == t_class_name:
                    means.append(total_run)
                    means_sort.append(sort_time)

    plt.yscale("log")
    plt.plot(labels[:len(means)], means, color='red', label='Total Execution Time', marker="x")
    plt.plot(labels[:len(means)], means_sort, color='blue', label='Sort Time', marker="x")

    plt.xlabel('figure size $\mathregular{(2^x = n)}$', fontweight='bold')
    plt.ylabel("ns")
    plt.title("Graham Scan Sort/Execution Times (" + t_class_name + ")")

    plt.legend()
    plt.savefig("GH_sort_exec_" + t_class_name)


def plot_GH_theoretical():
    f = []
    for (dirpath, dirnames, filenames) in walk("./data"):
        f.extend(filenames)
        break
    # Sort on fig size label
    f = sorted(f, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    square_means = []
    circle_means = []
    log_means = []
    quad_means = []
    for filename in f:
        if "GH" not in filename:
            continue
        with open("./data/" + filename) as file:
            for line in file:
                test_class = line.split(":")[0]
                total_run = int(line.split(":")[-1].split(",")[0])
                if test_class == "SQUARE":
                    square_means.append(total_run)
                if test_class == "CIRCLE":
                    circle_means.append(total_run)
                if test_class == "LOG":
                    log_means.append(total_run)
                if test_class == "QUADRATIC":
                    quad_means.append(total_run)

    exponents = np.power(2, np.asarray(labels[:len(square_means)]).astype(np.int_))
    log = np.log(exponents)
    # runime/(n * log(n))
    square_means = np.asarray(square_means) / (exponents * log)
    # runime/(n * log(n))
    circle_means = np.asarray(circle_means) / (exponents * log)
    # runime/(n * log(n))
    log_means = np.asarray(log_means) / (exponents * log)
    # runime/(n * log(n))
    quad_means = np.asarray(quad_means) / (exponents * log)

    plt.yscale("log")
    plt.plot(labels[:len(square_means)], square_means, color='red', label='SQUARE', marker="x")
    plt.plot(labels[:len(square_means)], circle_means, color='blue', label='CIRCLE', marker="x")
    plt.plot(labels[:len(square_means)], log_means, color='green', label='LOG', marker="x")
    plt.plot(labels[:len(square_means)], quad_means, color='purple', label='QUADRATIC', marker="x")

    plt.xlabel('figure size $\mathregular{(2^x = n)}$', fontweight='bold')
    plt.ylabel("Runtime/Theoretical Runtime")
    plt.title("GH Experimental vs Theoretical")

    plt.legend()
    plt.savefig("GH_runtime_theoretical")


def plot_GH_hullsize():
    f = []
    for (dirpath, dirnames, filenames) in walk("./data"):
        f.extend(filenames)
        break
    # Sort on fig size label
    f = sorted(f, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    square_means = []
    circle_means = []
    log_means = []
    quad_means = []
    for filename in f:
        if "GH" not in filename:
            continue
        with open("./data/" + filename) as file:
            for line in file:
                test_class = line.split(":")[0]
                total_run = int(line.split(":")[-1].split(",")[4])
                if test_class == "SQUARE":
                    square_means.append(total_run)
                if test_class == "CIRCLE":
                    circle_means.append(total_run)
                if test_class == "LOG":
                    log_means.append(total_run)
                if test_class == "QUADRATIC":
                    quad_means.append(total_run)

    plt.yscale("log")
    plt.plot(labels[:len(square_means)], square_means, color='red', label='SQUARE', marker="x")
    plt.plot(labels[:len(square_means)], circle_means, color='blue', label='CIRCLE', marker="x")
    plt.plot(labels[:len(square_means)], log_means, color='green', label='LOG', marker="x")
    plt.plot(labels[:len(square_means)], quad_means, color='purple', label='QUADRATIC', marker="x")

    plt.xlabel('figure size $\mathregular{(2^x = n)}$', fontweight='bold')
    plt.ylabel("Upper Hull Size")
    plt.title("GH Result Sizes")

    plt.legend()
    plt.savefig("GH_upper_hull")


def plot_MBQ_2d():
    f = []
    for (dirpath, dirnames, filenames) in walk("./data"):
        f.extend(filenames)
        break
    # Sort on fig size label
    f = sorted(f, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    square_means = []
    circle_means = []
    log_means = []
    quad_means = []
    for filename in f:
        if "MBQ" not in filename:
            continue
        with open("./data/" + filename) as file:
            for line in file:
                test_class = line.split(":")[0]
                total_run = int(line.split(":")[-1].split(",")[3])
                if test_class == "SQUARE":
                    square_means.append(total_run)
                if test_class == "CIRCLE":
                    circle_means.append(total_run)
                if test_class == "LOG":
                    log_means.append(total_run)
                if test_class == "QUADRATIC":
                    quad_means.append(total_run)

    plt.yscale("log")
    plt.plot(labels[:len(square_means)], square_means, color='red', label='SQUARE', marker="x")
    plt.plot(labels[:len(square_means)], circle_means, color='blue', label='CIRCLE', marker="x")
    plt.plot(labels[:len(square_means)], log_means, color='green', label='LOG', marker="x")
    plt.plot(labels[:len(square_means)], quad_means, color='purple', label='QUADRATIC', marker="x")

    plt.xlabel('figure size $\mathregular{(2^x = n)}$', fontweight='bold')
    plt.ylabel("#LP Calls")
    plt.title("MBQ Avg. 2D LP Calls")

    plt.legend()
    plt.savefig("MBQ_total_2d")


def plot_MBQ_1d():
    f = []
    for (dirpath, dirnames, filenames) in walk("./data"):
        f.extend(filenames)
        break
    # Sort on fig size label
    f = sorted(f, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    square_means = []
    circle_means = []
    log_means = []
    quad_means = []
    for filename in f:
        if "MBQ" not in filename:
            continue
        with open("./data/" + filename) as file:
            for line in file:
                test_class = line.split(":")[0]
                total_run = int(line.split(":")[-1].split(",")[2])
                if test_class == "SQUARE":
                    square_means.append(total_run)
                if test_class == "CIRCLE":
                    circle_means.append(total_run)
                if test_class == "LOG":
                    log_means.append(total_run)
                if test_class == "QUADRATIC":
                    quad_means.append(total_run)

    plt.yscale("log")
    plt.plot(labels[:len(square_means)], square_means, color='red', label='SQUARE', marker="x")
    plt.plot(labels[:len(square_means)], circle_means, color='blue', label='CIRCLE', marker="x")
    plt.plot(labels[:len(square_means)], log_means, color='green', label='LOG', marker="x")
    plt.plot(labels[:len(square_means)], quad_means, color='purple', label='QUADRATIC', marker="x")

    plt.xlabel('figure size $\mathregular{(2^x = n)}$', fontweight='bold')
    plt.ylabel("#LP Calls")
    plt.title("MBQ Avg. 1D LP Calls")

    plt.legend()
    plt.savefig("MBQ_total_1d")


def plot_GW_runtime_hulladjusted():
    f = []
    for (dirpath, dirnames, filenames) in walk("./data"):
        f.extend(filenames)
        break
    # Sort on fig size label
    f = sorted(f, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    square_means = []
    circle_means = []
    log_means = []
    quad_means = []
    for filename in f:
        if "GW" not in filename:
            continue
        with open("./data/" + filename) as file:
            for line in file:
                test_class = line.split(":")[0]
                total_run = int(line.split(":")[-1].split(",")[0])
                if test_class == "SQUARE":
                    square_means.append(total_run)
                if test_class == "CIRCLE":
                    circle_means.append(total_run)
                if test_class == "LOG":
                    log_means.append(total_run)
                if test_class == "QUADRATIC":
                    quad_means.append(total_run)

    square_means = np.asarray(square_means)
    exponents = np.power(2, np.asarray(labels[:len(square_means)]).astype(np.uint64))
    a = (exponents * exponents)
    # runime/(n * n * 1/3)
    square_means = square_means / (exponents * exponents * 0.3)
    # runtime/(n * n * 1/3(
    circle_means = circle_means / (exponents * exponents * 0.3)
    # runtime/(n²)
    log_means = log_means / (exponents * exponents)
    # runtmie/(n*2)
    quad_means = quad_means / (exponents * 2)

    plt.plot(labels[4:len(square_means)], square_means[4:], color='red', label='SQUARE', marker="x")
    plt.plot(labels[4:len(square_means)], circle_means[4:], color='blue', label='CIRCLE', marker="x")
    plt.plot(labels[4:len(square_means)], log_means[4:], color='green', label='LOG', marker="x")
    plt.plot(labels[4:len(square_means)], quad_means[4:], color='purple', label='QUADRATIC', marker="x")

    plt.xlabel('figure size $\mathregular{(2^x = n)}$', fontweight='bold')
    plt.ylabel("Runtime (ns)/Theoretical Runtime")
    plt.title("Giftwrap Experimental/Theoretical Results")

    plt.legend()
    plt.savefig("GW_total_time_hulladjusted_cut")


def plot_GW_runtime():
    f = []
    for (dirpath, dirnames, filenames) in walk("./data"):
        f.extend(filenames)
        break
    # Sort on fig size label
    f = sorted(f, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    square_means = []
    circle_means = []
    log_means = []
    quad_means = []
    for filename in f:
        if "GW" not in filename:
            continue
        with open("./data/" + filename) as file:
            for line in file:
                test_class = line.split(":")[0]
                total_run = int(line.split(":")[-1].split(",")[0])
                if test_class == "SQUARE":
                    square_means.append(total_run)
                if test_class == "CIRCLE":
                    circle_means.append(total_run)
                if test_class == "LOG":
                    log_means.append(total_run)
                if test_class == "QUADRATIC":
                    quad_means.append(total_run)
    plt.yscale('log')
    plt.plot(labels[:len(square_means)], square_means, color='red', label='SQUARE', marker="x")
    plt.plot(labels[:len(square_means)], circle_means, color='blue', label='CIRCLE', marker="x")
    plt.plot(labels[:len(square_means)], log_means, color='green', label='LOG', marker="x")
    plt.plot(labels[:len(square_means)], quad_means, color='purple', label='QUADRATIC', marker="x")

    plt.xlabel('figure size $\mathregular{(2^x = n)}$', fontweight='bold')
    plt.ylabel("ns")
    plt.title("Giftwrap Avg. Execution Time")

    plt.legend()
    plt.savefig("GW_total_time")


def plot_MBQ_runtime():
    f = []
    for (dirpath, dirnames, filenames) in walk("./data"):
        f.extend(filenames)
        break
    # Sort on fig size label
    f = sorted(f, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    square_means = []
    circle_means = []
    log_means = []
    quad_means = []
    for filename in f:
        if "MBQ" not in filename:
            continue
        with open("./data/" + filename) as file:
            for line in file:
                test_class = line.split(":")[0]
                total_run = int(line.split(":")[-1].split(",")[0])
                if test_class == "SQUARE":
                    square_means.append(total_run)
                if test_class == "CIRCLE":
                    circle_means.append(total_run)
                if test_class == "LOG":
                    log_means.append(total_run)
                if test_class == "QUADRATIC":
                    quad_means.append(total_run)

    plt.yscale("log")
    plt.plot(labels[:len(square_means)], square_means, color='red', label='SQUARE', marker="x")
    plt.plot(labels[:len(square_means)], circle_means, color='blue', label='CIRCLE', marker="x")
    plt.plot(labels[:len(square_means)], log_means, color='green', label='LOG', marker="x")
    plt.plot(labels[:len(square_means)], quad_means, color='purple', label='QUADRATIC', marker="x")

    plt.xlabel('figure size $\mathregular{(2^x = n)}$', fontweight='bold')
    plt.ylabel("ns")
    plt.title("MBQ Avg. Execution Time")

    plt.legend()
    plt.savefig("MBQ_total_time")


def plot_MBQ_recursion():
    f = []
    for (dirpath, dirnames, filenames) in walk("./data"):
        f.extend(filenames)
        break
    # Sort on fig size label
    f = sorted(f, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    square_means = []
    circle_means = []
    log_means = []
    quad_means = []
    for filename in f:
        if "MBQ" not in filename:
            continue
        with open("./data/" + filename) as file:
            for line in file:
                test_class = line.split(":")[0]
                total_run = int(line.split(":")[-1].split(",")[1])
                if test_class == "SQUARE":
                    square_means.append(total_run)
                if test_class == "CIRCLE":
                    circle_means.append(total_run)
                if test_class == "LOG":
                    log_means.append(total_run)
                if test_class == "QUADRATIC":
                    quad_means.append(total_run)

    plt.plot(labels[:len(square_means)], square_means, color='red', label='SQUARE', marker="x")
    plt.plot(labels[:len(square_means)], circle_means, color='blue', label='CIRCLE', marker="x")
    plt.plot(labels[:len(square_means)], log_means, color='green', label='LOG', marker="x")
    plt.plot(labels[:len(square_means)], quad_means, color='purple', label='QUADRATIC', marker="x")

    plt.xlabel('figure size $\mathregular{(2^x = n)}$', fontweight='bold')
    plt.ylabel("ns")
    plt.title("MBQ Avg. Recursion Depth")

    plt.legend()
    plt.savefig("MBQ_recursion_depth_total")


def plot_GH():
    f = []
    for (dirpath, dirnames, filenames) in walk("./data"):
        f.extend(filenames)
        break
    # Sort on fig size label
    f = sorted(f, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    square_means = []
    circle_means = []
    log_means = []
    quad_means = []
    shuf_quad_means = []
    shuf_log_means = []

    for filename in f:
        if "GH" not in filename:
            continue
        with open("./data/" + filename) as file:
            for line in file:
                test_class = line.split(":")[0]
                total_run = int(line.split(":")[-1].split(",")[0])
                if test_class == "SQUARE":
                    square_means.append(total_run)
                if test_class == "CIRCLE":
                    circle_means.append(total_run)
                if test_class == "LOG":
                    log_means.append(total_run)
                if test_class == "QUADRATIC":
                    quad_means.append(total_run)
                if test_class == "SHUFFLEDLOG":
                    shuf_log_means.append(total_run)
                if test_class == "SHUFFLEDQUAD":
                    shuf_quad_means.append(total_run)

    plt.yscale('log')
    plt.plot(labels[:len(square_means)], square_means, color='red', label='SQUARE', marker="x")
    plt.plot(labels[:len(square_means)], circle_means, color='blue', label='CIRCLE', marker="x")
    plt.plot(labels[:len(square_means)], log_means, color='green', label='LOG', marker="x")
    plt.plot(labels[:len(square_means)], quad_means, color='purple', label='QUADRATIC', marker="x")
    plt.plot(labels[:len(square_means)], shuf_quad_means, color='black', label='RANDOMISED QUAD', marker="x")
    plt.plot(labels[:len(square_means)], shuf_log_means, color='yellow', label='RANDOMISED LOG', marker="x")

    plt.xlabel('figure size $\mathregular{(2^x = n)}$', fontweight='bold')
    plt.ylabel("ns")
    plt.title("Graham Scan Avg. Execution Time")

    plt.legend()
    plt.savefig("GH_total_time")


def visualise():
    inputPointsX = []
    inputPointsXY = []
    hullX = []
    hullY = []
    lineNum = 0
    with open("visualiseFile") as file:
        for line in file:
            if lineNum == 0:
                points = line.split(";")
                points = points[:-1]
                for pointString in points:
                    var = pointString[pointString.find("(") + 1:pointString.find(")")]
                    xyvar = var.split(",")
                    inputPointsX.append(float(xyvar[0]))
                    inputPointsXY.append(float(xyvar[1]))
                lineNum += 1
                continue
            if lineNum == 1:
                points = line.split(";")
                points = points[:-1]
                for pointString in points:
                    var = pointString[pointString.find("(") + 1:pointString.find(")")]
                    xyvar = var.split(",")
                    hullX.append(float(xyvar[0]))
                    hullY.append(float(xyvar[1]))

    plt.scatter(inputPointsX, inputPointsXY, c="green")
    plt.scatter(hullX, hullY, c="r")
    plt.show()


if __name__ == '__main__':
    plot_GW_runtime_hulladjusted()

