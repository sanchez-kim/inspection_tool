import argparse


def main(args):
    with open(args.log, "r", encoding="ISO-8859-1") as f:
        data = f.readlines()

    no_list = []

    model_num = args.model.zfill(2)

    for item in data:
        if "No face detected" in item:
            temp = (
                item.split(" ")[-1].replace("\n", "").replace(f"M{model_num}/objs/", "")
            )
            no_list.append(temp)
        elif "Processing" in item:
            pass
        else:
            pass
            # print(item)

    no_list.sort()

    print("No face detected in:", no_list)
    print("Total: ", len(no_list))
    with open(f"M{model_num}_no_face.txt", "w") as f:
        for item in no_list:
            f.write("%s\n" % item.replace(".png", ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log", help="Log file path, eval.txt")
    parser.add_argument("-m", "--model", help="Model Number")
    args = parser.parse_args()

    main(args)
