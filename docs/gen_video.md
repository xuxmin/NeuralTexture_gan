

```py
import cv2


def main():
    fps = 10
    size = (518, 260)
    img_dir = 'D:/Code/SRTP/animal-pose-estimation/output/fish/pose_resnet_50/pose_net_batch2_adam_lr1e-3/'
    videowriter = cv2.VideoWriter(
        "output.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

    for i in range(1, 99):
        print(i)
        img = cv2.imread(img_dir + 'val_{}_0_pred.jpg'.format(i))
        if img is None:
            raise FileExistsError
        videowriter.write(img)

    videowriter.release()
    print("done")


if __name__ == "__main__":
    main()
```