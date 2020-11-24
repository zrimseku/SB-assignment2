import cv2
import os


def positive_description_file():
    info = {}
    for photo in os.listdir('AWEForSegmentation/train'):
        mask = cv2.imread(f'./AWEForSegmentation/trainannot_rect/{photo}', 0).astype(bool)
        info[photo] = []
        unfinished = {}
        i = 0
        while i < mask.shape[0]:
            j = 0
            while j < mask.shape[1]:
                if mask[i, j]:
                    if i == 0 or not mask[i-1, j]:                  # upper row
                        if j == 0 or not mask[i, j-1]:              # upper left corner
                            unfinished[j] = {'start_x': i}          # there can only be one unfinished on y=j
                            width = 0
                            while j < mask.shape[1] and mask[i, j]:
                                j += 1
                                width += 1
                            unfinished[j-width]['width'] = width
                    elif mask[i-1, j] and mask[i+1, j]:             # left column
                        j += unfinished[j]['width']
                    elif mask[i-1, j] and not mask[i+1, j]:         # lower left corner
                        height = i - unfinished[j]['start_x'] + 1
                        info[photo].append([unfinished[j]['start_x'], j, unfinished[j]['width'], height])
                        j += unfinished[j]['width']
                    else:
                        print(photo, i, j)
                else:
                    j += 1
            i += 1
    with open('info.dat', 'w') as f:
        for photo in info:
            entry = f"AWEForSegmentation/train/{photo}  {len(info[photo])}  "
            for el in info[photo]:
                entry += ' '.join(map(str, el)) + "   "
            f.write(entry.strip() + "\n")
    return info


def prepare_negative_from_positive():
    info = []
    for photo in os.listdir('AWEForSegmentation/train'):
        mask = cv2.imread(f'./AWEForSegmentation/trainannot_rect/{photo}', 0).astype(bool)
        negative = cv2.imread(f'./AWEForSegmentation/train/{photo}', cv2.IMREAD_COLOR)
        negative[mask] = (0, 0, 0)
        cv2.imwrite(f'AWEForSegmentation/negative/n_{photo}', negative)
        info.append(f'AWEForSegmentation/negative/n_{photo}')
        negative[mask] = (255, 255, 255)
        cv2.imwrite(f'AWEForSegmentation/negative/n_1{photo[1:]}', negative)
        info.append(f'AWEForSegmentation/negative/n_1{photo[1:]}')
    with open('bg.txt', 'w') as f:
        for line in info:
            f.write(line + "\n")


def repair_mask_0417():
    """Mask of photo 0417.png is of size (360, 481) and photo is (360, 480)"""
    mask = cv2.imread(f'./AWEForSegmentation/trainannot_rect/0417.png', -1)
    new_mask = mask[:, :480]
    cv2.imwrite(f'AWEForSegmentation/trainannot_rect/0417.png', new_mask)

