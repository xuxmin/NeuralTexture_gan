from collections import Iterable
import numpy as np

def map_light_8to16(light_idxs):
    if isinstance(light_idxs, int):
        bid = light_idxs // 8
        ret = bid * 2 + 1 if light_idxs % 4 >= 2 else bid * 2
    else:
        assert isinstance(light_idxs, Iterable)
        ret = []
        for light_idx in light_idxs:
            bid = light_idx // 8
            ret.append(bid * 2 + 1 if light_idx % 4 >= 2 else bid * 2)

    return ret

def downsample_light(light_idxs, size_src=1, size_dst=8):
    assert 16 % size_src == 0 and 16 % size_dst == 0
    if size_src >= size_dst:
        return light_idxs

    lights_per_board = (32 * 16 // (size_src * size_src))
    lights_per_board_ds = (32 * 16 // (size_dst * size_dst))
    if isinstance(light_idxs, int):
        bid = light_idxs // lights_per_board
        row = light_idxs % lights_per_board
        col = row % (32 // size_src) * size_src // size_dst
        row = row // (32 // size_src) * size_src // size_dst
        ret = bid * lights_per_board_ds + row * (32 // size_dst) + col
    else:
        assert isinstance(light_idxs, Iterable)
        ret = []
        for light_idx in light_idxs:
            bid = light_idx // (32 * 16 // (size_src * size_src))
            row = light_idx % lights_per_board
            col = row % (32 // size_src) * size_src // size_dst
            row = row // (32 // size_src) * size_src // size_dst
            tmp = bid * lights_per_board_ds + row * (32 // size_dst) + col
            ret.append(tmp)

    return ret



if __name__ == "__main__":
    test = range(384)
    # print(downsample_light(test, 1, 16))
    print(map_light_8to16(test))
