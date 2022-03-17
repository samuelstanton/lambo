import torch
import random

from lambo.transforms import SequenceTranslation


def test_seq_translation():
    translate_op = SequenceTranslation(max_shift=16)
    base_x = torch.tensor([-1, 0, 1, 2, -1])

    # no shift
    shift = 0
    rot_x = translate_op(base_x, shift)
    assert torch.all(rot_x == base_x), f"{shift}: {base_x} --> {out_x} != {rot_x}"
    shift = 3
    rot_x = translate_op(base_x, shift)
    assert torch.all(rot_x == base_x), f"{shift}: {base_x} --> {out_x} != {rot_x}"

    # right shift
    shift = 1
    rot_x = torch.tensor([-1, 2, 0, 1, -1])
    out_x = translate_op(base_x, shift)
    assert torch.all(out_x == rot_x), f"{shift}: {base_x} --> {out_x} != {rot_x}"
    shift = 4
    out_x = translate_op(base_x, shift)
    assert torch.all(out_x == rot_x), f"{shift}: {base_x} --> {out_x} != {rot_x}"

    # left shift
    shift = -1
    rot_x = torch.tensor([-1, 1, 2, 0, -1])
    out_x = translate_op(base_x, shift)
    assert torch.all(out_x == rot_x), f"{shift}: {base_x} --> {out_x} != {rot_x}"
    shift = -4
    out_x = translate_op(base_x, shift)
    assert torch.all(out_x == rot_x), f"{shift}: {base_x} --> {out_x} != {rot_x}"

    # random shift
    rot_x = torch.tensor([-1, 1, 2, 0, -1])
    random.seed(0)
    out_x = translate_op(base_x)
    assert torch.all(out_x == rot_x)


# TODO
def test_str_to_long_tensor():
    pass
