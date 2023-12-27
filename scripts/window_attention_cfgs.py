# TODO: define all window attention configurations in this .py file


vitb16_ss16 = {i: dict(window_size=16,     # 1/4 for 1024
                       shift=0) for i in [0, 1, 3, 4, 6, 7, 9, 10]}


vitl14_ss16 = {i: dict(window_size=16,    # 1/4 for 896
                       shift=0) for i in list(range(0, 5)) + list(range(6, 11)) \
                                        + list(range(12, 17)) + list(range(18, 23))}

vitb16_shift8_ss16 = {i: dict(window_size=16, shift=0)
                      for i in [0, 3, 6, 9]}

vitb16_shift8_ss16.update({i: dict(window_size=16, shift=8)
                           for i in [1, 4, 7, 10]})
