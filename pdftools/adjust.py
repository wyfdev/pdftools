from re import split as re_split, sub as re_sub, match as re_match
from sys import stderr
from pathlib import Path
from contextlib import nullcontext

from typing import Tuple, List

import cv2
import numpy as np

from tqdm import tqdm
from pymupdf import open as pdf_open
from pymupdf import Pixmap, csGRAY
from pymupdf import Rect, Page

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as wait_futures


def adjust_pdf_margin_manual(src: str, dst: str, plan_text=None, plan_file=None, verbose=False):
    # file
    srcfile = Path(src)
    dstfile = Path(dst)

    if not srcfile.expanduser().exists():
        print(f'No such a file "{srcfile}"', file=stderr)
        exit(1)

    if plan_file:
        with open(plan_file) as f:
            plan_text = re_split(r'\s+', f.read())

    def parse_plan(secs: list[str], end_page_num: int) -> dict[int, float]:
        """parse plan text to dict in format of {page_num: movex}
           secs iterate are in format of:
            - page_num=movex:   "10=5"       -> {10: 5}
            - page_range=movex: "10-12=3"    -> {10: 3, 11: 3, 12: 3}
            - page_range~movex: "10-12~3"    -> {10: 3, 11:-3, 12: 3}
            - page_range~movex: "10-end~3"   -> {10: 3, 11:-3, ..., end_page_num: 3}
            - page_range~movex: "10-end-2~3" -> {10: 3, 11:-3, ..., end_page_num-2: 3}
        """
        plan: dict[int, float] = {}
        for sec in secs:
            if not sec:
                continue
            if 'end' in sec:
                if (m := re_match(r'^(\d+)-(end-(\d+))[=~].+$', sec)):
                    s, t, m = m.groups() # type: ignore # start, tag, minus: ('100', 'end-5', '5')
                    end_page_num = max(int(s), end_page_num-int(m))
                    sec = sec.replace(t, str(end_page_num))
                else:
                    sec = sec.replace('end', str(end_page_num))
            p = r'^(\d+)(-)?(\d+)?(=|~)(-?\d+(?:\.\d+)?)$'
            sec = re_sub(p, r'\1 \2 \3 \4 \5', sec)
            sec = re_split(r'\s+', sec)  # type: ignore
            match sec:
                case (n, '=', x):
                    plan[int(n)] = float(x)
                case (s, '-', e, '=', x):
                    for n in range(int(s), int(e)+1):
                        plan[n] = float(x)
                case (s, '-', e, '~', x):
                    for i, n in enumerate(range(int(s), int(e)+1)):
                        plan[n] = float(x) * (-1 if i%2 else 1)
        return plan

    pdf = pdf_open(srcfile)

    plan = parse_plan(plan_text, end_page_num=len(pdf))

    if not plan:
        print('No plan given.')
        return

    # iterate each pages
    for page_num, movex_str in tqdm(plan.items(), desc='adjusting : '):
        movep = float(movex_str) / 100
        page = pdf.load_page(int(page_num)-1)
        data = vars(page.mediabox)
        movex = float(data['x1'] - data['x0']) * movep
        data['x0'] -= movex
        data['x1'] -= movex
        if verbose:
            print(f'  {page_num:>3} >> {movep:>5} : {data}')
        page.set_mediabox(Rect(**data))

    # save to file
    print('saveing   : ', end='')
    pdf.save(dstfile)
    print('Done')




def adjust_pdf_margin_auto(src: str, dst: str, x_threshold=0.05, skips=None):

    def parse_range(skips: str):
        the_range: List[int] = []
        for part in skips.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                the_range.extend(range(start - 1, end))
            else:
                the_range.append(int(part) - 1)

        class SkipRange:
            def __contains__(self, item):
                return item in the_range
        return SkipRange()

    skips = parse_range(skips) if skips else list()

    srcfile = Path(src)
    dstfile = Path(dst)
    pdf = pdf_open(srcfile)

    def image_convert(imgarr: np.array) -> np.array:
        # >> thresholding
        thresh, imgarr = cv2.threshold(imgarr, 210, 255, cv2.THRESH_BINARY)

        # >> sharping
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        imgarr = cv2.filter2D(imgarr, -1, kernel)

        return imgarr

    def cal_page_movex(page) -> Tuple[float, float]:

        class ClifNotFoundError(Exception):
            pass

        def cal_margin(imgarr: np.array) -> tuple[float, float]:
            """
            return: (left_clif, right_clif) in persentage
            """
            shape_h, shape_w = imgarr.shape

            # down sampling for calculate
            tmp_w = 200
            tmp_h = (tmp_w * shape_h) // shape_w
            tmparr = cv2.resize(imgarr, (tmp_w, tmp_h), interpolation=cv2.INTER_AREA)
            tmparr = cv2.bitwise_not(tmparr)

            def cov_threshold(img):
                _, img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
                return img
            tmparr = cov_threshold(tmparr)

            def cov_mask_characters(img, thresh=220):
                contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Iterate over the contours and draw a rectangle around each character.
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(img , (x, y), (x + w, y + h), 255, -1)
                return img
            tmparr = cov_mask_characters(tmparr)

            def cal_weight_x():
                x_axis_weight = np.zeros(tmp_w, np.float32)
                h = tmparr.shape[0]
                for x in range(tmp_w):  # left -> right
                    x_axis_weight[x] = \
                        sum((1 if i>220 else 0) for i in tmparr[:, x]) / h
                return x_axis_weight
            x_axis_weight = cal_weight_x()

            def cal_clif_l():
                # from center to left |----- <-* -----|, stop at weight clif
                for i in range(int(tmp_w * (1/3)), 0, -1):
                    if np.mean(x_axis_weight[max(0, i-3): i]) < x_threshold:
                        return i

            def cal_clif_r():
                # from center to right |----- *-> -----|, stop at weight clif
                for i in range(int(tmp_w * (2/3)), tmp_w):
                    if np.mean(x_axis_weight[i: min(tmp_w, i+3)]) < x_threshold:
                        return i

            def cal_clif_r_by_r_to_l():
                # from right to center |---------- <-*|, stop at weight clif
                for i in range(tmp_w, int(tmp_w * (2/3)), -1):
                    if np.mean(x_axis_weight[max(0, i-3): i]) > x_threshold:
                        return i

            try:
                mlp = cal_clif_l() / tmp_w
                mrp = cal_clif_r() / tmp_w

                if mlp > 0.35 and mrp < 0.65:
                    raise ClifNotFoundError('clif found been extream', (mlp, mrp))

                return mlp, mrp
            except Exception as err:
                # print('clif not found')
                return (0, 0)
                # raise ClifNotFoundError('clif not found')

        def cal_movex(page, mlp: float, mrp: float):
            shape_h, shape_w = imgarr.shape

            movex = (((mlp + mrp) / 2) - 0.5)     # round()
            if abs(movex) > 0.2:
                movex = 0

            return movex

        try:
            pix = page.get_pixmap()
            # convert to gray
            pix_gray = Pixmap(csGRAY, pix)
            # convert to array
            imgarr = np.frombuffer(pix_gray.samples, dtype=np.uint8) \
                .reshape(pix.height, pix.width)

            mlp, mrp = cal_margin(imgarr)
            movex = cal_movex(page, mlp, mrp)
            return movex
        except Exception as err:
            raise err
            # return (0, 0)

    def do_page_adjust(page, movex: float):
        data = vars(page.mediabox)
        move = float(data['x1'] - data['x0']) * movex
        data['x0'] += move
        data['x1'] += move
        page.set_mediabox(Rect(**data))

    #
    # iterate each pages
    #
    with nullcontext('scan file'), tqdm(total=len(pdf), desc='scanning  ') as pbar:
        plan = dict()

        def plan_page_adjusting(page_id):
            if page_id not in skips:
                page = pdf.load_page(page_id)
                movex = cal_page_movex(page)
                plan[page_id] = movex
            else:
                plan[page_id] = 0
            pbar.update(1)

        with ThreadPoolExecutor(max_workers=None) as executor:
            wait_futures([executor.submit(plan_page_adjusting, page_id)
                            for page_id in range(len(pdf))])

    with nullcontext('adjusting'):
        # waiting for all tasks
        for id, movex in tqdm(plan.items(), desc='adjusting '):
            page = pdf.load_page(id)
            do_page_adjust(page, movex)

    with nullcontext('saveing'):
        print('saveing   : ', end='')
        pdf.save(dstfile)
        print('Done')